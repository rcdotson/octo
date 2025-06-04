"""
This script demonstrates how to finetune Octo to a new observation space (single camera + proprio)
and new action space (bimanual) using a simulated ALOHA cube handover dataset (https://tonyzhaozh.github.io/aloha/).

To run this example, first download and extract the dataset from here: https://rail.eecs.berkeley.edu/datasets/example_sim_data.zip

python examples/02_finetune_new_observation_action.py --pretrained_path=hf://rail-berkeley/octo-small-1.5 --data_dir=...
"""
from absl import app, flags, logging
import flax
import jax
import optax
import tensorflow as tf
import tqdm
import wandb

from octo.data.dataset import make_single_dataset
from octo.data.utils.data_utils import NormalizationType
from octo.model.components.action_heads import L1ActionHead, DiffusionActionHead
from octo.model.components.tokenizers import LowdimObsTokenizer
from octo.model.octo_model import OctoModel
from octo.utils.jax_utils import initialize_compilation_cache
from octo.utils.spec import ModuleSpec
from octo.utils.train_utils import (
    freeze_weights,
    merge_params,
    process_text,
    TrainState,
)
from octo.utils.train_callbacks import (
    RolloutVisualizationCallback,
    SaveCallback,
    ValidationCallback,
    VisualizationCallback,
)

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "pretrained_path", None, "Path to pre-trained Octo checkpoint directory."
)
flags.DEFINE_string("data_dir", None, "Path to finetuning dataset, in RLDS format.")
flags.DEFINE_string("save_dir", None, "Directory for saving finetuning checkpoints.")
#flags.DEFINE_integer("batch_size", 256, "Batch size for finetuning.  256 for 0-image workloads works")
flags.DEFINE_integer("batch_size", 64, "Batch size for finetuning.  32 for 2-image workloads seems to work")

"""flags.DEFINE_bool(
    "freeze_transformer",
    True,
    "Whether pre-trained transformer weights should be frozen.",
)"""

ACTION_DIMS = 8
ACTION_HORIZON = 8
WINDOW_SIZE = 1
WANDB_PROJECT = "octo_jan"
DATASET_NAME = "pickup_blue_can8hz_dataset:1.0.0"
ITERATIONS = 20000
SAVE_ITERATIONS=1000
DIFFUSION_MODEL=True

def main(_):
    assert (
        FLAGS.batch_size % jax.device_count() == 0
    ), "Batch size must be divisible by device count."

    initialize_compilation_cache()
    # prevent tensorflow from using GPU memory since it's only used for data loading
    tf.config.set_visible_devices([], "GPU")

    name = FLAGS.save_dir.split("/")[-1]

    # setup wandb for logging
    wandb.init(name=name, project=WANDB_PROJECT)

    # load pre-trained model
    logging.info("Loading pre-trained model...")
    pretrained_model = OctoModel.load_pretrained(FLAGS.pretrained_path)

    # make finetuning dataset
    # apply Gaussian normalization, load chunks of 50 actions since we'll train with action chunking
    # delete goal images in the data loader since we will train a language-conditioned-only policy
    # TODO: directly load this from raw data to make it less opaque?
    logging.info("Loading finetuning dataset...")

    dataset_kwargs=dict(
            name=DATASET_NAME,
            data_dir=FLAGS.data_dir,
            image_obs_keys= {"primary": "image", "wrist": "image_wrist_1"},
            #image_obs_keys= {"wrist": "image_wrist_1"},
            #depth_obs_keys={"primary": "image_depth", "wrist": "image_wrist_2_depth"},
            #proprio_obs_key="state",            
            language_key="language_instruction",
            action_proprio_normalization_type=NormalizationType.NORMAL,
            action_normalization_mask=[True, True, True, True, True, True, False, False],
            #absolute_action_mask=[True] * 7,       
            standardize_fn = ModuleSpec.create("octo.data.oxe.oxe_standardization_transforms:curve_dataset_transform")
            )
    
    dataset = make_single_dataset(
        dataset_kwargs=dataset_kwargs,
        traj_transform_kwargs=dict(
            window_size=WINDOW_SIZE,
            action_horizon=ACTION_HORIZON,
        ),
        frame_transform_kwargs=dict(
            resize_size={"primary": (256, 256), "wrist": (128, 128)}
            #resize_size={"wrist": (256, 256)},
            
        ),
          train=True,
    )
    train_data_iter = (
        dataset.repeat()
        .unbatch()
        .shuffle(10000)  # can reduce this if RAM consumption too high
        .batch(FLAGS.batch_size)
        .iterator()
    )

    # run text tokenizer over batch (this needs to happen before training / sharding) + delete unused keys
    text_processor = pretrained_model.text_processor

    def process_batch(batch):
        batch = process_text(batch, text_processor)
        del batch["dataset_name"]
        return batch

    train_data_iter = map(process_batch, train_data_iter)
    example_batch = next(train_data_iter)

    # load pre-training config and modify --> remove wrist cam, add proprio input, change action head
    # following Zhao et al. we use "action chunks" of length 50 and L1 loss for ALOHA
    config = pretrained_model.config

    #del config["model"]["observation_tokenizers"]["wrist"]  

    ###r
    config["model"]["observation_tokenizers"]["proprio"] = ModuleSpec.create(
        LowdimObsTokenizer,
        n_bins=256,
        bin_type="normal",
        low=-2.0,
        high=2.0,
        obs_keys=["proprio"],
    )
    if not DIFFUSION_MODEL:
        # Fully override the old action head with a new one (for smaller changes, you can use update_config)
        config["model"]["heads"]["action"] = ModuleSpec.create(
            L1ActionHead,
            action_horizon=ACTION_HORIZON,
            action_dim=ACTION_DIMS,
            readout_key="readout_action",
        )
    else:
        config["model"]["heads"]["action"] = ModuleSpec.create(
            DiffusionActionHead,
            action_horizon=ACTION_HORIZON,
            action_dim=ACTION_DIMS,
            use_map=True,
            loss_type = "l1",
            readout_key="readout_action",
        )

    # initialize weights for modified Octo model, then merge in all applicable pre-trained weights
    # new position encodings for proprio inputs & weights for new action head will remain "from scratch"
    logging.info("Updating model for new observation & action space...")
    model = OctoModel.from_config(
        config,
        example_batch,
        text_processor,
        verbose=True,
        dataset_statistics=dataset.dataset_statistics,
    )
    merged_params = merge_params(model.params, pretrained_model.params)
    # can perform any additional parameter surgery here...
    # ...
    model = model.replace(params=merged_params)
    del pretrained_model

    # create optimizer & train_state, optionally freeze keys for pre-trained transformer
    # train_state bundles parameters & optimizers
    learning_rate = optax.join_schedules(
        [optax.linear_schedule(0, 6e-5, 100), optax.constant_schedule(6e-5)], [100]
    )
    tx = optax.adamw(learning_rate)
    frozen_keys = model.config["optimizer"]["frozen_keys"]
    frozen_keys.append("BlockTransformer_0")
    #frozen_keys = {}
    tx = freeze_weights(tx, model.params, frozen_keys)
    train_state = TrainState.create(
        rng=jax.random.PRNGKey(1234),
        model=model,
        tx=tx,
    )

    # define loss function and train step
    def loss_fn(params, batch, rng, train=True):
        bound_module = model.module.bind({"params": params}, rngs={"dropout": rng})
        transformer_embeddings = bound_module.octo_transformer(
            batch["observation"],
            batch["task"],
            batch["observation"]["timestep_pad_mask"],
            train=train,
        )
        action_loss, action_metrics = bound_module.heads["action"].loss(
            transformer_embeddings,  # Action head knows to pull out the action readout_key
            batch["action"],
            batch["observation"]["timestep_pad_mask"],
            batch["action_pad_mask"],
            train=train,
        )
        return action_loss, action_metrics

    def wandb_log(info, step):
        wandb.log(flax.traverse_util.flatten_dict(info, sep="/"), step=step)

    modes_to_evaluate = ["text_conditioned"]
    dataset_kwargs_list = [dataset_kwargs]

    val_dataset_kwargs = dict(
        traj_transform_kwargs=dict(
            window_size=WINDOW_SIZE,
            action_horizon=ACTION_HORIZON,
        ),
        frame_transform_kwargs=dict(
        resize_size={"primary": (256, 256), "wrist": (128, 128)},
        #depth_resize_size={"primary": (256, 256), "wrist": (256, 256)},
        ),
        batch_size=24
    )
    
    val_callback = ValidationCallback(
            loss_fn=loss_fn,
            process_batch_fn=process_batch,
            text_processor=text_processor,
            val_dataset_kwargs_list=dataset_kwargs_list,
            dataset_kwargs=val_dataset_kwargs, #FLAGS.config,
            modes_to_evaluate=modes_to_evaluate,
            #**FLAGS.config.val_kwargs,
            val_shuffle_buffer_size=5000,
            num_val_batches=1000
        )
    @jax.jit
    def train_step(state, batch):
        rng, dropout_rng = jax.random.split(state.rng)
        (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.model.params, batch, dropout_rng, train=True
        )
        new_state = state.apply_gradients(grads=grads, rng=rng)
        return new_state, info

    # run finetuning loop
    logging.info("Starting finetuning...")
    iterations = ITERATIONS+1
    for i in tqdm.tqdm(range(iterations), total=iterations, dynamic_ncols=True):
        batch = next(train_data_iter)
        #print(batch["observation"]["image_primary"][0])
        train_state, update_info = train_step(train_state, batch)
        if (i + 1) % 100 == 0:
            update_info = jax.device_get(update_info)
            wandb.log(
                flax.traverse_util.flatten_dict({"training": update_info}, sep="/"),
                step=i,
            )
        if (i + 1) % SAVE_ITERATIONS == 0:
            val_metrics = val_callback(train_state, i + 1)
            wandb_log(val_metrics, step=i)
            
        if (i % SAVE_ITERATIONS == 0) and i>0:
            # save checkpoint
            train_state.model.save_pretrained(step=i, checkpoint_path=FLAGS.save_dir)
    


if __name__ == "__main__":
    app.run(main)