import argparse
import os

def parse_all_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")

    parser.add_argument("--run_name", type=str, default='', help="Run name")

    # item detail options
    parser.add_argument(
        "--use_item_details",
        action="store_true",
        default=False,
        help="Whether to use per-item textual details (polyvore_item_des.npy)."
    )
    parser.add_argument(
        "--item_description_path",
        type=str,
        default="../datasets/polyvore/polyvore_item_des.json",
        help="Path to json file that contains per-item description lists."
    )
    parser.add_argument(
        "--description_prompt_weight",
        type=float,
        default=0.3,
        help="Weight for the description CLS when combining with category CLS."
    )
    parser.add_argument(
        "--category_prompt_weight",
        type=float,
        default=0.7,
        help="Weight for the category CLS when combining with description CLS."
    )
    # GNN-specific hyperparameters
    parser.add_argument(
        "--use_gnn",
        action="store_true",
        help="Enable the Outfit Compatibility GAT module during training."
    )
    parser.add_argument(
        "--pretrained_evaluator_ckpt",
        type=str,
        default="./compatibility_evaluator/ifashion-ckpt/ifashion_evaluator.pth",
        help="Path to the pretrained compatibility evaluator model checkpoint."
    )
    parser.add_argument(
        "--max_history_per_slot",
        type=int,
        default=1000,
        help="Max history items per slot for GAT graph construction."
    )
    
    # 新增：GNN 微调相关参数
    parser.add_argument(
        "--gat_learning_rate",
        type=float,
        default=1e-4, # 默认给一个很小的学习率，防止破坏预训练
        help="Learning rate for GAT when unfreezing."
    )

    # [NEW] Option to freeze backbone for attention-only tuning (Protects image quality)
    parser.add_argument("--freeze_unet_backbone", action="store_true", default=True, 
                        help="Freeze UNet ResNet blocks...")

    # [NEW] Load pre-trained DiFashion weights (Resume from Stage 1)
    parser.add_argument("--difashion_checkpoint_path", type=str, default="./difashion_ifashion_checkpoint/checkpoint-15000", 
                        help="Path to a trained DiFashion checkpoint folder (e.g., '.../checkpoint-50000') containing 'unet' and 'fashion_encoder' subfolders.")

    # 基础超参数
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="./stable-diffusion-2-base",
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default='../datasets/ifashion',
        # default='../datasets/polyvore',
        help="A folder containing the dataset for training and inference."
    )
    parser.add_argument(
        "--item_features_path",
        type=str,
        default="../datasets/ifashion/all_item_features.npy",
        help="Path to the item features file."
    )
    parser.add_argument(
        '--img_folder_path',
        default='../semantic_category/'
        # default='../polyvore_folder_path/291x291'
    )
    parser.add_argument(
        "--data_processed",
        type=bool,
        default=True,
        help="if the data is processed or not."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default='',
        help="The name of the Dataset for training and inference."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output_train_gat/polyvore/difashion_gat_ug_v5_stage3",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=".cache/",
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=123, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        default=False,
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--use_mutual_guidance",
        type=bool,
        default=True
    )
    parser.add_argument(
        "--use_history",
        type=bool,
        default=True
    )
    parser.add_argument(
        "--conditioning_dropout_prob",
        type=float,
        default=0.2,
        help="Conditioning dropout probability.",
    )
    parser.add_argument(
        "--coupling_dropout_prob",
        type=float,
        default=0.3,
        help="Coupling dropout probability.",
    )
    parser.add_argument(
        "--cate_conditioning_dropout_prob",
        type=float,
        default=0.5,
        # default=0.2,
        help="Category conditioning dropout probability.",
    )
    parser.add_argument(
        "--category_emb_size",
        type=int,
        default=64,
        help="Fashion item category embedding size.",
    )
    parser.add_argument(
        "--hid_dim",
        type=int,
        default=256,
        help="Fashion encoder hidden dim."
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.1,
        help="The weight of mutual guidance."
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=5.0,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. ",
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs."
        ),
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument("--use_ema_fashion", action="store_true", help="Whether to use EMA model for fashion encoder.")
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier."
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="./logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=100,
        help=(
            "Save a checkpoint of the training state every X updates."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=(
            "Max number of checkpoints to store."
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default="latest",
        help=(
            "Whether training should be resumed from a previous checkpoint."
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="fashion_outfit_generation",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers"
        ),
    )
    
    parser.add_argument("--category_guidance_scale", type=float, default=12.0) 
    parser.add_argument("--hist_guidance_scale", type=float, default=4.0)
    parser.add_argument("--mutual_guidance_scale", type=float, default=5.0)
    parser.add_argument("--num_inference_steps", type=int, default=50)

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    args.output_dir = os.path.join(args.output_dir, args.run_name)
 
    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args