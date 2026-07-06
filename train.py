"""Script to fine-tune Stable Diffusion for Fashion Outfit Generation and Recommendation (GAT Integrated)"""

import argparse
import logging
import math
import os
import sys

import gc

import accelerate
import datasets
import numpy as np
from PIL import Image
from per_DiFashion.train_difashion_gat_ema import freeze_unet_backbone
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torchvision import transforms
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from packaging import version
from tqdm.auto import tqdm

import diffusers
from diffusers import UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
import open_clip
import data_utils
from models.model_components import MutualEncoder, ClipFeatureExtractor
from models.GRDFRec import DiFashion
from args import parse_all_args
from torch_geometric.data import Data, Batch
from torch.utils.tensorboard import SummaryWriter
from metrics_utils import calculate_metrics_online
from graph_utils import get_item_feature, build_batch_graph, freeze_unet_backbone




# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.16.0")


logger = get_logger(__name__, log_level="INFO")

def main():
    args = parse_all_args()
    
    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )
    
    if args.report_to == "wandb":
        if is_wandb_available():
            import wandb
            wandb.init(project="difashion")
        else:
            args.report_to = "tensorboard"

    logging_dir = args.logging_dir
    accelerator_project_config = ProjectConfiguration(total_limit=args.checkpoints_total_limit, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    device = accelerator.device
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)
    
    # ... (data loading: train_dict, history, maps etc.) ...
    logger.info("Data loading......")
    data_path = os.path.join(args.data_path, args.dataset_name)
    train_dict = np.load(os.path.join(data_path, "train.npy"), allow_pickle=True).item()
    valid_fitb_dict = np.load(os.path.join(data_path, "fitb_valid.npy"), allow_pickle=True).item()
    train_history = np.load(os.path.join(data_path, "train_history.npy"), allow_pickle=True).item()
    valid_history = np.load(os.path.join(data_path, "valid_history.npy"), allow_pickle=True).item()
    valid_grd_dict = np.load(os.path.join(data_path, "valid_grd.npy"), allow_pickle=True).item()
    new_id_cate_dict = np.load(os.path.join(data_path, "id_cate_dict.npy"), allow_pickle=True).item()
    all_image_paths = np.load(os.path.join(data_path, "all_item_image_paths.npy"), allow_pickle=True)
    item_features_bank = np.load(os.path.join(data_path, "cnn_features_clip.npy"), allow_pickle=True)

    texture_feat_path = os.path.join(data_path, "processed", "texture_features_clip.npy")
    if os.path.exists(texture_feat_path):
        logger.info(f"Loading texture features from {texture_feat_path}...")
        # 注意：我们存的是字典，所以要用 .item() 取出
        texture_features_bank = np.load(texture_feat_path, allow_pickle=True).item()
        logger.info(f"Loaded {len(texture_features_bank)} texture entries.")
    else:
        logger.warning("Texture features not found! Will use zero padding.")
        texture_features_bank = {}
        

    img_trans = transforms.Compose([
        transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
        transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
        transforms.ToTensor()
    ])
    img_dataset = data_utils.ImagePathDataset(args.img_folder_path, all_image_paths, img_trans, do_normalize=True)
    null_img = img_dataset[0].to(device)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    logger.info("Build the diffusion model (DiFashion Integrated)...")
    diffusion = DiFashion(args, logger, len(new_id_cate_dict), device)
    logger.info("Completed.")

    if args.difashion_checkpoint_path:
        logger.info(f"\n{'='*40}\nLoading Stage 1 DiFashion weights from: {args.difashion_checkpoint_path}\n{'='*40}")
        
        # 1. Load UNet (覆盖 SD 原始权重)
        unet_path = os.path.join(args.difashion_checkpoint_path, "unet")
        if os.path.exists(unet_path):
            logger.info(f"Loading UNet from {unet_path}...")
            try:
                # [FIX] Force use_safetensors=False to avoid error
                temp_unet = UNet2DConditionModel.from_pretrained(unet_path, use_safetensors=False)
                
                # Check channel dimensions
                if temp_unet.config.in_channels != diffusion.unet.config.in_channels:
                    logger.warning(f"[Warning] Channel mismatch! Loaded: {temp_unet.config.in_channels}, Current: {diffusion.unet.config.in_channels}.")
                
                diffusion.unet.load_state_dict(temp_unet.state_dict(), strict=False)
                logger.info("UNet weights loaded successfully.")
                
                # [WEIGHT STETHOSCOPE] Check if extra channels are non-zero
                with torch.no_grad():
                    extra_ch_weight = diffusion.unet.conv_in.weight[:, 4:, :, :]
                    extra_mean = extra_ch_weight.mean().item()
                    extra_std = extra_ch_weight.std().item()
                    if extra_std < 1e-6:
                        logger.warning(f"\n[WEIGHT CHECK] ⚠️  Extra channels appear to be ZERO/INIT (std={extra_std:.6f}). Did the checkpoint train them?")
                    else:
                        logger.info(f"\n[WEIGHT CHECK] ✅ Extra channels loaded with non-zero weights (mean={extra_mean:.4f}, std={extra_std:.4f})")

                del temp_unet
            except Exception as e:
                logger.error(f"Failed to load UNet from checkpoint: {e}")
                raise e
        else:
            logger.warning(f"UNet checkpoint not found at {unet_path}")

        # 2. Load Fashion Encoder (互惠编码器)
        fe_path = os.path.join(args.difashion_checkpoint_path, "fashion_encoder")
        if os.path.exists(fe_path):
            logger.info(f"Loading FashionEncoder from {fe_path}...")
            try:
                temp_fe = MutualEncoder.from_pretrained(fe_path, use_safetensors=False)
                diffusion.fashion_encoder.load_state_dict(temp_fe.state_dict())
                logger.info("FashionEncoder weights loaded successfully.")
                del temp_fe
            except Exception as e:
                logger.error(f"Failed to load FashionEncoder: {e}")
                raise e
        else:
            logger.warning(f"FashionEncoder checkpoint not found at {fe_path}")
            
    else:
        logger.warning("\n" + "!"*40)
        logger.warning("NO --difashion_checkpoint_path PROVIDED!")
        logger.warning("Model is using RANDOM Mutual Encoder and BASE SD UNet.")
        logger.warning("!"*40 + "\n")
    
    if args.use_ema:
        ema_unet = EMAModel(diffusion.unet.parameters(), model_cls=UNet2DConditionModel, model_config=diffusion.unet.config)
        if args.difashion_checkpoint_path:
            unet_ema_path = os.path.join(args.difashion_checkpoint_path, "unet_ema")
            if os.path.exists(unet_ema_path):
                logger.info(f"Loading pre-trained UNet EMA from {unet_ema_path}...")
                try:
                    load_model = EMAModel.from_pretrained(unet_ema_path, UNet2DConditionModel)
                    ema_unet.load_state_dict(load_model.state_dict())
                    logger.info("UNet EMA weights loaded successfully.")
                    del load_model
                except Exception as e:
                    logger.warning(f"Failed to load UNet EMA: {e}")
    
    if args.use_ema_fashion:
        ema_encoder = EMAModel(diffusion.fashion_encoder.parameters(), model_cls=MutualEncoder, model_config=diffusion.fashion_encoder.config)
        if args.difashion_checkpoint_path:
            fashion_ema_path = os.path.join(args.difashion_checkpoint_path, "fashion_encoder_ema")
            if os.path.exists(fashion_ema_path):
                logger.info(f"Loading pre-trained FashionEncoder EMA from {fashion_ema_path}...")
                try:
                    load_model = EMAModel.from_pretrained(fashion_ema_path, MutualEncoder)
                    ema_encoder.load_state_dict(load_model.state_dict())
                    logger.info("FashionEncoder EMA weights loaded successfully.")
                    del load_model
                except Exception as e:
                    logger.warning(f"Failed to load FashionEncoder EMA: {e}")


    # [CRITICAL] APPLY FREEZE STRATEGY HERE (MUST BE AFTER LOADING WEIGHTS)
    if args.freeze_unet_backbone:
        freeze_unet_backbone(diffusion.unet)

    # 构建 {path -> [desc...]} 的映射，来自 polyvore_item_des.npy
    def load_item_description_map(desc_path, all_paths):
        import json
        if desc_path is None or not os.path.exists(desc_path):
            logger.warning("No item_description_path provided or file not found. Details will fall back to neutral prompts.")
            return {}
        try:
            mp = {}
            mp_idx = {}
            # Support both .npy array and .json list of dicts
            if desc_path.endswith('.json'):
                with open(desc_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, dict) and "data" in data:
                    entries = data["data"]
                else:
                    entries = data
                for entry in entries:
                    try:
                        idx = int(entry.get("index")) if entry.get("index") is not None else None
                        img_path = entry.get("image_path")
                        annotations = entry.get("annotations", [])
                        if annotations is None:
                            annotations = []
                        if idx is not None:
                            mp_idx[idx] = [str(x) for x in annotations]
                        if img_path is not None:
                            key = str(os.path.basename(img_path))
                            mp[key] = [str(x) for x in annotations]
                    except Exception:
                        continue
            else:
                arr = np.load(desc_path, allow_pickle=True)
                if not isinstance(arr, np.ndarray):
                    logger.warning(f"Description array is not a numpy array: type={type(arr)}")
                    return {}
                n_desc = len(arr)
                n_paths = len(all_paths)
                limit = min(n_desc, n_paths)
                for i in range(limit):
                    key = str(all_paths[i])
                    vals = arr[i]
                    if isinstance(vals, (list, tuple)):
                        mp[key] = [str(x) for x in vals]
                        mp_idx[i] = [str(x) for x in vals]
                    else:
                        vlist = [str(vals)] if vals is not None else []
                        mp[key] = vlist
                        mp_idx[i] = vlist
            return mp, mp_idx
        except Exception as e:
            logger.warning(f"Failed to load descriptions: {e}")
            return {}, {}
            
    # Load item descriptions only when enabled
    if args.use_item_details:
        item_description_map, item_description_idx_map = load_item_description_map(args.item_description_path, all_image_paths)
    else:
        item_description_map, item_description_idx_map = {}, {}

    # with accelerator.main_process_first():
    processed_dir = os.path.join(data_path, "processed")
    os.makedirs(processed_dir, exist_ok=True)

    def _load_obj(p):
        try:
            return np.load(p, allow_pickle=True).item()
        except Exception:
            return None

    pt_train = os.path.join(processed_dir, "train.npy")
    pt_valid = os.path.join(processed_dir, "fitb_valid.npy")
    pt_test = os.path.join(processed_dir, "fitb_test.npy")

    if args.data_processed and os.path.exists(pt_train) and os.path.exists(pt_valid):
        # load preprocessed dicts and optional history latents
        train_data_dict = _load_obj(pt_train) or {}
        valid_data_dict = _load_obj(pt_valid) or {}
        test_data_dict = _load_obj(pt_test)
        train_hist_latents = _load_obj(os.path.join(processed_dir, "train_hist_latents.npy"))
        valid_hist_latents = _load_obj(os.path.join(processed_dir, "valid_hist_latents.npy"))
    else:
        # fallback: run preprocessing and persist results
        train_data_dict, train_hist_latents = data_utils.preprocess_dataset(
            train_dict, data_path, new_id_cate_dict, train_history, img_dataset,
            diffusion.tokenizer, diffusion.vae, device,
            all_item_paths=all_image_paths,
            item_description_map=item_description_map,
            item_description_idx_map=item_description_idx_map,
            use_item_details=args.use_item_details,
        )
        valid_data_dict, valid_hist_latents = data_utils.preprocess_dataset(
            valid_fitb_dict, data_path, new_id_cate_dict, valid_history, img_dataset,
            diffusion.tokenizer, diffusion.vae, device,
            all_item_paths=all_image_paths,
            item_description_map=item_description_map,
            item_description_idx_map=item_description_idx_map,
            use_item_details=args.use_item_details,
        )
        test_data_dict = None
        if test_fitb_dict is not None and test_history is not None:
            test_data_dict, test_hist_latents = data_utils.preprocess_dataset(
                test_fitb_dict, data_path, new_id_cate_dict, test_history, img_dataset,
                diffusion.tokenizer, diffusion.vae, device,
                all_item_paths=all_image_paths,
                item_description_map=item_description_map,
                item_description_idx_map=item_description_idx_map,
                use_item_details=args.use_item_details,
            )

        # persist commonly used processed artifacts
        try:
            np.save(pt_train, train_data_dict)
            np.save(pt_valid, valid_data_dict)
            if test_data_dict is not None:
                np.save(pt_test, test_data_dict)
            np.save(os.path.join(processed_dir, "train_hist_latents.npy"), train_hist_latents)
            np.save(os.path.join(processed_dir, "valid_hist_latents.npy"), valid_hist_latents)
            if test_fitb_dict is not None and test_history is not None:
                np.save(os.path.join(processed_dir, "test_hist_latents.npy"), test_hist_latents)
        except Exception:
            pass
    
    valid_hist_features = None
    processed_dir = os.path.join(data_path, "processed")
    clip_valid_p = os.path.join(processed_dir, "clip_valid_hist_features.npy")
    
    logger.info(f"Checking for validation history features at: {clip_valid_p}")
    
    if os.path.exists(clip_valid_p):
        try:
            valid_hist_features = np.load(clip_valid_p, allow_pickle=True).item()
            logger.info(f"Successfully loaded valid_hist_features from {clip_valid_p} (size: {len(valid_hist_features)})")
        except Exception as e:
            logger.warning(f"Failed to load existing {clip_valid_p}: {e}")
    
    if valid_hist_features is None:
        logger.info("valid_hist_features not found. Trying to build for metrics calculation...")
        # 需要 unwrap model 获取 feature extractor
        unwrapped_model = accelerator.unwrap_model(diffusion)
        feat_extractor = getattr(unwrapped_model, 'image_feature_extractor', None)
        
        if feat_extractor is None:
                if not hasattr(accelerator, '_local_feat_extractor'):

                    accelerator._local_feat_extractor = ClipFeatureExtractor().to(device)

                feat_extractor = accelerator._local_feat_extractor

        if feat_extractor is not None:
            if accelerator.is_main_process:
                logger.info("Building valid_hist_features... (This may take a few minutes)")
                vhf = data_utils.create_feature_history(
                    valid_history, feat_extractor, img_dataset, device
                )
                os.makedirs(processed_dir, exist_ok=True)
                np.save(clip_valid_p, vhf)
                logger.info(f"Built and saved valid_hist_features to {clip_valid_p}")
            
            accelerator.wait_for_everyone()
            
            valid_hist_features = np.load(clip_valid_p, allow_pickle=True).item()
        else:
            logger.warning("Cannot build history features: Model missing 'image_feature_extractor'.")

    logger.info(f"use_gnn={args.use_gnn}")

    # Safe collate: gracefully skip optional None fields
    def safe_collate(batch_list):
        from torch.utils.data._utils.collate import default_collate as torch_default_collate
        keys = set()
        for b in batch_list:
            keys.update(b.keys())
        out = {}
        for k in keys:
            vals = [b.get(k, None) for b in batch_list]
            if all(v is None for v in vals):
                continue
            if (k.startswith("details") or k.startswith("details_")) and any(v is None for v in vals):
                continue
            if any(v is None for v in vals):
                raise TypeError(f"Required field '{k}' contains None in the batch.")
            out[k] = torch_default_collate(vals)
        return out

    train_dataset = data_utils.FashionDiffusionData(train_data_dict)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, 
        shuffle=True, 
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        collate_fn=safe_collate,
    )

    valid_dataset = data_utils.FashionDiffusionData(valid_data_dict)
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset, 
        shuffle=False, 
        batch_size=3,
        num_workers=args.dataloader_num_workers,
        collate_fn=safe_collate,
    )
    logger.info("dataloader built.")

    # Create EMA for the unet.
    if args.use_ema:
        ema_unet = EMAModel(diffusion.unet.parameters(), model_cls=UNet2DConditionModel, model_config=diffusion.unet.config)
    
    if args.use_ema_fashion:
        ema_encoder = EMAModel(diffusion.fashion_encoder.parameters(), model_cls=MutualEncoder, model_config=diffusion.fashion_encoder.config)
    
    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if args.use_ema:
                ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))
            if args.use_ema_fashion:
                ema_encoder.save_pretrained(os.path.join(output_dir, "fashion_encoder_ema"))

            for i, model in enumerate(models):
                model.fashion_encoder.save_pretrained(os.path.join(output_dir, "fashion_encoder"))
                model.unet.save_pretrained(os.path.join(output_dir, "unet"))
            

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DConditionModel)
                ema_unet.load_state_dict(load_model.state_dict())
                ema_unet.to(device)
                del load_model
            
            if args.use_ema_fashion:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "fashion_encoder_ema"), MutualEncoder)
                ema_encoder.load_state_dict(load_model.state_dict())
                ema_encoder.to(device)
                del load_model

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()
                load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                model.unet.register_to_config(**load_model.config)
                model.unet.load_state_dict(load_model.state_dict())
                del load_model

                # load mutual encoder into model
                load_model = MutualEncoder.from_pretrained(input_dir, subfolder="fashion_encoder")
                model.fashion_encoder.register_to_config(**load_model.config)
                model.fashion_encoder.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)



    logger.info("Building optimizer with Differential Learning Rates...")
    

    unet_params = [p for p in diffusion.unet.parameters() if p.requires_grad]
    fashion_params = [p for p in diffusion.fashion_encoder.parameters() if p.requires_grad]
    print(f"Base UNet trainable params: {len(unet_params)}, FashionEncoder trainable params: {len(fashion_params)}")

    gnn_params = []
    if getattr(diffusion, 'gat_projector', None) is not None:
        gnn_params.extend(list(diffusion.gat_projector.parameters()))
    if getattr(diffusion, 'gat_model', None) is not None:
        gnn_params.extend(list(diffusion.gat_model.parameters()))

    seen_param_ids = set()
    unique_base_params = []
    for p in unet_params + fashion_params:
        if id(p) not in seen_param_ids:
            seen_param_ids.add(id(p))
            unique_base_params.append(p)

    unique_gnn_params = []
    for p in gnn_params:
        if id(p) not in seen_param_ids:
            seen_param_ids.add(id(p))
            unique_gnn_params.append(p)

    param_groups = []
    if len(unique_base_params) > 0:
        param_groups.append({"params": unique_base_params, "lr": args.learning_rate, "name": "base"})
    
    if len(unique_gnn_params) > 0:
        param_groups.append({"params": unique_gnn_params, "lr": args.gat_learning_rate, "name": "gnn"})

    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("Please install bitsandbytes to use 8-bit Adam.")
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        param_groups,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Scheduler
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    logger.info("Prepare everything with our accelerator...")
    diffusion, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        diffusion, optimizer, train_dataloader, lr_scheduler
    )
    accelerator.print(f"Device: {accelerator.device}")
    accelerator.print(f"Num processes: {accelerator.num_processes}")

    if args.use_ema:
        ema_unet.to(device)
    if args.use_ema_fashion:
        ema_encoder.to(device)

    # Recalculate training steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Trackers
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers("difashion", config=tracker_config)
        tensorboard_log_dir = os.path.join(args.logging_dir, args.run_name if args.run_name else "default_run")
        writer = SummaryWriter(log_dir=tensorboard_log_dir)
        logger.info(f"TensorBoard logs will be saved to: {tensorboard_log_dir}")

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    
    global_step = 0
    first_epoch = 0

    # Resume
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting new run.")
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))


            manual_gat_path = os.path.join(args.output_dir, path, "gat_manual_backup.pt")
            if os.path.exists(manual_gat_path):
                logger.info(f"Found manual GAT backup at {manual_gat_path}, forcing load...")
                try:
                    gat_data = torch.load(manual_gat_path, map_location='cpu')
                    unwrapped_model = accelerator.unwrap_model(diffusion)
                    
                    if 'gat_model' in gat_data and getattr(unwrapped_model, 'gat_model', None) is not None:
                        unwrapped_model.gat_model.load_state_dict(gat_data['gat_model'])
                        logger.info("Successfully RESTORED GAT Model from manual backup.")
                        
                    if 'gat_projector' in gat_data and getattr(unwrapped_model, 'gat_projector', None) is not None:
                        unwrapped_model.gat_projector.load_state_dict(gat_data['gat_projector'])
                        logger.info("Successfully RESTORED GAT Projector from manual backup.")
                except Exception as e:
                    logger.warning(f"Failed to load manual GAT backup: {e}")
            else:
                logger.warning(f"No manual GAT backup found at {manual_gat_path}. Relying on Accelerate default load.")



            global_step = int(path.split("-")[1])
            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)
    
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, args.num_train_epochs):
        diffusion.train()
        train_loss = 0.0

        for step, batch in enumerate(train_dataloader):
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    # progress_bar.update(1)
                    pass
                continue


            if args.use_gnn:
  
                uids_b = batch["uids"]
                outfits_b = batch["outfits"]
                category_b = batch["category"]
                masked_outfits_b = outfits_b.clone() 
                bsz, olen = outfits_b.shape
                
                mask_indices = torch.randint(0, olen, (bsz,), device=outfits_b.device)
                
                for k in range(bsz):
                    masked_outfits_b[k, mask_indices[k]] = 0 
                graph_data = build_batch_graph(
                    uids_b, 
                    masked_outfits_b, 
                    category_b, 
                    train_history, 
                    item_features_bank,
                    texture_features_bank, 
                    device=accelerator.device,
                    max_hist=args.max_history_per_slot
                )
                
                batch["graph"] = graph_data

            mask_ratio = args.conditioning_dropout_prob
            coupling_mask_ratio = args.coupling_dropout_prob
            cate_mask_ratio = args.cate_conditioning_dropout_prob

            with accelerator.accumulate(diffusion):
                
                loss, base_loss, _, _ = diffusion(
                    batch,
                    img_dataset,
                    train_hist_latents,
                    None, 
                    null_img,
                    mask_ratio,
                    coupling_mask_ratio,
                    cate_mask_ratio,
                    weight_dtype,
                    generator,
                )

                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                accelerator.backward(loss)
                if accelerator.sync_gradients and accelerator.is_main_process:
                 unwrapped = accelerator.unwrap_model(diffusion)

                 if hasattr(unwrapped, 'module'): 
                     unwrapped = unwrapped.module

                 if args.use_gnn and getattr(unwrapped, 'gat_model', None) is not None:
                    p = list(unwrapped.gat_model.parameters())[0]
                    if p.grad is not None:
                        grad_norm = p.grad.data.norm(2).item()
                        writer.add_scalar("Gradients/GAT_Backbone_Norm", grad_norm, global_step)
                    else:
                        writer.add_scalar("Gradients/GAT_Backbone_Norm", -1.0, global_step) 
                    
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(diffusion.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_unet.step(diffusion.unet.parameters())
                if args.use_ema_fashion:
                    ema_encoder.step(diffusion.fashion_encoder.parameters())

                progress_bar.update(1)
                global_step += 1

                current_lrs = {}
                for i, param_group in enumerate(optimizer.param_groups):
                    group_name = param_group.get("name", f"group_{i}")
                    current_lrs[group_name] = param_group["lr"]
                
                if accelerator.is_main_process:
                    for group_name, lr in current_lrs.items():
                        writer.add_scalar(f"LearningRate/{group_name}", lr, global_step)

                accelerator.log({"train_loss": train_loss}, step=global_step)

                if accelerator.is_main_process:
                    writer.add_scalar("Loss/train", loss.item(), global_step)
                    
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    accelerator.wait_for_everyone()
                    
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    
                    if accelerator.is_main_process:
                        logger.info(f"Saved state to {save_path}")
                        unwrapped_model = accelerator.unwrap_model(diffusion)
                        gat_backup = {}
                        if getattr(unwrapped_model, 'gat_model', None) is not None:
                            gat_backup['gat_model'] = unwrapped_model.gat_model.state_dict()
                        if getattr(unwrapped_model, 'gat_projector', None) is not None:
                            gat_backup['gat_projector'] = unwrapped_model.gat_projector.state_dict()
                        
                        if gat_backup:
                            torch.save(gat_backup, os.path.join(save_path, "gat_manual_backup.pt"))
                            logger.info(f"Manually saved GAT backup to {save_path}/gat_manual_backup.pt")

                    accelerator.wait_for_everyone()

                # Validation
                if accelerator.is_main_process:
                    if global_step % 100 == 0: # Validation frequency
                        diffusion.eval()
                        logger.info("Running validation...")

                        gc.collect()
                        torch.cuda.empty_cache()

                        unwrapped_model = accelerator.unwrap_model(diffusion)

                        if unwrapped_model.vae.dtype != torch.float16:
                            print("Converting VAE to float16 for validation...")
                            unwrapped_model.vae.to(dtype=torch.float16)
                        
                        if hasattr(unwrapped_model.vae, "enable_slicing"):
                            unwrapped_model.vae.enable_slicing()
                            print("INFO: VAE Slicing Enabled (降低显存占用)")
                        if hasattr(unwrapped_model.vae, "enable_tiling"):
                            unwrapped_model.vae.enable_tiling()
                            print("INFO: VAE Tiling Enabled")
                        
                        # Store EMA params
                        if args.use_ema:
                            ema_unet.store(diffusion.unet.parameters())
                            ema_unet.copy_to(diffusion.unet.parameters())
                        if args.use_ema_fashion:
                            ema_encoder.store(diffusion.fashion_encoder.parameters())
                            ema_encoder.copy_to(diffusion.fashion_encoder.parameters())

                        with torch.autocast(str(accelerator.device).replace(":0", ""), enabled=accelerator.mixed_precision == "fp16"):
                            image_path = os.path.join(args.output_dir, "samples")
                            os.makedirs(image_path, exist_ok=True)
                            
                            outputs = {}
                            for i, batch_val in enumerate(valid_dataloader):
                                # Inject Graph for Validation
                                if args.use_gnn:
                                    uids_v = batch_val["uids"]
                                    outfits_v = batch_val["outfits"] 
                                    category_v = batch_val["category"]
                                    graph_data_val = build_batch_graph(
                                        uids_v, outfits_v, category_v, valid_history, item_features_bank, texture_features_bank,
                                        device=accelerator.device, max_hist=args.max_history_per_slot
                                    )
                                else:
                                    graph_data_val = None
                                    print("GNN not used, skipping graph construction for validation.")

                                uids = batch_val["uids"].to(device)
                                oids = batch_val["oids"].to(device)
                                category_input_ids = batch_val.get("category_input_ids", batch_val.get("input_ids")).to(device)
                                details_input_ids = batch_val.get("details_input_ids", None)
                                if details_input_ids is not None:
                                    details_input_ids = details_input_ids.to(device)
                                category = batch_val["category"].to(device)
                                olists = batch_val["outfits"].to(device)
                                
                                # Prepare outfit images
                                outfit_images = []
                                for olist in olists:
                                    for iid in olist:
                                        outfit_images.append(img_dataset[iid])
                                outfit_images = torch.stack(outfit_images).to(device)

                                batch_outputs, _ = unwrapped_model.fashion_generation(
                                    batch_graph=graph_data_val, # Pass the graph
                                    uids=uids,
                                    oids=oids,
                                    input_ids=None, 
                                    category_input_ids=category_input_ids,
                                    details_input_ids=details_input_ids,
                                    olists=olists,
                                    outfit_images=outfit_images,
                                    category=category,
                                    history=valid_hist_latents,
                                    num_inference_steps=args.num_inference_steps,
                                    category_guidance_scale=args.category_guidance_scale,
                                    hist_guidance_scale=args.hist_guidance_scale,
                                    mutual_guidance_scale=args.mutual_guidance_scale,
                                    null_img=null_img,
                                    generator=generator,
                                    return_dict=False
                                )

                                for uid, oid_map in batch_outputs.items():
                                    for oid, res in oid_map.items():
                                        found_idx = -1
                                        for b_idx, curr_oid in enumerate(oids):
                                            if int(curr_oid.item()) == int(oid):
                                                found_idx = b_idx
                                                break
                                        
                                        if found_idx != -1:
                                            res['full_cates'] = category[found_idx] # 保存完整类别
                                            
                                            start_img_idx = found_idx * 4 
                                            current_olist = olists[found_idx]
                                            
                                            ctx_imgs = []
                                            for item_idx, iid in enumerate(current_olist):
                                                if iid.item() != 0:
                                                    img_tensor = outfit_images[start_img_idx + item_idx]
                                                    ctx_imgs.append(img_tensor)
                                            
                                            res['context_images'] = ctx_imgs

                                outputs.update(batch_outputs)
                                if i > 3: break
                            try:
                                logger.info("Calculating online metrics for validation batch...")
                                
                                feat_extractor = getattr(unwrapped_model, 'image_feature_extractor', None)
                                if feat_extractor is None:
                                     if not hasattr(accelerator, '_local_feat_extractor'):
                                        accelerator._local_feat_extractor = ClipFeatureExtractor().to(device)
                                     feat_extractor = accelerator._local_feat_extractor

                                if feat_extractor is not None and valid_hist_features is not None:
                                    metrics = calculate_metrics_online(
                                        outputs, 
                                        feat_extractor, 
                                        valid_hist_features, 
                                        device,
                                        resolution=args.resolution
                                    )
                                    
                                    p_score = metrics['Personalization']
                                    c_score = metrics['Compatibility']
                                    
                                    logger.info(f"[VAL STEP {global_step}] Personalization: {p_score:.4f} | Compatibility: {c_score:.4f}")
                                    
                                    writer.add_scalar("Val/Personalization", p_score, global_step)
                                    writer.add_scalar("Val/Compatibility", c_score, global_step)
                                else:
                                    logger.warning("Skipping metrics: Feature extractor or valid history missing.")
                                    
                            except Exception as e:
                                logger.error(f"Metrics calculation failed: {e}")
                                import traceback
                                traceback.print_exc()

                            if global_step % 1000 == 0: 
                                for i,uid in enumerate(outputs):
                                    uid_image_path = os.path.join(image_path, str(uid))
                                    if not os.path.exists(uid_image_path):
                                        os.makedirs(uid_image_path)
                                    
                                    for oid in outputs[uid]:
                                        data = outputs[uid][oid]
                                        images = data.get("images", [])
                                        cates = data.get("cates", [])
                                        full_cates = data.get("full_cates", []) 

                                        if len(full_cates) == 0: continue 

                                        oid_image_path = os.path.join(uid_image_path, str(oid))
                                        if not os.path.exists(oid_image_path):
                                            os.makedirs(oid_image_path)
                                        
                                        grd_exist = False
                                        files_in_oid_image_path = os.listdir(oid_image_path)
                                        for filename in files_in_oid_image_path:
                                            if "grd" in filename:
                                                grd_exist = True

                                        if not grd_exist:
                                            semantic_cates = []
                                            for j,cate in enumerate(full_cates):
                                                c_val = cate.item() if hasattr(cate, "item") else int(cate)
                                                if c_val in new_id_cate_dict:
                                                    semantic_cates.append(new_id_cate_dict[c_val])
                                                else:
                                                    semantic_cates.append(str(c_val))

                                            grd_imgs = []
                                            for iid in valid_grd_dict[oid]["outfits"]:
                                                img = Image.open(os.path.join(args.img_folder_path, all_image_paths[iid]))
                                                grd_imgs.append(img)

                                            merge_and_save_images(
                                                grd_imgs,
                                                os.path.join(oid_image_path, 'grd_' + '_'.join(semantic_cates) + '.jpg')
                                            )
                                            
                                        semantic_cates = []
                                        for j,cate in enumerate(cates):
                                            c_val = cate.item() if hasattr(cate, "item") else int(cate)
                                            if c_val in new_id_cate_dict:
                                                semantic_cates.append(new_id_cate_dict[c_val])
                                            else:
                                                semantic_cates.append(str(c_val))
                                        
                                        merge_and_save_images(
                                            images,
                                            os.path.join(oid_image_path, f"{global_step}_{args.mutual_guidance_scale}_{args.hist_guidance_scale}_" + '_'.join(semantic_cates) + '.jpg')
                                        )
                        
                        # Restore EMA
                        if args.use_ema:
                            ema_unet.restore(diffusion.unet.parameters())
                        if args.use_ema_fashion:
                            ema_encoder.restore(diffusion.fashion_encoder.parameters())
                        
                        torch.cuda.empty_cache()

            if global_step >= args.max_train_steps:
                break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        writer.close()
    accelerator.end_training()

def merge_and_save_images(images, save_path):
    cols = math.ceil(math.sqrt(len(images)))
    width = images[0].width
    height = images[0].height
    total_width = width * cols
    total_height = height * cols

    merged_image = Image.new('RGB', (total_width, total_height), color=(255, 255, 255))
    for i in range(len(images)):
        row = i // cols
        col = i % cols
        merged_image.paste(images[i], (col * width, row * height))
        
    merged_image.save(save_path)

if __name__ == "__main__":
    main()