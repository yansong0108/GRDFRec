#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Script to fine-tune Stable Diffusion for Fashion Outfit Generation and Recommendation"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import argparse
import logging
import math
import shutil
from pathlib import Path
from tqdm import tqdm

import accelerate
import datasets
import numpy as np
import PIL
from PIL import Image
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm

import diffusers
from diffusers import UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available

from torch_geometric.data import Data, Batch

import data_utils
from models.model_components import MutualEncoder, ClipFeatureExtractor
from models.GRDFRec import DiFashion
# Will error if the minimal version of diffusers is not installed. Remove at your own risks.

# accelerate launch --config_file config.yaml inf4eval_hisloss_gnn_ema.py --task FITB --mode valid --use_gnn --use_ema --use_ema_fashion --mixed_precision fp16

check_min_version("0.16.0")

logger = get_logger(__name__, log_level="INFO")

def parse_all_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")

        # 个性化损失的超参数
    parser.add_argument(
        "--use_personalization_loss",
        action="store_true",
        help="Whether to use personalization loss during training."
    )
    parser.add_argument(
        "--personal_lambda",
        type=float,
        default=0.05,
        help="Weight for the personalization loss term."
    )
    parser.add_argument(
        "--personal_lambda_warmup_steps",
        type=int,
        default=5000,
        help="Number of steps for the linear warmup of personal_lambda. Defaults to 0, which means no warmup.",
    )
    parser.add_argument(
        "--sim_func",
        type=str,
        default="cosine",
        choices=["cosine", "l2"],
        help="Similarity function for personalization loss calculation."
    )
    parser.add_argument(
        "--feature_extractor_path",
        type=str,
        default=None,
        help="Path to pretrained feature extractor model for personalization loss."
    )

    # GNN-specific hyperparameters
    parser.add_argument(
        "--use_gnn",
        action="store_true",
        help="Enable the Outfit Compatibility GNN module during training."
    )
    parser.add_argument(
        "--gat_learning_rate",
        type=float,
        default=1e-4, # 默认给一个很小的学习率，防止破坏预训练
        help="Learning rate for GAT when unfreezing."
    )
    parser.add_argument("--max_history_per_slot", type=int, default=1000)


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
        # default='../datasets/ifashion',
        default='../datasets/polyvore',
        help="A folder containing the dataset for training and inference."
    )
    parser.add_argument(
        '--img_folder_path',
        type=str,
        # default='../semantic_category'
        default='../polyvore_folder_path/291x291'
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
        default="./output_train_gat/polyvore/difashion_gat_ug_ema",
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
        "--task",
        type=str,
        default="GOR",
        # default="FITB",
        help="The task for evaluation: FITB or GOR (Generative Outfit Recommendation)."
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
        "--num_inference_steps",
        type=int,
        default=50
    )
    parser.add_argument(
        "--category_guidance_scale",
        type=float,
        default=12.0
    )
    parser.add_argument(
        "--hist_guidance_scale",
        type=float,
        default=4.0
    )
    parser.add_argument(
        "--mutual_guidance_scale",
        type=float,
        default=5.0
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="test"
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=5, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=20000,
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
        default=1e-5,
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
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
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
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
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
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=1000,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=(
            "Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator` `ProjectConfiguration`."
            " See Accelerator::save_state https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.save_state"
            " for more docs"
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        # default="latest",
        default="--checkpointing_steps",
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest` to automatically select the last available checkpoint.'
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
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument("--run_name", type=str, default='', help="Run name")

    

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    args.output_dir = os.path.join(args.output_dir, args.run_name)
 
    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args

def get_item_feature(item_id, item_features, feature_dim=1024): # 注意：ViT-H 是 1024 维
    iid = int(item_id)
    raw_feat = None

    if isinstance(item_features, np.ndarray):
        if 0 <= iid < len(item_features):
            raw_feat = item_features[iid]
    elif isinstance(item_features, dict):
        raw_feat = item_features.get(iid)

    if raw_feat is not None:
        feat = raw_feat
        if isinstance(feat, dict):
            for key in ['feature', 'clip_feature', 'embedding', 'vec', 'feat']:
                if key in feat: feat = feat[key]; break
        
        feat = np.asarray(feat)
        if feat.dtype == object or feat.ndim == 0:
            return np.zeros(feature_dim, dtype=np.float32)
        
        feat = feat.flatten()
        if feat.shape[0] > feature_dim: 
            feat = feat[:feature_dim]
        elif feat.shape[0] < feature_dim:
            pad = np.zeros(feature_dim - feat.shape[0], dtype=feat.dtype)
            feat = np.concatenate([feat, pad])
        return feat.astype(np.float32)
    
    else:
        print(f"[Warning] Feature not found for Item ID: {item_id}")

    return np.zeros(feature_dim, dtype=np.float32)

def build_batch_graph(uids, outfits, outfit_cates, history, item_features, texture_features, device, max_hist=10, feature_dim=1024):
    batch_data_list = []
    bsz = outfits.shape[0]
    
    # Combined dimension: 1024 (Global) + 1024 (Texture) = 2048
    combined_dim = feature_dim * 2 
    
    for i in range(bsz):
        uid = int(uids[i].item())
        outfit_items = outfits[i].tolist() 
        current_cates = outfit_cates[i].tolist()
        
        slot_feats = []
        hist_feats = []
        slot_to_hist_map = {} 

        # -----------------------------------------------------------
        # 1. Slot Nodes (Target/Context) -> Dimension 2048
        # -----------------------------------------------------------
        for item_id in outfit_items:
            iid = int(item_id)
            if iid == 0: 
                # Target: Fill with 0s (2048 dim)
                slot_feats.append(np.zeros(combined_dim, dtype=np.float32))
            else:
                # Context: Global + Texture
                g_feat = get_item_feature(iid, item_features, feature_dim) # 1024
                t_feat = get_item_feature(iid, texture_features, feature_dim) # 1024
                combined = np.concatenate([g_feat, t_feat]) # 2048
                slot_feats.append(combined)
        
        num_slots = len(slot_feats)
        
        # -----------------------------------------------------------
        # 2. History Nodes -> [FIX] Must also be 2048 dim!
        # -----------------------------------------------------------
        for slot_idx, target_iid in enumerate(outfit_items):
            target_iid = int(target_iid)
            cate = current_cates[slot_idx]

            user_hist = history.get(uid, {})
            hist_item_ids = user_hist.get(cate) or user_hist.get(str(cate)) or user_hist.get(int(cate))
            
            current_hist_indices = []
            if hist_item_ids is not None:
                valid_ids = [int(x) for x in hist_item_ids if int(x) != int(target_iid)]
                if max_hist is not None: valid_ids = valid_ids[-max_hist:]
                
                start_idx = len(hist_feats)
                added_count = 0
                for hid in valid_ids:
                    # [CRITICAL FIX HERE]
                    # History nodes must also concatenate texture features!
                    g_feat_h = get_item_feature(hid, item_features, feature_dim) # 1024
                    t_feat_h = get_item_feature(hid, texture_features, feature_dim) # 1024
                    
                    combined_hist = np.concatenate([g_feat_h, t_feat_h]) # 2048
                    
                    hist_feats.append(combined_hist)
                    added_count += 1
                
                if added_count > 0:
                    current_hist_indices = list(range(start_idx, start_idx + added_count))
            
            slot_to_hist_map[slot_idx] = current_hist_indices
            
        # -----------------------------------------------------------
        # 3. Construct Graph Data
        # -----------------------------------------------------------
        all_feats = slot_feats + hist_feats
        
        if len(all_feats) == 0:
            x = torch.zeros((num_slots, combined_dim), dtype=torch.float32)
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        else:
            # Now all arrays in the list are shape (2048,), so stack will work
            x = torch.tensor(np.stack(all_feats), dtype=torch.float32)
            
            src, dst = [], []
            for s1 in range(num_slots):
                for s2 in range(num_slots):
                    if s1 != s2:
                        src.append(s1); dst.append(s2)
            for s_idx in range(num_slots):
                h_indices = slot_to_hist_map.get(s_idx, [])
                for h_rel in h_indices:
                    h_global = num_slots + h_rel
                    src.append(s_idx); dst.append(h_global)
                    src.append(h_global); dst.append(s_idx)
            
            if len(src) == 0: edge_index = torch.zeros((2, 0), dtype=torch.long)
            else: edge_index = torch.tensor([src, dst], dtype=torch.long)
                
        d = Data(x=x, edge_index=edge_index)
        slot_mask = torch.zeros(x.shape[0], dtype=torch.bool)
        slot_mask[:num_slots] = True
        d.slot_mask = slot_mask
        batch_data_list.append(d)
        
    return Batch.from_data_list(batch_data_list).to(device)





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
    
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()
    
    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
    
    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    logger.info(f"Args: {args}")
    logger.info("Data loading......")
    data_path = os.path.join(args.data_path, args.dataset_name)

    if args.data_processed:
        train_dict = np.load(os.path.join(data_path, "processed", "train.npy"), allow_pickle=True).item()

        train_history = np.load(os.path.join(data_path, "processed", "train_hist_latents.npy"), allow_pickle=True).item()

        if args.mode == "test":
            test_fitb_dict = np.load(os.path.join(data_path, "processed", "fitb_test.npy"), allow_pickle=True).item()
            test_history = np.load(os.path.join(data_path, "processed", "test_hist_latents.npy"), allow_pickle=True).item()
            test_orig_history = np.load(os.path.join(data_path, "test_history.npy"), allow_pickle=True).item()
        else:
            test_fitb_dict = np.load(os.path.join(data_path, "processed", "fitb_valid.npy"), allow_pickle=True).item()
            test_history = np.load(os.path.join(data_path, "processed", "valid_hist_latents.npy"), allow_pickle=True).item()
            test_orig_history = np.load(os.path.join(data_path, "valid_history.npy"), allow_pickle=True).item()
    else:
        train_dict = np.load(os.path.join(data_path, 'train.npy'), allow_pickle=True).item()

        valid_fitb_dict = np.load(os.path.join(data_path, "fitb_valid.npy"), allow_pickle=True).item()
        test_fitb_dict = np.load(os.path.join(data_path, "fitb_test.npy"), allow_pickle=True).item()

        train_history = np.load(os.path.join(data_path, "train_history.npy"), allow_pickle=True).item()
        valid_history = np.load(os.path.join(data_path, "valid_history.npy"), allow_pickle=True).item()
        test_history = np.load(os.path.join(data_path, "test_history.npy"), allow_pickle=True).item()

    if args.mode == "test":
        test_grd_dict = np.load(os.path.join(data_path, "test_grd.npy"), allow_pickle=True).item()
    else:
        test_grd_dict = np.load(os.path.join(data_path, "valid_grd.npy"), allow_pickle=True).item()


    new_id_cate_dict = np.load(os.path.join(data_path, "id_cate_dict.npy"), allow_pickle=True).item()
    all_image_paths = np.load(os.path.join(data_path, "all_item_image_paths.npy"), allow_pickle=True)
    item_features_bank = np.load(os.path.join(data_path, "cnn_features_clip.npy"), allow_pickle=True)

    texture_feat_path = os.path.join(data_path, "processed", "texture_features_clip.npy")
    if os.path.exists(texture_feat_path):
        logger.info(f"Loading texture features from {texture_feat_path}...")
        # 注意：我们存的是字典，所以要用 .item() 取出
        texture_features_bank = np.load(texture_feat_path, allow_pickle=True).item()
        logger.info(f"Loaded {len(texture_features_bank)} texture entries.")
    
    
    img_trans = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor()
        ]
    )
    img_dataset = data_utils.ImagePathDataset(args.img_folder_path, all_image_paths, img_trans, do_normalize=True)
    null_img = img_dataset[0].to(device)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    logger.info("Build the diffusion model......")
    diffusion = DiFashion(args, logger, len(new_id_cate_dict), device)
    logger.info("Completed.")

    with accelerator.main_process_first():
        if args.data_processed:
            train_data_dict = train_dict
            test_data_dict = test_fitb_dict

            train_hist_latents = train_history
            test_hist_latents = test_history

            logger.info(f"Successfully loaded the processed data for training and validation.")
        else:
            logger.info(f"Preprocess datasets for DiFashion.")
            train_data_dict, train_hist_latents = data_utils.preprocess_dataset(train_dict, data_path,
                new_id_cate_dict, train_history, img_dataset, diffusion.tokenizer, diffusion.vae, device)

            save_path = os.path.join(data_path, "processed")
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            np.save(os.path.join(save_path, "new_train.npy"), np.array(train_data_dict))
            np.save(os.path.join(save_path, "train_hist_latents.npy"), np.array(train_hist_latents))

            valid_data_dict, valid_hist_latents = data_utils.preprocess_dataset(valid_fitb_dict, data_path,
                new_id_cate_dict, valid_history, img_dataset, diffusion.tokenizer, diffusion.vae, device)
            
            np.save(os.path.join(save_path, "new_fitb_valid.npy"), np.array(valid_data_dict))
            np.save(os.path.join(save_path, "valid_hist_latents.npy"), np.array(valid_hist_latents))

            test_data_dict, test_hist_latents = data_utils.preprocess_dataset(test_fitb_dict, data_path,
                new_id_cate_dict, test_history, img_dataset, diffusion.tokenizer, diffusion.vae, device)
            
            np.save(os.path.join(save_path, "new_fitb_test.npy"), np.array(test_data_dict))
            np.save(os.path.join(save_path, "test_hist_latents.npy"), np.array(test_hist_latents))

            logger.info(f"Successfully processed and saved the dataset for training, validation and test into {save_path}.")
    
    # Safe collate: gracefully skip optional None fields (e.g., details_*) to prevent default_collate errors
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

    test_dataset = data_utils.FashionDiffusionData(test_data_dict)
    if args.task == "FITB":
        test_batch_size = 5
        # test_batch_size = 5
    else:
        test_batch_size = 1
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, 
        shuffle=False, 
        batch_size=test_batch_size,
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

    if args.gradient_checkpointing:
        diffusion.unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    logger.info("build the optimizer...")
    
    # 1. 收集基础参数 (UNet + FashionEncoder)
    # 只收集 requires_grad=True 的参数 (尊重 freeze_unet_backbone 设置)
    unet_params = [p for p in diffusion.unet.parameters() if p.requires_grad]
    fashion_params = [p for p in diffusion.fashion_encoder.parameters() if p.requires_grad]
    print(f"Base UNet trainable params: {len(unet_params)}, FashionEncoder trainable params: {len(fashion_params)}")
    
    # 2. 收集 GAT 相关所有参数 (Projector + Backbone)
    gnn_params = []
    if getattr(diffusion, 'gat_projector', None) is not None:
        gnn_params.extend(list(diffusion.gat_projector.parameters()))
    # [FIX] 显式加入 GAT Model 参数！
    if getattr(diffusion, 'gat_model', None) is not None:
        gnn_params.extend(list(diffusion.gat_model.parameters()))

    # 3. 参数去重
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

    # 4. 构建组
    param_groups = []
    if len(unique_base_params) > 0:
        param_groups.append({"params": unique_base_params, "lr": args.learning_rate, "name": "base"})
    
    if len(unique_gnn_params) > 0:
        # 使用较大的学习率
        param_groups.append({"params": unique_gnn_params, "lr": args.gat_learning_rate, "name": "gnn"})

    # Initialize the optimizer
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

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # Evaluation-only: scheduler is not required

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    logger.info("Prepare model with optimizer (no scheduler) for DeepSpeed eval...")
    diffusion, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        diffusion, optimizer, train_dataloader, lr_scheduler
    )


    if args.use_ema:
        ema_unet.to(device)
    
    if args.use_ema_fashion:
        ema_encoder.to(device)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers("difashion", config=tracker_config)

        

    # Inference phase
    global_step = 0
    # inf_list = ["checkpoint-5000","checkpoint-6000","checkpoint-7000","checkpoint-8000","checkpoint-9000","checkpoint-10000","checkpoint-11000","checkpoint-12000","checkpoint-13000","checkpoint-14000","checkpoint-15000","checkpoint-16000","checkpoint-17000","checkpoint-18000","checkpoint-19000","checkpoint-20000"]
    inf_list = ["checkpoint-48000"]
    scale_list = [2.0]

    logger.info(f"inf list: {inf_list}")
    logger.info(f"scale list: {scale_list}")

    if args.mode == "test":
        save_path = os.path.join(args.output_dir, "eval-test")
    else:
        save_path = os.path.join(args.output_dir, "eval")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for path in inf_list:
        global_step = int(path.split("-")[1])
        
        save_grd = True
        grd_save_path = os.path.join(save_path, f"{args.task}-grd-new.npy")
        if os.path.exists(grd_save_path):
            print(f"Groundtruth file already exists in {grd_save_path}.")
            save_grd = False

        
        accelerator.print(f"Resuming weights from checkpoint {path}")
        checkpoint_path = os.path.join(args.output_dir, path)
        model = accelerator.unwrap_model(diffusion)

        logger.info("Loading BASE weights first to guarantee architecture...")
        model.unet.load_state_dict(UNet2DConditionModel.from_pretrained(checkpoint_path, subfolder="unet").state_dict())
        model.fashion_encoder.load_state_dict(MutualEncoder.from_pretrained(checkpoint_path, subfolder="fashion_encoder").state_dict())

        if args.use_ema:
            logger.info(f"Safely injecting UNet EMA weights from: {checkpoint_path}/unet_ema")
            ema_unet_wrapper = EMAModel(model.unet.parameters(), model_cls=UNet2DConditionModel, model_config=model.unet.config)
            ema_unet_loaded = EMAModel.from_pretrained(os.path.join(checkpoint_path, "unet_ema"), UNet2DConditionModel)
            ema_unet_wrapper.load_state_dict(ema_unet_loaded.state_dict())
            ema_unet_wrapper.copy_to(model.unet.parameters())
            del ema_unet_wrapper, ema_unet_loaded
        if args.use_ema_fashion:
            logger.info(f"Safely injecting Fashion Encoder EMA from: {checkpoint_path}/fashion_encoder_ema")
            ema_enc_wrapper = EMAModel(model.fashion_encoder.parameters(), model_cls=MutualEncoder, model_config=model.fashion_encoder.config)
            
            ema_enc_loaded = EMAModel.from_pretrained(os.path.join(checkpoint_path, "fashion_encoder_ema"), MutualEncoder)
            ema_enc_wrapper.load_state_dict(ema_enc_loaded.state_dict())
            
            ema_enc_wrapper.copy_to(model.fashion_encoder.parameters())
            del ema_enc_wrapper, ema_enc_loaded
            
        if args.use_gnn:
            manual_gat_path = os.path.join(checkpoint_path, "gat_manual_backup.pt")
            if os.path.exists(manual_gat_path):
                gat_state = torch.load(manual_gat_path, map_location=device)
                if 'gat_model' in gat_state and model.gat_model is not None:
                    model.gat_model.load_state_dict(gat_state['gat_model'])
                if 'gat_projector' in gat_state and model.gat_projector is not None:
                    model.gat_projector.load_state_dict(gat_state['gat_projector'])

        if accelerator.is_main_process:
            diffusion.eval()
            unwrapped_model = accelerator.unwrap_model(diffusion)
            print("INFO: Forcing VAE to float32 to completely prevent NaN Blue/Gray blocks...")
            unwrapped_model.vae.to(dtype=torch.float32)
            
            for scale in scale_list:
                # You can change the conditional scales during inference
                hist_guidance_scale = args.hist_guidance_scale
                mutual_guidance_scale = args.mutual_guidance_scale
                category_guidance_scale = args.category_guidance_scale

                gen_save_path = os.path.join(save_path, f"{args.task}-checkpoint-{global_step}-cate{category_guidance_scale}-mutual{mutual_guidance_scale}-hist{hist_guidance_scale}")
                if os.path.exists(gen_save_path):
                    logger.info(f"{args.task}-checkpoint-{global_step}-cate{category_guidance_scale}-mutual{mutual_guidance_scale}-hist{hist_guidance_scale} has already been infered on Task {args.task}. Skip.")
                    continue

                logger.info(f"Running validation on {args.task}-checkpoint-{global_step}-cate{category_guidance_scale}-mutual{mutual_guidance_scale}-hist{hist_guidance_scale}...")
                generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)  # refresh the generator with the same seed for another ckpt/guidance_scale

                with torch.no_grad():
                    with torch.autocast(
                        str(accelerator.device).replace(":0", ""), enabled=accelerator.mixed_precision == "fp16"
                    ):       
                        outputs = {}
                        all_grds = {}
                        for i,batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
                            uids = batch["uids"].to(device)
                            oids = batch["oids"].to(device)
                            category = batch["category"].to(device)
                            olists = batch["outfits"].to(device)
                            
                            if args.use_gnn:

                                original_history = test_orig_history
                                    
                                masked_outfits = olists.clone() 
                                bsz, olen = olists.shape
                                
                                # [修改后] GOR Logic
                                if args.task == "GOR":
                                    # 强制将所有位置都 Mask 掉 (设为 0)
                                    # 假设 0 是 padding/mask token ID
                                    masked_outfits = torch.zeros_like(olists).to(device) 
                                else:
                                    # FITB Logic (保持不变)
                                    mask_indices = torch.randint(0, olen, (bsz,), device=olists.device)
                                    for k in range(bsz):
                                        masked_outfits[k, mask_indices[k]] = 0
                                graph_data = build_batch_graph(
                                    uids, masked_outfits, category, original_history, item_features_bank, texture_features_bank,
                                    device=accelerator.device, max_hist=args.max_history_per_slot
                                )
                                batch["graph"] = graph_data
                            
                            # input_ids = batch["input_ids"].to(device)
                            category_input_ids = batch.get("category_input_ids", batch.get("input_ids")).to(device)
                            details_input_ids = batch.get("details_input_ids", None)
                            if details_input_ids is not None:
                                details_input_ids = details_input_ids.to(device)
                            

                            outfit_images = []
                            if args.task == "FITB":
                                for olist in olists:
                                    for iid in olist:
                                        outfit_images.append(img_dataset[iid])
                            else:
                                for olist in olists:
                                    for iid in olist:
                                        outfit_images.append(img_dataset[0])
                                olists = torch.zeros_like(olists, dtype=int).to(device)
                            outfit_images = torch.stack(outfit_images).to(device)

                            batch_outputs, _ = unwrapped_model.fashion_generation(
                                batch_graph=graph_data,
                                uids=uids,
                                oids=oids,
                                category_input_ids=category_input_ids,
                                details_input_ids=details_input_ids,
                                olists=olists,
                                outfit_images=outfit_images,
                                category=category,
                                history=test_hist_latents,
                                num_inference_steps=args.num_inference_steps,
                                category_guidance_scale=category_guidance_scale,
                                hist_guidance_scale=hist_guidance_scale,
                                mutual_guidance_scale=mutual_guidance_scale,
                                null_img=null_img,
                                generator=generator,
                                return_dict=False,
                            )

                            outputs, all_grds = save_batch_outputs(outputs, all_grds, batch_outputs, gen_save_path, args.task,
                                    args.img_folder_path, all_image_paths, test_grd_dict, save_grd)

                            np.save(gen_save_path, np.array(outputs))

                            if save_grd:
                                np.save(grd_save_path, np.array(all_grds))

                torch.cuda.empty_cache()
    
    logger.info(f"All the checkpoints in the inf_list have been inferenced for evaluation.")
    logger.info(f"inf list: {inf_list}")

def save_batch_outputs(all_outputs, all_grds, outputs, gen_save_path, task, all_img_folder_path, all_image_paths, test_grd_dict, save_grd=True):
    for uid in outputs:
        for oid in outputs[uid]:
            imgs = outputs[uid][oid]["images"]
            img_paths = []
            img_folder_path = os.path.join(gen_save_path, "images", str(uid), str(oid))
            if not os.path.exists(img_folder_path):
                os.makedirs(img_folder_path)

            if task == "GOR":
                merged_img_path = os.path.join(img_folder_path, "all.jpg")
                merge_and_save_images(imgs, merged_img_path)

            for i,img in enumerate(imgs):
                img_path = os.path.join(img_folder_path, f"{str(i)}.jpg")
                img.save(img_path)
                img_paths.append(img_path)
            outputs[uid][oid]["image_paths"] = img_paths

            del outputs[uid][oid]["images"]

            if uid not in all_outputs:
                all_outputs[uid] = {}
            if oid not in all_outputs[uid]:
                all_outputs[uid][oid] = outputs[uid][oid]

            if task == "FITB":
                # save grd images
                grd_images = []
                for iid in test_grd_dict[oid]["outfits"]:
                    img = Image.open(os.path.join(all_img_folder_path, all_image_paths[iid]))
                    grd_images.append(img)
                grd_img_path = os.path.join(gen_save_path, "images", str(uid), str(oid), "grd.jpg")
                merge_and_save_images(grd_images, grd_img_path)

    if save_grd:
        for uid in outputs:
            for oid in outputs[uid]:
                if uid not in all_grds:
                    all_grds[uid] = {}
                if oid not in all_grds[uid]:
                    all_grds[uid][oid] = {}
                    all_grds[uid][oid]["outfits"] = test_grd_dict[oid]["outfits"]

                    # only save image paths for evaluation
                    img_paths = []
                    for cate in outputs[uid][oid]["cates"]:
                        idx = torch.where(outputs[uid][oid]["full_cates"] == cate)[0]
                        iid = test_grd_dict[oid]["outfits"][idx]
                        img_path = os.path.join(all_img_folder_path, all_image_paths[iid])
                        img_paths.append(img_path)
                    all_grds[uid][oid]["image_paths"] = img_paths

    return all_outputs, all_grds

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

def extract_number(filename):
    num_str = ''.join(filter(str.isdigit, filename))
    return int(num_str) if num_str else 0

def get_subdirectories(folder_path):
    subdirectories = [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name)) and name != "eval"]
    return sorted(subdirectories, key=lambda x: extract_number(os.path.basename(x)))

if __name__ == "__main__":
    main()
