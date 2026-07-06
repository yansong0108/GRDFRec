import copy
import os
import random
from fileinput import filename

import numpy as np
import scipy.sparse as sp
import torch
import torch.utils.data as data
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

import torch.nn as parallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler


from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class ImagePathDataset(Dataset):
    def __init__(self, folder_path, paths, trans=None, do_normalize=True):
        self.folder_path = folder_path
        self.paths = paths
        self.trans = trans
        self.do_normalize = do_normalize
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        path = os.path.join(self.folder_path, self.paths[idx])
        img = Image.open(path).convert('RGB')
        if self.trans is not None:
            img = self.trans(img)
        if self.do_normalize:
            img = 2 * img - 1
        return img.to(memory_format=torch.contiguous_format).float()

class ImagePathProcess(Dataset):
    def __init__(self, folder_path, paths):
        self.folder_path = folder_path
        self.paths = paths
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        path = os.path.join(self.folder_path, self.paths[idx])
        img = Image.open(path).convert('RGB')
        return img

class FashionDiffusionData(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data["uids"])
    
    def __getitem__(self, index):
        uids = self.data["uids"][index]
        oids = self.data["oids"][index]
        outfits = self.data["outfits"][index] 
        input_ids = self.data["input_ids"][index]
        category = self.data["category"][index]

        # support both category_input_ids and details_input_ids if present
        category_input_ids = self.data.get("category_input_ids", None)
        details_input_ids = self.data.get("details_input_ids", None)
        if category_input_ids is not None:
            category_input_ids = category_input_ids[index]
        if details_input_ids is not None:
            details_input_ids = details_input_ids[index]

        return {"uids": uids, "oids": oids, "outfits": outfits, 
                "input_ids": input_ids, "category": category,
                "category_input_ids": category_input_ids, "details_input_ids": details_input_ids}

class FashionFITBData(Dataset):
    def __init__(self, data, all_test_grd, fill_num=1):
        self.data = data
        self.test_grd = all_test_grd
        self.fill_num = fill_num
    
    def __len__(self):
        return len(self.data["uids"])
    
    def __getitem__(self, index):
        uids = self.data["uids"][index]
        oids = self.data["oids"][index]
        outfits = torch.tensor(self.test_grd["outfits"][index])
        for i in range(self.fill_num):
            outfits[i] = 0
        input_ids = self.data["input_ids"][index]
        category = self.data["category"][index]

        # category/details tokens if exist
        category_input_ids = self.data.get("category_input_ids", None)
        details_input_ids = self.data.get("details_input_ids", None)
        if category_input_ids is not None:
            category_input_ids = category_input_ids[index]
        if details_input_ids is not None:
            details_input_ids = details_input_ids[index]

        return {"uids": uids, "oids": oids, "outfits": outfits, 
                "input_ids": input_ids, "category": category,
                "category_input_ids": category_input_ids, "details_input_ids": details_input_ids}

def create_feature_history(raw_history, feature_extractor, img_dataset, device, batch_size=32):
    """
    Converts a history of image IDs into a history of averaged feature vectors.
    """
    feature_extractor.to(device)
    feature_extractor.eval()
    
    feature_history = {}
    
    with torch.no_grad():
        for user_id, categories in tqdm(raw_history.items(), desc="Processing user history"):
            feature_history[user_id] = {}
            for category_id, item_ids in categories.items():
                if not item_ids:
                    continue
                
                # Fetch images for the current category
                hist_images = torch.stack([img_dataset[iid] for iid in item_ids]).to(device)
                
                all_features = []
                # Process in batches to avoid memory errors
                for i in range(0, len(hist_images), batch_size):
                    batch_images = hist_images[i:i + batch_size]
                    # Extract features (output is a feature map)
                    feature_maps = feature_extractor(batch_images)
                    # Flatten the feature maps to vectors
                    feature_vectors = feature_maps.view(feature_maps.size(0), -1)
                    all_features.append(feature_vectors)
                
                # Concatenate all features for the category and compute the average
                if all_features:
                    all_features = torch.cat(all_features, dim=0)
                    avg_feature = all_features.mean(dim=0, keepdim=True) # Shape [1, D]
                    feature_history[user_id][category_id] = avg_feature.cpu()

    return feature_history

# ##### Preprocessing the datasets.

def preprocess_dataset(data, data_path, id_cate_dict, history, img_dataset, tokenizer, vae, device, **kwargs):

    """
    Extended preprocess_dataset:
    - supports optional kwargs:
        all_item_paths: list of image paths (index matches item id)
        item_description_map: dict mapping path-string -> list_of_description_tokens (or list of strings)
        use_item_details: bool (whether to produce details_input_ids)
    - returns (data_dict, hist_latents)
    """

    all_item_paths = kwargs.get("all_item_paths", None)
    item_description_map = kwargs.get("item_description_map", {}) or {}
    item_description_idx_map = kwargs.get("item_description_idx_map", {}) or {}
    use_item_details = kwargs.get("use_item_details", False)

    def contains_any_special_cate(category, special_cates):
         for special_cate in special_cates:
             if special_cate in category:
                 return True
         return False

     # process text prompts
    def tokenize_category(data):
        data["input_ids"] = []
        data["category_input_ids"] = []
        for outfit_category in data["category"]:
            category_prompts = []
            for cid in outfit_category:
                category = id_cate_dict[cid]
                special_cates = ["pants", "earrings"]
                if contains_any_special_cate(category, special_cates):
                    category_prompts.append("A photo of a pair of " + category + ", on white background, high quality")
                else:
                    category_prompts.append("A photo of a " + category + ", on white background, high quality")
            inputs = tokenizer(
                category_prompts, max_length=77, padding="max_length", truncation=True, return_tensors="pt"
            )
            data["input_ids"].append(inputs.input_ids)
            data["category_input_ids"].append(inputs.input_ids)
            
        return data

    data = tokenize_category(data)

    # prepare item-level detail prompts if requested
    if use_item_details:
        data["details_input_ids"] = []
        max_len = tokenizer.model_max_length if hasattr(tokenizer, "model_max_length") else 77
        for outfit in data["outfits"]:
            detail_prompts = []
            for iid in outfit:
                # 仅使用索引映射，避免路径/长度错配
                desc_list = item_description_idx_map.get(int(iid), [])
                if isinstance(desc_list, (list, tuple)):
                    prompt = " ".join([str(x) for x in desc_list]) if len(desc_list) > 0 else ""
                else:
                    prompt = str(desc_list) if desc_list is not None else ""
                # 调试输出（无路径）
                detail_prompts.append(prompt)

            # tokenization per outfit: produces tensor [olen, max_len]
            inputs = tokenizer(detail_prompts, max_length=max_len, padding="max_length", truncation=True, return_tensors="pt")
            data["details_input_ids"].append(inputs.input_ids)
    all_latents_path = os.path.join(data_path, "all_item_latents.npy")
    if os.path.exists(all_latents_path):
        all_latents = np.load(all_latents_path, allow_pickle=True)
        all_latents = torch.tensor(all_latents)
    else:
        vae = vae.to(device)
        batch_size = 8
        all_latents = []
        start_iid_list = list(range(0, len(img_dataset), batch_size))
        last_iid = len(img_dataset)
        with torch.no_grad():
            for start_iid in tqdm(start_iid_list):
                end_iid = start_iid + batch_size if start_iid + batch_size < last_iid else last_iid
                batch_imgs = []
                for i in range(start_iid, end_iid):
                    batch_imgs.append(img_dataset[i])
                batch_imgs = torch.stack(batch_imgs, dim=0).to(memory_format=torch.contiguous_format).float().to(device)
                batch_latents = vae.encode(batch_imgs).latent_dist.mode() * vae.config.scaling_factor
                all_latents.append(batch_latents)
            all_latents = torch.cat(all_latents, dim=0)
            all_latents = all_latents.cpu()
            np.save(all_latents_path, np.array(all_latents))


    hist_latents = {}
    for uid in history:
        if uid not in hist_latents:
            hist_latents[uid] = {}
        for cate in history[uid]:
            iids = history[uid][cate]
            hist_img_latents = all_latents[iids]
            hist_latents[uid][cate] = hist_img_latents.mean(dim=0)
    
    hist_latents["null"] = all_latents[0]
        
    outfit_category = []
    for category in data["category"]:
        category = torch.tensor(category)
        outfit_category.append(category)
    data["category"] = outfit_category

    outfits = []
    for outfit in data["outfits"]:
        outfit = torch.tensor(outfit).long()  # use as indices
        outfits.append(outfit)
    data["outfits"] = outfits
    
    return (data, hist_latents)

