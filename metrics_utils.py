import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import logging
from PIL import Image

logger = logging.getLogger(__name__)

def calculate_metrics_online(outputs, feature_extractor, valid_hist_features, device, resolution=512):
    """
    轻量级在线指标计算 (Debug 版 - 增强兼容性计算)
    """

    def preprocess_for_clip(img_input):
        
        if isinstance(img_input, torch.Tensor):
            img = img_input.clone()
            if img.dim() == 4: img = img.squeeze(0) # Remove batch dim if present
            
            if img.min() < 0:
                img = (img + 1.0) / 2.0
            img = torch.clamp(img, 0.0, 1.0)
            img = F.interpolate(img.unsqueeze(0), size=(224, 224), mode='bicubic', align_corners=False).squeeze(0)
            
            norm = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                                        std=[0.26862954, 0.26130258, 0.27577711])
            return norm(img)

        elif isinstance(img_input, Image.Image):
            trans = transforms.Compose([
                transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                                     std=[0.26862954, 0.26130258, 0.27577711])
            ])
            return trans(img_input)
            
        return None

    personal_scores = []
    compat_scores = []
    
    for uid_key, oid_dict in outputs.items():
        try: uid = int(uid_key)
        except: uid = uid_key
        
        user_hist = None
        if valid_hist_features:
            if uid in valid_hist_features: user_hist = valid_hist_features[uid]
            elif str(uid) in valid_hist_features: user_hist = valid_hist_features[str(uid)]
            elif isinstance(uid, (str, float)) and int(uid) in valid_hist_features: user_hist = valid_hist_features[int(uid)]

        for oid_key, data in oid_dict.items():
            imgs = data.get("images", [])
            cates = data.get("cates", [])
            context_imgs = data.get("context_images", []) # List of Tensors
            
            if not imgs: continue
            gen_pixels = torch.stack([preprocess_for_clip(img) for img in imgs]).to(device)
            with torch.no_grad():
                if hasattr(feature_extractor, 'encode_image'): gen_feats = feature_extractor.encode_image(gen_pixels)
                else: gen_feats = feature_extractor(gen_pixels).pooler_output
            gen_feats = F.normalize(gen_feats, dim=-1)


            if user_hist:
                for idx, cate_val in enumerate(cates):
                    cid = int(cate_val.item()) if hasattr(cate_val, 'item') else int(cate_val)
                    hist_list = None
                    if cid in user_hist: hist_list = user_hist[cid]
                    elif str(cid) in user_hist: hist_list = user_hist[str(cid)]
                    
                    if hist_list is not None and len(hist_list) > 0:
                        hist_feats_t = torch.as_tensor(np.stack(hist_list), device=device, dtype=gen_feats.dtype)
                        if hist_feats_t.dim() == 1: hist_feats_t = hist_feats_t.unsqueeze(0)
                        hist_feats_t = F.normalize(hist_feats_t, dim=-1)
                        
                        sims = torch.matmul(hist_feats_t, gen_feats[idx])
                        k = min(3, sims.numel())
                        score = sims.topk(k).values.mean().item()
                        personal_scores.append(score)

            if len(context_imgs) > 0:

                ctx_pixels = torch.stack([preprocess_for_clip(img) for img in context_imgs]).to(device)
                with torch.no_grad():
                    if hasattr(feature_extractor, 'encode_image'): ctx_feats = feature_extractor.encode_image(ctx_pixels)
                    else: ctx_feats = feature_extractor(ctx_pixels).pooler_output
                ctx_feats = F.normalize(ctx_feats, dim=-1)

                sim_matrix = torch.matmul(gen_feats, ctx_feats.t())
                

                outfit_score = sim_matrix.mean().item()
                compat_scores.append(outfit_score)
                    

            elif gen_feats.size(0) > 1:
                sim_mat = torch.matmul(gen_feats, gen_feats.t())
                mask = torch.triu(torch.ones_like(sim_mat), diagonal=1).bool()
                pairwise = sim_mat[mask]
                if pairwise.numel() > 0:
                    compat_scores.append(pairwise.mean().item())

    p_mean = np.mean(personal_scores) if personal_scores else 0.0
    c_mean = np.mean(compat_scores) if compat_scores else 0.0
    
    logger.info(f"[Metrics Summary] Personalization: {len(personal_scores)} items. Compatibility: {len(compat_scores)} outfits.")
    
    return {
        "Personalization": p_mean,
        "Compatibility": c_mean
    }