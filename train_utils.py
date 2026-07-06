import logging

import numpy as np
import torch
from torch_geometric.data import Data, Batch


logger = logging.getLogger(__name__)


def get_item_feature(item_id, item_features, feature_dim=1024): 
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
    
    combined_dim = feature_dim * 2 
    
    for i in range(bsz):
        uid = int(uids[i].item())
        outfit_items = outfits[i].tolist() 
        current_cates = outfit_cates[i].tolist()
        
        slot_feats = []
        hist_feats = []
        slot_to_hist_map = {} 

        for item_id in outfit_items:
            iid = int(item_id)
            if iid == 0: 
                slot_feats.append(np.zeros(combined_dim, dtype=np.float32))
            else:
                g_feat = get_item_feature(iid, item_features, feature_dim)
                t_feat = get_item_feature(iid, texture_features, feature_dim)
                combined = np.concatenate([g_feat, t_feat])
                slot_feats.append(combined)
        
        num_slots = len(slot_feats)
        
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
                    g_feat_h = get_item_feature(hid, item_features, feature_dim)
                    t_feat_h = get_item_feature(hid, texture_features, feature_dim)
                    
                    combined_hist = np.concatenate([g_feat_h, t_feat_h])
                    
                    hist_feats.append(combined_hist)
                    added_count += 1
                
                if added_count > 0:
                    current_hist_indices = list(range(start_idx, start_idx + added_count))
            
            slot_to_hist_map[slot_idx] = current_hist_indices
            
        all_feats = slot_feats + hist_feats
        
        if len(all_feats) == 0:
            x = torch.zeros((num_slots, combined_dim), dtype=torch.float32)
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        else:
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


def freeze_unet_backbone(unet):
    """
    Freezes the entire UNet except for:
    1. 'attn2' (Cross-Attention) layers: where GAT/Text injection happens.
    2. 'conv_in': because we modified the input channels.
    """
    logger.info("Freezing UNet Backbone (ResNets, Down/Up samplers)... Only training Attention & ConvIn.")
    
    trainable_count = 0
    frozen_count = 0
    
    for name, param in unet.named_parameters():
        if "attn2" in name:
            param.requires_grad = True
            trainable_count += 1
        elif "conv_in" in name:
            param.requires_grad = True
            trainable_count += 1
        else:
            param.requires_grad = False
            frozen_count += 1
            
    logger.info(f"UNet Frozen Layers: {frozen_count}, Trainable Layers: {trainable_count}")
