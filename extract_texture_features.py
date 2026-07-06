import os
import numpy as np
import torch
from PIL import Image
import open_clip
from tqdm import tqdm

ALL_PATHS_FILE = "../datasets/polyvore/all_item_image_paths.npy"

TEXTURE_ROOT_DIR = "../polyvore_folder_path_texture_output/291x291" 

OUTPUT_DIR = "../datasets/polyvore/processed"
OUTPUT_FILENAME = "texture_features_clip.npy"

CLIP_MODEL_NAME = 'ViT-H-14'
CLIP_PRETRAINED_PATH = './laion2b-s32b-b79K/open_clip_pytorch_model.bin' # 请确保此路径正确

# ==========================================================

def main():
    # 0. 准备环境
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"正在加载 CLIP 模型 ({CLIP_MODEL_NAME})...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL_NAME, 
        pretrained=CLIP_PRETRAINED_PATH
    )
    model.to(device)
    model.eval()

    all_image_paths = np.load(ALL_PATHS_FILE, allow_pickle=True)
    total_items = len(all_image_paths)

    texture_feat_dict = {} # 结果容器: {iid: feature}
    missing_count = 0
    success_count = 0

    BATCH_SIZE = 128 
    batch_images = []
    batch_iids = []
    
    for iid, relative_path in tqdm(enumerate(all_image_paths), total=total_items, unit="img"):
        
        
        if relative_path == 'empty_image.png':
            continue

        try:
            rel_dir = os.path.dirname(relative_path)
            file_name_no_ext = os.path.splitext(os.path.basename(relative_path))[0]
            texture_filename = f"{file_name_no_ext}_micro_texture.png"
            
            texture_full_path = os.path.join(TEXTURE_ROOT_DIR, rel_dir, texture_filename)
            
            if os.path.exists(texture_full_path):
                image = Image.open(texture_full_path).convert("RGB")
                preprocessed_image = preprocess(image) # 预处理 (Tensor)
                
                batch_images.append(preprocessed_image)
                batch_iids.append(iid)
            else:
                missing_count += 1
        
        except Exception as e:
            # 读取出错
            missing_count += 1
            continue

        if len(batch_images) >= BATCH_SIZE or (iid == total_items - 1 and len(batch_images) > 0):
            with torch.no_grad(), torch.cuda.amp.autocast():
                image_tensor = torch.stack(batch_images).to(device)
                features = model.encode_image(image_tensor)
                features = features / features.norm(dim=-1, keepdim=True)
                features_np = features.cpu().numpy().astype(np.float32) # float32 省空间
            
            for k, feat in enumerate(features_np):
                current_iid = batch_iids[k]
                texture_feat_dict[current_iid] = feat
                success_count += 1
            
            batch_images = []
            batch_iids = []

    # 4. 保存结果
    save_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    print("\n" + "="*40)
    print(f"🎉 提取完成！")
    print(f" - 总物品数: {total_items}")
    print(f" - 成功提取: {success_count}")
    print(f" - 缺失/跳过: {missing_count}")
    print(f"⏳ 正在保存到: {save_path} ...")
    
    np.save(save_path, texture_feat_dict)
    print(f"✅ 保存成功！文件大小约为: {os.path.getsize(save_path) / (1024*1024):.2f} MB")
    print("="*40)

if __name__ == "__main__":
    main()