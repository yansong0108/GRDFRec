import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

import open_clip
from torch_geometric.nn import GATConv

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin


class CompatibilityGAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4, dropout=0.2):
        super(CompatibilityGAT, self).__init__()
        # Layer 1
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout, concat=True)
        # Layer 2
        self.gat2 = GATConv(hidden_channels * heads, out_channels, heads=1, dropout=dropout, concat=False)

    def forward(self, x, edge_index):
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = self.gat2(x, edge_index)
        return x


class GATProjector(nn.Module):
    """
    将 GAT 输出 (例如 512维) 映射到 Text Embedding 维度 (例如 768/1024维)
    [UPGRADE] 支持多 Token 输出 (num_tokens)，增加信息带宽。
    """
    def __init__(self, gat_dim, visual_dim, output_dim, num_tokens=4):
        super().__init__()
        self.num_tokens = num_tokens
        self.output_dim = output_dim

        self.input_dim = visual_dim + gat_dim
        
        self.input_norm = nn.LayerNorm(self.input_dim)

        self.net = nn.Sequential(
            nn.Linear(self.input_dim, output_dim * num_tokens),
            nn.SiLU(),
            nn.Linear(output_dim * num_tokens, output_dim * num_tokens),
        )
        
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        last_linear = self.net[-1]
        nn.init.zeros_(last_linear.weight)
        nn.init.zeros_(last_linear.bias)
            
    def forward(self, original_feats, gat_feats):
        # x: [Batch, GAT_Dim] 
        if getattr(self, 'debug_gnn', False):
            print(f"\n[GAT Projector] original_feats.shape: {original_feats.shape}, gat_feats.shape: {gat_feats.shape}")
        x = torch.cat([original_feats, gat_feats], dim=-1)

        if getattr(self, 'debug_gnn', False):
            print(f"[GAT Projector] Concatenated x.shape: {x.shape}")
        x = self.input_norm(x) # Normalize first
        x = self.net(x)
        return x.view(-1, self.num_tokens, self.output_dim)


class MutualEncoder(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(self, cate_num, cate_emb_size, latent_channels, latent_size, hid_dim):
        super().__init__()
        self.category_embedding = nn.Embedding(cate_num, cate_emb_size)
        self.latent_channels = latent_channels
        self.latent_size = latent_size
        self.mlp = nn.Sequential(
            nn.Linear(latent_channels * latent_size * latent_size, hid_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(hid_dim, latent_channels * latent_size * latent_size),
            nn.Tanh()  # restrict the output in [-1., 1.]
        )
    
    def forward(self, mutual_emb):
        bsz = mutual_emb.shape[0]
        mutual_emb = mutual_emb.view(bsz, -1)
        mutual_guidance = self.mlp(mutual_emb)
        mutual_guidance = mutual_guidance.view(
            bsz,
            self.latent_channels,
            self.latent_size,
            self.latent_size
        )
        return mutual_guidance


class ClipFeatureExtractor(nn.Module):
    """
    A feature extractor for images using a model from the open_clip library.
    Used for calculating Personalization/Compatibility metrics.
    """
    def __init__(self, model_name='ViT-H-14', pretrained_path='./laion2b-s32b-b79K/open_clip_pytorch_model.bin'):
        super().__init__()
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, 
            pretrained=pretrained_path
        )
        self.to_pil = transforms.ToPILImage()

    def forward(self, x):
        return self.encode_image(x)

    def encode_image(self, x):
        model_device = next(self.model.parameters()).device
        model_dtype = next(self.model.parameters()).dtype
        
        if x.device != model_device:
            x = x.to(model_device)

        if x.dim() == 4:  # Images [N, C, H, W]
            if x.shape[1] == 4:
                x = x[:, :3, :, :]
            processed_images = torch.stack([
                self.preprocess(self.to_pil(img)) for img in x
            ]).to(model_device)
            
            with torch.no_grad():
                features = self.model.encode_image(
                    processed_images.to(dtype=model_dtype)
                )
            
            features = features / features.norm(dim=-1, keepdim=True)
            return features
            
        elif x.dim() == 2:
            return x
        else:
            raise ValueError(f"Unsupported input dim: {x.dim()}")
