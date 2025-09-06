import torch
import torch.nn as nn
from model.prompt_net import *
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import json
from model.descriptors import *
from collections import OrderedDict
from torch.nn import functional as F
from PIL import Image
import torch
from torchvision import transforms
_tokenizer = _Tokenizer()
import numpy as np
from einops import repeat
from collections import OrderedDict
import os


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


def exists(val):
    return val is not None

class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim=None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x_q, x_k=None, x_v=None, **kwargs):
        x_q = self.norm(x_q)

        if exists(x_k):
            x_k = self.norm_context(x_k)
        else:
            x_k = x_q

        return self.fn(x_q, x_k, x_v, **kwargs)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim)
        )
    def forward(self, x):
        return self.net(x)
    
    
class ImageEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()

        self.conv1 = clip_model.conv1
        self.class_embedding = clip_model.class_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.ln_pre = clip_model.ln_pre
        self.transformer = clip_model.transformer
        self.ln_post = clip_model.ln_post
        self.proj = clip_model.proj

    def forward(self, x: torch.Tensor, crossattn=None, textfeat_ctx=None, apply_lora=None, cross_loraparams=None, use_lora=None):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        #x = torch.cat([x, vis_ctx], dim=1)  
        if textfeat_ctx is not None:
            x_patch = x[:, 1:, :]
            if use_lora == False:
                x_patch = crossattn(x_patch, textfeat_ctx, textfeat_ctx)
            else:
                queryA, queryB, keyA, keyB, valueA, valueB = cross_loraparams()              
                x_patch = crossattn(apply_lora(x_patch, queryA, queryB), apply_lora(textfeat_ctx, keyA, keyB), apply_lora(textfeat_ctx, valueA, valueB))
            x_patch = x_patch.permute(0, 2, 1)
            vis_ctx = nn.AdaptiveAvgPool1d(4)(x_patch)
            vis_ctx = vis_ctx.permute(0, 2, 1)
            #vis_ctx = crossattn(textfeat_ctx, vis_ctx)
            x = torch.cat([x, vis_ctx], dim=1) 
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


def aggregate_similarity(similarity_matrix_chunk, aggregation_method='mean'):
    if aggregation_method == 'max': return similarity_matrix_chunk.max(dim=1)[0]
    elif aggregation_method == 'sum': return similarity_matrix_chunk.sum(dim=1)
    elif aggregation_method == 'mean': return similarity_matrix_chunk.mean(dim=1)
    else: raise ValueError("Unknown aggregate_similarity")


class CrossLoRAparams(nn.Module):
    def __init__(
            self, dtype
    ):
        super().__init__()
        self.cross_queryA = nn.Parameter(torch.randn(768, 8, dtype=dtype) * 0.02)
        self.cross_queryB = nn.Parameter(torch.randn(8, 768, dtype=dtype) * 0.02)
        
        self.cross_keyA = nn.Parameter(torch.randn(768, 8, dtype=dtype) * 0.02)
        self.cross_keyB = nn.Parameter(torch.randn(8, 768, dtype=dtype) * 0.02)
        
        self.cross_valueA = nn.Parameter(torch.randn(768, 8, dtype=dtype) * 0.02)
        self.cross_valueB = nn.Parameter(torch.randn(8, 768, dtype=dtype) * 0.02)
        
    def forward(self):
        return self.cross_queryA, self.cross_queryB, self.cross_keyA, self.cross_keyB, self.cross_valueA, self.cross_valueB


class CrossAttention(nn.Module):
    def __init__(
            self,
            latent_dim,
            kv_dim,
            cross_heads=4,
            seq_dropout_prob=0.
    ):
        super().__init__()
        self.seq_dropout_prob = seq_dropout_prob

        self.cross_attend_blocks = nn.ModuleList([
            PreNorm(latent_dim,
                    nn.MultiheadAttention(latent_dim, num_heads=cross_heads, kdim=kv_dim, vdim=kv_dim,
                                          dropout=seq_dropout_prob, batch_first=True),
                    context_dim=kv_dim),
            FeedForward(latent_dim)])

    def forward(self, query, key, value, mask=None):
        cross_attn, cross_ff = self.cross_attend_blocks
        x, _ = cross_attn(query, key, value, key_padding_mask=mask)
        x = cross_ff(x)+x
        return x  
        
            
class PromptLearner(nn.Module):
    def __init__(self, cfg, clip_model):
        super().__init__()
        
        n_ctx = cfg.MODEL.N_CTX
        self.n_ctx = n_ctx
        dtype = clip_model.dtype
        
        self.cross_loraparams = CrossLoRAparams(dtype)
        
        self.crossattn = CrossAttention(
            latent_dim=768,
            kv_dim=768,
            cross_heads= 4,
        )
          
        self.linear = nn.Linear(512,768)
        
        self.crossattn.half()
        self.linear.half()
        self.use_lora = False  # Initially False

    def apply_lora(self, x, A, B):
        """Applies LoRA transformation when use_lora is enabled"""
        if self.use_lora:
            return x + x @ A @ B
        return x   

    def forward(self, text_feat):
        textfeat_ctx = self.linear(text_feat)
        crossattn = self.crossattn
        apply_lora = self.apply_lora
        cross_loraparams = self.cross_loraparams
        use_lora = self.use_lora
        
        return crossattn, textfeat_ctx, apply_lora, cross_loraparams, use_lora

    
def truncate_text(text, max_length):
    return text[:max_length]

        
def compute_description_encodings(model, gpt_descriptions, device):
    description_encodings = OrderedDict()
    for k, v in gpt_descriptions.items():
        truncated_descriptions = [truncate_text(d, 77) for d in v]
        tokens = clip.tokenize(truncated_descriptions).to(device)
        description_encodings[k] = F.normalize(model.encode_text(tokens))
    return description_encodings
 
flip_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5)
])
           
class FedMVP(nn.Module):
    def __init__(self, cfg, clip_model,  device='cuda:7'):
        super().__init__()
        self.cfg = cfg
        self.dtype = clip_model.dtype
        self.device = device
        self.text_encoder = TextEncoder(clip_model)
        self.prompt_learner = PromptLearner(cfg, clip_model)
        self.clip_model_zs = clip_model
    
    
    def forward(self, image, classnames, dataname):
        
        if dataname == 'Caltech101':
            gpt_descriptions = load_gpt_descriptions('attributes/caltech101.json', classnames, dataname)
            descriptions = load_gpt_descriptions('attributes/caltech101.json', classnames, dataname)     
                               
        if dataname == 'OxfordFlowers':
            gpt_descriptions = load_gpt_descriptions('attributes/oxford_flowers.json', classnames, dataname)
            descriptions = load_gpt_descriptions('attributes/oxford_flowers.json', classnames, dataname)
            
        if dataname == 'FGVCAircraft':
            gpt_descriptions = load_gpt_descriptions('attributes/fgvc_aircraft.json', classnames, dataname)
            descriptions = load_gpt_descriptions('attributes/fgvc_aircraft.json', classnames, dataname)
            
        if dataname == 'UCF101':
            gpt_descriptions = load_gpt_descriptions('attributes/ucf101.json', classnames, dataname)
            descriptions = load_gpt_descriptions('attributes/ucf101.json', classnames, dataname)
                                            
        if dataname == 'OxfordPets':
            gpt_descriptions = load_gpt_descriptions('attributes/oxford_pets.json', classnames, dataname)
            descriptions = load_gpt_descriptions('attributes/oxford_pets.json', classnames, dataname)
                                           
        if dataname == 'Food101':
            gpt_descriptions = load_gpt_descriptions('attributes/food101.json', classnames, dataname)
            descriptions = load_gpt_descriptions('attributes/food101.json', classnames, dataname)
            
        if dataname == 'DescribableTextures':
            gpt_descriptions = load_gpt_descriptions('attributes/dtd.json', classnames, dataname)
            descriptions = load_gpt_descriptions('attributes/dtd.json', classnames, dataname)
            
        if dataname =='StanfordCars':
            gpt_descriptions = load_gpt_descriptions('attributes/stanford_cars.json', classnames, dataname)
            descriptions = load_gpt_descriptions('attributes/stanford_cars.json', classnames, dataname)
                                      
        if dataname == 'SUN397':
            gpt_descriptions = load_gpt_descriptions('attributes/sun397.json', classnames, dataname)
            descriptions = load_gpt_descriptions('attributes/sun397.json', classnames, dataname)
        
        if dataname in ['PACS_artpainting', 'PACS_cartoon', 'PACS_photo', 'PACS_sketch']:
            gpt_descriptions = load_gpt_descriptions('attributes/pacs.json', classnames, dataname)
            descriptions = load_gpt_descriptions('attributes/pacs.json', classnames, dataname) 
            
        if dataname in ['OfficeHome_art', 'OfficeHome_clipart', 'OfficeHome_product', 'OfficeHome_realworld']:
            gpt_descriptions = load_gpt_descriptions('attributes/officehome.json', classnames, dataname)
            descriptions = load_descriptions('attributes/officehome.json', classnames, dataname)
        
        if dataname in ['VLCS_CALTECH', 'VLCS_LABELME', 'VLCS_PASCAL', 'VLCS_SUN']:
            gpt_descriptions = load_gpt_descriptions('attributes/vlcs.json', classnames, dataname)
            descriptions = load_gpt_descriptions('attributes/vlcs.json', classnames, dataname)
        
        if dataname in ['TerraIncognita_L38', 'TerraIncognita_L43', 'TerraIncognita_L46', 'TerraIncognita_L100']:
            gpt_descriptions = load_gpt_descriptions('attributes/terraincognita.json', classnames, dataname)
            descriptions = load_descriptions('attributes/terraincognita.json', classnames, dataname)
            
        if dataname in ['DomainNet_clipart', 'DomainNet_infograph', 'DomainNet_painting', 'DomainNet_quickdraw', 'DomainNet_real', 'DomainNet_sketch']:
            gpt_descriptions = load_gpt_descriptions('attributes/domainnet.json', classnames, dataname)
            descriptions = load_gpt_descriptions('attributes/domainnet.json', classnames, dataname)
     
                
        description_encodings = compute_description_encodings(self.clip_model_zs, gpt_descriptions, self.device)
        description_loads = compute_description_encodings(self.clip_model_zs, descriptions, self.device)
         
        
        image_description_similarity = [None]*len(classnames)
        image_description_similarity_cumulative = [None]*len(classnames)
        
        text_description_similarity = [None]*len(classnames)
        text_description_similarity_cumulative = [None]*len(classnames)
        logit_scale = self.clip_model_zs.logit_scale.exp()
        
        text_feat = torch.stack(list(description_loads.values()), dim=0)
        text_feat = text_feat.reshape(text_feat.shape[0]*text_feat.shape[1], text_feat.shape[2])
        text_feat = text_feat.expand(image.shape[0],-1,-1)
        
        crossattn, textfeat_ctx, apply_lora, cross_loraparams, use_lora = self.prompt_learner(text_feat)
        
        image_features = self.image_encoder(image.type(self.dtype), crossattn, textfeat_ctx.type(self.dtype), apply_lora, cross_loraparams, use_lora)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        batch_image_features = self.clip_model_zs.encode_image(flip_transform(image.type(self.dtype)))
        batch_image_features = batch_image_features / batch_image_features.norm(dim=-1, keepdim=True)
        
        for i, (k, v) in enumerate(description_encodings.items()):
            image_dot_product_matrix = logit_scale * image_features @ v.t()
        
            image_description_similarity[i] = image_dot_product_matrix
            image_description_similarity_cumulative[i] = aggregate_similarity(image_description_similarity[i])
            
        image_logits = torch.stack(image_description_similarity_cumulative, dim=1)
        vis_cos = torch.nn.CosineSimilarity(dim=1,eps=1e-07)
        vis_score = vis_cos(image_features, batch_image_features)
        vis_score = 1.0-torch.mean(vis_score)
        
        return image_logits, vis_score

