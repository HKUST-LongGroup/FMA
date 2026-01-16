import torch
import torch.nn as nn
import clip
from .utils import PlainMultiheadAttentionLoRA, CLS2DIR
from torch.cuda.amp import autocast

# add lora to all attention places, including text and vision encoder, all layers and qkv.    

def apply_lora(clip_model):
    list_lora_layers = []

    # LoRA on text encoder
    for i, block in enumerate(clip_model.transformer.resblocks):
        for name, children in block.named_children():
            if  isinstance(children, nn.MultiheadAttention):
                qkvlora = PlainMultiheadAttentionLoRA(children, enable_lora=['q','k','v'], r=2, lora_alpha=1, dropout_rate=0.25)
                setattr(block, name, qkvlora) # block.attn = qkvlora
                list_lora_layers.append(qkvlora)
    # LoRA on vision encoder
    for i, block in enumerate(clip_model.visual.transformer.resblocks):
        for name, children in block.named_children():
            if  isinstance(children, nn.MultiheadAttention):
                qkvlora = PlainMultiheadAttentionLoRA(children, enable_lora=['q','k','v'], r=2, lora_alpha=1, dropout_rate=0.25)
                setattr(block, name, qkvlora) # block.attn = qkvlora
                list_lora_layers.append(qkvlora)
    return list_lora_layers

# all templeates are : a photo of a {}
class LoRAFeatureExtractor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.clip_model, self.clip_processor = clip.load(cfg.clip_type, device=cfg.device)
        self.device = cfg.device
        self.clip_model.to(self.device)
        # Additional LoRA-specific initialization can be added here
        self.cfg = cfg
        list_lora_layers = apply_lora(self.clip_model)

        ckpt_path = f'./checkpoints/{CLS2DIR[cfg.dataset]}/{cfg.feature_extractor}_vit_b16_{cfg.num_shots}s.pth'
        self.load_lora(ckpt_path, list_lora_layers)

        # generate class embeddings
        classnames = [name.replace("_", " ") for name in cfg.classnames]
        prompts = [ f"a photo of a {cname}." for cname in classnames]
        with torch.no_grad(), autocast():
            class_embeddings = self.clip_model.encode_text(clip.tokenize(prompts).to(self.device))
            self.class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)

    
        # do not update the parameters of clip model
        for param in self.parameters():
            param.requires_grad = False

    def load_lora(self, ckpt_path, list_lora_layers):

        loaded_data = torch.load(ckpt_path,map_location='cpu')

        weights = loaded_data['weights']
        for i, layer in enumerate(list_lora_layers):
            layer_weights = weights[f'layer_{i}']
            with torch.no_grad():
                layer.q_proj.w_lora_A.copy_(layer_weights['q_proj']['w_lora_A'])
                layer.q_proj.w_lora_B.copy_(layer_weights['q_proj']['w_lora_B'])

                layer.k_proj.w_lora_A.copy_(layer_weights['k_proj']['w_lora_A'])
                layer.k_proj.w_lora_B.copy_(layer_weights['k_proj']['w_lora_B'])

                layer.v_proj.w_lora_A.copy_(layer_weights['v_proj']['w_lora_A'])
                layer.v_proj.w_lora_B.copy_(layer_weights['v_proj']['w_lora_B'])
          

        print(f'LoRA weights loaded from {ckpt_path}')

        self.to(self.cfg.device)
    @torch.no_grad()
    def forward(self, images,labels):
            images, labels = images.to(self.device), labels.to(self.device)
            image_features = self.clip_model.encode_image(images) # preprocess is integrated in dataloader

            text_features = self.class_embeddings[labels]
            # Normalize the features
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

            return image_features, text_features, self.class_embeddings