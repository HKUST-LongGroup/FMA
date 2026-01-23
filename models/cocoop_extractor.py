import torch
import torch.nn as nn
import clip
from collections import OrderedDict
from .utils import CLS2DIR
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from einops import pack, repeat, rearrange

_tokenizer = _Tokenizer()

# For acceleration, we delete all 'for loops' in the original code and use batch matrix operations.
class PromptLearner(nn.Module):
    def __init__(self, cfg, clip_model):
        super().__init__()
      
        self.n_cls = len(cfg.classnames)
        vis_dim = clip_model.visual.output_dim
        prompt_prefix = 'a photo of a'
        self.n_ctx = len(prompt_prefix.split(" "))

        self.meta_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(vis_dim // 16, vis_dim))
        ])) 

        self.ctx = nn.Parameter(torch.randn(self.n_ctx, vis_dim))
        classnames = [name.replace("_", " ") for name in cfg.classnames]
        prompts =[prompt_prefix + ' ' + cname + '.' for cname in classnames]

        self.tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(cfg.device) # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(self.tokenized_prompts).float()
        
        # These token vectors will be saved when in save_model(),
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + self.n_ctx :, :])  # CLS, EOS

        self.load_cocoop(cfg)

        # don't update the feature extractor weights
        for param in self.parameters():
            param.requires_grad = False
    
    def load_cocoop(self, cfg):
        ckpt_path = f'./checkpoints/{CLS2DIR[cfg.dataset]}/{cfg.feature_extractor}_vit_b16_{cfg.num_shots}s.pth'
        print(f"Loading CoCoOp context weights from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.load_state_dict(ckpt['state_dict'])
        self.to(cfg.device)

    

    def forward(self, im_features):
   
        bias = self.meta_net(im_features)  # (batch, ctx_dim)
        bias = repeat(bias, 'b d -> b n d', n=self.n_ctx) # (batch, n_ctx, ctx_dim)
        ctx = repeat(self.ctx, 'n d -> b n d', b=im_features.size(0))  # (batch, n_ctx, ctx_dim)
        ctx_shifted = ctx + bias           # (batch, n_ctx, ctx_dim)
        ctx_shifted = repeat(ctx_shifted, 'b n d -> b c n d', c=self.n_cls)  # (batch, n_cls, n_ctx, ctx_dim)

        prefix = repeat(self.token_prefix, 'c 1 d -> b c 1 d', b=im_features.size(0))  # (batch, n_cls, 1, ctx_dim)
        suffix = repeat(self.token_suffix, 'c s d -> b c s d', b=im_features.size(0))  # (batch, n_cls, s, ctx_dim)

        prompts, _ = pack([prefix, ctx_shifted, suffix], 'b c * d')
       
        
        return prompts # (batch_size, n_cls, 77, ctx_dim)
    
class CoCoOpFeatureExtractor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.clip_model, self.clip_processor = clip.load(config.clip_type, device=config.device)
        self.device = config.device
        self.clip_model.to(self.device)
        self.prompt_learner = PromptLearner(config, self.clip_model) # load a pretrained CoCoOp prompt learner
   
        
    # the class embeddings are instance-specific, so we cannot cache them in init function (like CoOp)
    @torch.no_grad()
    def forward(self, images, labels):
        images, labels = images.to(self.device), labels.to(self.device)
        image_features = self.clip_model.encode_image(images) # preprocess is integrated in dataloader  (batch_size, dim)
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

        # obtain class embeddings and text features for each instance
        prompts = self.prompt_learner(image_features)  # (batch_size, n_cls, 77, ctx_dim)
        prompts = rearrange(prompts, 'b c n d -> (b c) n d')  # (batch_size*n_cls, 77, ctx_dim)
        x = prompts.type(self.clip_model.dtype)
        x = x + self.clip_model.positional_embedding.type(self.clip_model.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND, N = batch_size*n_cls
        x = self.clip_model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD, N = batch_size*n_cls
        x = self.clip_model.ln_final(x).type(self.clip_model.dtype)

        x = rearrange(x, '(b c) n d -> b c n d', b=images.size(0))  # (batch_size, n_cls, 77, dim)
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        class_embeddings = x[:,torch.arange(x.shape[1]), self.prompt_learner.tokenized_prompts.argmax(dim=-1),:] @ self.clip_model.text_projection  # (batch_size, n_cls, dim)

        class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)

        text_features = class_embeddings[torch.arange(images.size(0)), labels]  # (batch_size, dim)

        return image_features, text_features, class_embeddings

    

