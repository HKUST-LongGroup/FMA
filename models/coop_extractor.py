
import torch
import torch.nn as nn
import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from einops import pack, repeat

_tokenizer = _Tokenizer()

cls2dir = {
    "OxfordPets": "oxford_pets",
    "OxfordFlowers": "oxford_flowers",
    "FGVCAircraft": "fgvc_aircraft",
    "DescribableTextures": "dtd",
    "EuroSAT": "eurosat",
    "StanfordCars": "stanford_cars",
    "Food101": "food101",
    "SUN397": "sun397",
    "Caltech101": "caltech101",
    "UCF101": "ucf101",
    "ImageNet": "imagenet",}
# From CoOp, but ONLY support the setting: n_ctx=16, ctx_init=false, csc=False
class PromptLearner(nn.Module):
    def __init__(self, cfg,  clip_model):
        super().__init__()
        n_cls = len(cfg.classnames)
        n_ctx = 16
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]

     
        ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype,device=cfg.device)
        nn.init.normal_(ctx_vectors, std=0.02)
        prompt_prefix = " ".join(["X"] * n_ctx)


        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in cfg.classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

  
        tokenized_prompts = clip.tokenize(prompts).to(cfg.device)  # [n_cls, dim]
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype) # input to the transformer

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

        self.load_ctx(cfg)

    def load_ctx(self,cfg):
        ckpt_path = f'./checkpoints/{cls2dir[cfg.dataset]}/{cfg.feature_extractor}/vit_b16_{cfg.num_shots}s.pth'
        print(f"Loading CoOp context weights from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt['state_dict']
        self.ctx.data = state_dict['ctx'].to(self.ctx.device)
        self.to(cfg.device)

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = repeat(ctx, 'n d -> c n d', c=self.n_cls)

        prompts, _ = pack([self.token_prefix, ctx, self.token_suffix], 'n * d')
        return prompts


class CoOpFeatureExtractor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.clip_model, self.clip_processor = clip.load(config.clip_type, device=config.device)
        self.device = config.device
        self.clip_model.to(self.device)
        self.prompt_learner = PromptLearner(config, self.clip_model) # load a pretrained prompt learner
   
        
        with torch.no_grad():
            # from encode_text function in CLIP
            prompts = self.prompt_learner()  # [n_cls, n_ctx + 3, d_model]
            x = prompts.type(self.clip_model.dtype)
            x = x + self.clip_model.positional_embedding.type(self.clip_model.dtype)
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.clip_model.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = self.clip_model.ln_final(x).type(self.clip_model.dtype)

            # x.shape = [batch_size, n_ctx, transformer.width]
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            class_embeddings = x[torch.arange(x.shape[0]), self.prompt_learner.tokenized_prompts.argmax(dim=-1)] @ self.clip_model.text_projection
            self.class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
        
    @torch.no_grad()
    def forward(self, images,labels):
        images, labels = images.to(self.device), labels.to(self.device)
        image_features = self.clip_model.encode_image(images) # preprocess is integrated in dataloader
        text_features = self.class_embeddings[labels]
        # Normalize the features
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

        return image_features, text_features, self.class_embeddings