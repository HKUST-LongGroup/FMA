import torch
import torch.nn as nn
import clip
from .clip_extractor import CUSTOM_TEMPLATES, CLS2DIR



class Adapter(nn.Module):
    def __init__(self, c_in, reduction = 4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = self.fc(x)
        return x   
    
class AdapterFeatureExtractor(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.clip_model, self.clip_processor = clip.load(config.clip_type, device=config.device)
        self.device = config.device
        self.clip_model.to(self.device)
        self.adapter = Adapter(self.clip_model.visual.output_dim)

        # prepare class embeddings
        classnames = [name.replace("_", " ") for name in config.classnames]
        template = CUSTOM_TEMPLATES.get(config.dataset, "a photo of a {}.")
        self.prompts = [template.format(c.replace("_", " ")) for c in classnames]
        with torch.no_grad():
            class_embeddings = self.clip_model.encode_text(clip.tokenize(self.prompts).to(self.device))
            self.class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
        
        self.load_adapter(config)

        # don't update the feature extractor weights (CLIP + Adapter)
        for param in self.parameters():
            param.requires_grad = False
        import pdb; pdb.set_trace()
    
    def load_adapter(self, config):
        ckpt_path = f'./checkpoints/{CLS2DIR[config.dataset]}/{config.feature_extractor}_vit_b16_{config.num_shots}s.pth'
        print(f"Loading Adapter weights from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt['state_dict']
        self.adapter.load_state_dict(state_dict)
        self.adapter.to(self.device)
    
    @torch.no_grad()
    def forward(self, images,labels):
        images, labels = images.to(self.device), labels.to(self.device)
        image_features = self.clip_model.encode_image(images) # preprocess is integrated in dataloader
        x = self.adapter(image_features)
        ratio = 0.2
        image_features = image_features * (1-ratio) + ratio * x

        text_features = self.class_embeddings[labels]
        # Normalize the features
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

        return image_features, text_features, self.class_embeddings
                        