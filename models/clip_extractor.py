
import torch
import torch.nn as nn
import clip
from .utils import CUSTOM_TEMPLATES

class CLIPFeatureExtractor(nn.Module):
    """Feature extractor using pre-trained CLIP model."""
    def __init__(self, config):
        super(CLIPFeatureExtractor, self).__init__()
        self.clip_model, self.clip_processor = clip.load(config.clip_type, device=config.device)
        self.device = config.device
        self.clip_model.to(self.device)
        classnames = [name.replace("_", " ") for name in config.classnames]
        template = CUSTOM_TEMPLATES.get(config.dataset, "a photo of a {}.")
        self.prompts = [template.format(c.replace("_", " ")) for c in classnames]

        with torch.no_grad():
            class_embeddings = self.clip_model.encode_text(clip.tokenize(self.prompts).to(self.device))
            self.class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)

        # don't update CLIP weights
        for param in self.parameters():
            param.requires_grad = False
    @torch.no_grad()
    def forward(self, images,labels):
        images, labels = images.to(self.device), labels.to(self.device)
        image_features = self.clip_model.encode_image(images) # preprocess is integrated in dataloader

        text_features = self.class_embeddings[labels]
        # Normalize the features
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

        return image_features, text_features, self.class_embeddings