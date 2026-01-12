from .clip_extractor import CLIPFeatureExtractor
from .coop_extractor import CoOpFeatureExtractor

def get_extractor(config):
    """Extract NORMALIZED image, text and  features from the batch using the model specified in config."""

    if config.feature_extractor == 'clip': # Pre-trained CLIP
        extractor = CLIPFeatureExtractor(config)
        return extractor
    if config.feature_extractor == 'coop': # Pre-trained CoOp
        extractor = CoOpFeatureExtractor(config)
        return extractor
    else:
        raise NotImplementedError(f"Feature extractor '{config.feature_extractor}' is not implemented.")