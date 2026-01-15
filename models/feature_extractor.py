from .clip_extractor import CLIPFeatureExtractor
from .coop_extractor import CoOpFeatureExtractor
from .adapter_extractor import AdapterFeatureExtractor
from .cocoop_extractor import CoCoOpFeatureExtractor
def get_extractor(config):
    """Extract NORMALIZED image, text and  features from the batch using the model specified in config."""

    if config.feature_extractor == 'clip': # Pre-trained CLIP
        extractor = CLIPFeatureExtractor(config)
    if config.feature_extractor == 'coop': # Pre-trained CoOp
        extractor = CoOpFeatureExtractor(config)
    if config.feature_extractor == 'adapter': # Pre-trained Adapter
        extractor = AdapterFeatureExtractor(config)
    if config.feature_extractor == 'cocoop': # Pre-trained CoCoOp
        extractor = CoCoOpFeatureExtractor(config)
    else:
        raise NotImplementedError(f"Feature extractor '{config.feature_extractor}' is not implemented.")
    
    return extractor