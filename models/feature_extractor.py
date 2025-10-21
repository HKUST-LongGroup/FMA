from .clip_extractor import CLIPFeatureExtractor

def get_extractor(config):
    """Extract NORMALIZED image, text and  features from the batch using the model specified in config."""

    if config.feature_extractor == 'clip': # Pre-trained CLIP
        extractor = CLIPFeatureExtractor(config)
        return extractor
    else:
        raise NotImplementedError(f"Feature extractor '{config.feature_extractor}' is not implemented.")