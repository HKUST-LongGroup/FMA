import torch
from models.fm import DeepFlowMatchingNet
from torch.utils.data import DataLoader
import clip
import torch.nn.functional as F
from config import DefaultConfig
from config import set_seed
import sys
from torch.cuda.amp import autocast
from models.feature_extractor import get_extractor
from datasets import build_dataset
from einops import einsum

@torch.no_grad()
def test_fma(model, data_loader,feat_extractor,steps,stepsize,cfg):
    # Perform Early Stopping Inference
    device = cfg.device
    model.eval()
    correct = 0
    total = 0
    
    for images,labels in data_loader:
        with autocast():
            image_features, _, class_embeddings = feat_extractor(images,labels)
            transfer_features = image_features
            # perform inference steps
            t = torch.zeros((image_features.shape[0],1),device=device)
            for i in range(steps):
                drift = model(transfer_features, t)
                transfer_features = transfer_features +drift*stepsize
                t = t + stepsize
            # calculate accuracy
            transfer_features = transfer_features / transfer_features.norm(dim=-1, keepdim=True)
            if cfg.feature_extractor == 'cocoop':
                similarities = einsum(transfer_features, class_embeddings, 'batch dim, batch cls dim -> batch cls')
            else:
                similarities = (transfer_features @ class_embeddings.T).softmax(dim=-1) #[N,C]
            predicted_labels = similarities.argmax(dim=-1).cpu()
            correct += (predicted_labels == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    print(f'Steps:{steps}, StepSize: {stepsize}, Acc: {accuracy:.4f}, Correct:{correct}, Total:{total}')
    return accuracy



def get_prob_matrix(features, class_embeddings, temperature):
    """
    Compute the probability matrix based on cosine similarities.

    Args:
        features (torch.Tensor): Tensor of shape (N, D) representing N feature vectors.
        class_embeddings (torch.Tensor): Tensor of shape (C, D) representing C class embeddings.
        temperature (float): Temperature scaling factor.

    Returns:
        torch.Tensor: Probability matrix of shape (N, C).
    """
    features = features / features.norm(dim=-1, keepdim=True)  # Normalize features
    class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)

    # Compute cosine similarities
    similarities = torch.matmul(features, class_embeddings.T)  # Shape: (N, C)

    # Scale by temperature
    logits = similarities * temperature

    # Apply softmax to get probabilities
    probs = F.softmax(logits, dim=-1)

    return probs

@torch.no_grad()
def test_fma_adaptive(model, data_loader,feat_extractor,steps,stepsize,cfg):
    # Perform Early Stopping Inference
    device = cfg.device
    model.eval()
    correct = 0
    total = 0
    total_steps = 0
    for images,labels in data_loader:
        with autocast():
            image_features, _, class_embeddings = feat_extractor(images,labels)
            transfer_features = image_features #[ B, dim]
            # perform inference steps
            t = torch.zeros((image_features.shape[0],1),device=device)
            prob = torch.zeros((image_features.shape[0],class_embeddings.shape[0]),device=device)
            for i in range(steps):
                # prob_matrix: [B,C]
                prob_matrix = get_prob_matrix(transfer_features, class_embeddings, feat_extractor.clip_model.logit_scale.exp().item())
                if i in cfg.sample_steps:
                    prob =  (prob + prob_matrix) 
                drift = model(transfer_features, t)
                transfer_features = transfer_features +drift*stepsize 

                t = t + stepsize
            # calculate accuracy
            # transfer_features = transfer_features / transfer_features.norm(dim=-1, keepdim=True)
            # if cfg.feature_extractor == 'cocoop':
            #     similarities = einsum(transfer_features, class_embeddings, 'batch dim, batch cls dim -> batch cls')
            # else:
            #     similarities = (transfer_features @ class_embeddings.T).softmax(dim=-1) #[N,C]
            similarities = prob
            predicted_labels = similarities.argmax(dim=-1).cpu()
            correct += (predicted_labels == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total
    print(f'Steps:{steps}, StepSize: {stepsize}, Acc: {accuracy:.4f}, Correct:{correct}, Total:{total}')
    return accuracy


if __name__ == "__main__":
    timestamp = sys.argv[1] # each timestamp corresponding to an exp.

    print(f'Loading Checkpoint at Experiment with timestamp: {timestamp }')
    config_path = f"./checkpoints/exp/{timestamp}/config.json"
    model_path = f"./checkpoints/exp//{timestamp}/model.pth"
    cfg = DefaultConfig.from_json(config_path)
    cfg.sample_steps = [0,4,8]
    print(cfg)
    set_seed(cfg.seed)

     # Prepare dataset
    dataset = build_dataset(cfg)
    cfg.classnames = dataset.classnames

    
    train_loader = DataLoader(dataset.train_x, batch_size=cfg.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(dataset.test, batch_size=cfg.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    val_loader = DataLoader(dataset.val, batch_size=cfg.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    # Initialize the model
    clip_model, _ = clip.load(cfg.clip_type, device=cfg.device, jit=False)
    dim = clip_model.visual.output_dim
    print(f"CLIP model output dimension: {dim}")
    model = DeepFlowMatchingNet(in_channels=dim,model_channels=dim, out_channels=dim,num_res_blocks=cfg.blocks).to(cfg.device)

    # load model
    state_dict = torch.load(model_path,map_location=cfg.device)
    model.load_state_dict(state_dict)

    feat_extractor = get_extractor(cfg)
   
    print("🤖 Evaluating on Test Dataset...")
    # for steps in [0,1,2,3,4,5,6,7,8,9,10]:
    #     test_fma(model,test_loader,feat_extractor,steps=steps,stepsize=0.1,cfg=cfg)
    steps = 10
    test_fma_adaptive(model,test_loader,feat_extractor,steps=steps,stepsize=1.0/steps,cfg=cfg)
   

