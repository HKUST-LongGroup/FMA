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
            similarities = (transfer_features @ class_embeddings.T).softmax(dim=-1) #[N,C]
            predicted_labels = similarities.argmax(dim=-1).cpu()
            correct += (predicted_labels == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    print(f'Steps:{steps}, StepSize: {stepsize}, Acc: {accuracy:.4f}, Correct:{correct}, Total:{total}')
    return accuracy




if __name__ == "__main__":
    exp = sys.argv[1] # each timestamp corresponding to an exp.
    print(f'Loading Checkpoint at Experiment with timestamp: {exp}')
    config_path = f"./checkpoints/{exp}/config.json"
    model_path = f"./checkpoints/{exp}/model.pth"
    cfg = DefaultConfig.from_json(config_path)
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
    for steps in [0,1,2,3,4,5,6,7,8,9,10]:
        test_fma(model,test_loader,feat_extractor,steps=steps,stepsize=0.1,cfg=cfg)
   

