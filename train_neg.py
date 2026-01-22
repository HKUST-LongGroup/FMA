import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import os, sys
import clip
from models.fm import DeepFlowMatchingNet
from models.feature_extractor import get_extractor
from datasets import build_dataset
from config import DefaultConfig,SimpleLogger, set_seed
from test import test_fma

def main():
    #set up config, seed and log
    cfg = DefaultConfig()
    cfg.parse_args()  # parse command line args to override the defaults
    cfg.save()
    set_seed(cfg.seed)
    sys.stdout = SimpleLogger(os.path.join(cfg.save_dir,"log.txt"))
    print(cfg)

     # Prepare dataset
    dataset = build_dataset(cfg)
    cfg.classnames = dataset.classnames

    
    train_loader = DataLoader(dataset.train_x, batch_size=cfg.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(dataset.test, batch_size=cfg.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    val_loader = DataLoader(dataset.val, batch_size=cfg.batch_size, shuffle=False, num_workers=8, pin_memory=True)



    # Initialize the model
    clip_model,_ = clip.load(cfg.clip_type, device=cfg.device, jit=False)
    dim = clip_model.visual.output_dim
    print(f"CLIP model output dimension: {dim}")
    model = DeepFlowMatchingNet(in_channels=dim,model_channels=dim, out_channels=dim,num_res_blocks=cfg.blocks).to(cfg.device)
    feat_extractor = get_extractor(cfg)
    # optimizer setting
    num_training_steps,warmup_steps = cfg.epochs*len(train_loader), cfg.warmup_epochs*len(train_loader)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_training_steps-warmup_steps, eta_min=0)

    scalar = GradScaler()
    cur_step = 0

     # Initial test before training

    print(f'[Testing] on Test Dataset:',end=' ')
    test_fma(model,test_loader,feat_extractor,steps=0,stepsize=0.1,cfg=cfg)

    model.train()
    for epoch in range(cfg.epochs):
    # every 20 epochs, evaluate the model on the test set, then save the best model

        for images,labels in train_loader:
            
            # Using different feature extraction to extract paired image and text features
            with autocast():

                image_features, text_features, class_embeddings = feat_extractor(images,labels)
                t = torch.rand(image_features.size(0),1,device=cfg.device)
                # FMC Framework
                # (1) interpolate the features
                noise = torch.randn_like(image_features)
                def get_neg_labels(num_classes,neg_samples, pos_labels):
                    batch_size = len(pos_labels)
                    pos_labels = pos_labels.to(cfg.device) # [B,1]

                    noise = torch.rand(batch_size,num_classes-1,device=cfg.device) # [B, C-1]
                    random_indices = torch.argsort(noise,dim=1)[:,:neg_samples] #[ B, neg_samples]
                    offset = random_indices + 1 # [B, neg_samples]
                    neg_labels = (pos_labels.unsqueeze(1) + offset) % num_classes

                    return neg_labels
                num_classes = class_embeddings.shape[0] if cfg.feature_extractor != 'cocoop' else class_embeddings.shape[1]
                # neg_labels = get_neg_labels(num_classes=num_classes,neg_samples=cfg.neg_samples, pos_labels=labels) #[B, neg_samples]
                # if cfg.feature_extractor == 'cocoop': #class_embeddings : [B, n_cls, dim]
                    # neg_text_features = class_embeddings[torch.arange(class_embeddings.size(0)).unsqueeze(1), neg_labels] # class _embeddings: [B, dim]
                # else:
                    # neg_text_features = class_embeddings[neg_labels] # class _embeddings: [B, dim]
                # neg_text_features = torch.mean(neg_text_features,dim=1) # [B, dim]
                interpolated_features = (1 - t) * image_features + t * text_features  - cfg.neg_gamma*t*(1-t)*neg_text_features + cfg.gamma*torch.sqrt(t*(1-t))*noise
                
                # (2) predict the velocity
                velocity_pred = model(interpolated_features,t)

                # (3) feature transfer to the target x_1
                # transfer_features = interpolated_features + velocity* (1-t)
                velocity_gt = text_features - image_features - cfg.neg_gamma*(1-2*t)*neg_text_features + cfg.gamma*0.5*(1-2*t)/torch.sqrt(t*(1-t))*noise

                loss = torch.sum((velocity_pred-velocity_gt)**2,dim=1).mean() 

            optimizer.zero_grad()
            scalar.scale(loss).backward()
            scalar.step(optimizer)
            scalar.update()
            cur_step +=1
            if cur_step < warmup_steps:
                # linear warmup
                lr_scale = float(cur_step) / float(warmup_steps)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = cfg.lr * lr_scale
            else:

                scheduler.step()

        print(f"[Training] Epoch [{epoch+1}/{cfg.epochs}], Loss: {loss.item():.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}") 
        
        if (epoch+1) % 50 == 0:
            print(f'[Testing] On Test Dataset:',end=' ')
            test_fma(model,test_loader,feat_extractor,steps=1,stepsize=0.1, cfg=cfg)

            print(f'[Testing] On Training Dataset:',end=' ')
            test_fma(model,train_loader,feat_extractor,steps=1,stepsize=0.1,cfg=cfg)

    best_acc = 0.0
    best_steps = 0
     # Final test on test set with multiple steps
    print(f'Final Testing On Test Dataset')
    for steps in [0,1,2,3,4,5,6,7,8,9,10]:
        test_acc = test_fma(model,test_loader,feat_extractor,steps=steps,stepsize=0.1, cfg=cfg)
        if test_acc > best_acc:
            best_acc = test_acc
            best_steps = steps

    torch.save(model.state_dict(), os.path.join(cfg.save_dir,'model.pth'))
    print(f"Dataset:{cfg.dataset}; Feature Extractor:{cfg.feature_extractor}; Best Test accuracy: {best_acc:.4f}, at Steps:{best_steps}; Velocity saved at {cfg.save_dir}/model.pth")

    return 


if __name__ == '__main__' :

    main()