
"""
Binary Mask Denoising using UNet
Author: Mohamed Khalil Kiri
Date: 2025-10-26

Full pipeline to train a UNet to denoise binary masks.
Includes synthetic data generation, composite loss, training, evaluation, 
and visualization of predictions and metrics.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import disk, rectangle, polygon
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

# ------------------------
# Synthetic Dataset
# ------------------------
class SyntheticMaskDataset(Dataset):
    """Generates synthetic noisy-clean binary mask pairs on-the-fly."""
    def __init__(self, num_samples=1000, img_size=64, noise_range=(0.4,0.8), augment=True):
        self.num_samples = num_samples
        self.img_size = img_size
        self.noise_range = noise_range
        self.augment = augment

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        mask = self._generate_mask()
        if self.augment:
            mask = self._augment(mask)
        noisy = self._add_noise(mask)
        return (torch.tensor(noisy[None,:,:].copy(), dtype=torch.float32),
                torch.tensor(mask[None,:,:].copy(), dtype=torch.float32))

    def _generate_mask(self):
        mask = np.zeros((self.img_size,self.img_size), dtype=np.float32)
        for _ in range(np.random.randint(2,5)):
            shape_type = np.random.choice(['circle','rectangle','polygon'])
            if shape_type=='circle':
                x,y = np.random.randint(8,self.img_size-8,2)
                r = np.random.randint(3,10)
                rr,cc = disk((x,y), radius=r, shape=mask.shape)
                mask[rr,cc]=1.0
            elif shape_type=='rectangle':
                x,y = np.random.randint(5,self.img_size-15,2)
                w,h = np.random.randint(4,12,2)
                rr,cc = rectangle((x,y), extent=(w,h), shape=mask.shape)
                mask[rr,cc]=1.0
            else: # polygon
                x,y = np.random.randint(10,self.img_size-10,2)
                size = np.random.randint(4,8)
                rr = [x, x-size, x+size]
                cc = [y+size, y-size, y-size]
                rr,cc = polygon(rr,cc,shape=mask.shape)
                mask[rr,cc]=1.0
        return mask

    def _add_noise(self, mask):
        noisy = mask + np.random.uniform(*self.noise_range)*np.random.randn(self.img_size,self.img_size)
        if np.random.rand()>0.7:
            sp = np.random.rand(self.img_size,self.img_size)
            noisy[sp<0.05]=1.0
            noisy[sp>0.95]=0.0
        noisy = np.clip(noisy,0,1)
        if np.random.rand()>0.5:
            noisy = gaussian_filter(noisy,sigma=0.8)
        return noisy

    def _augment(self, mask):
        if np.random.rand()>0.5:
            mask = np.fliplr(mask).copy()
        if np.random.rand()>0.5:
            mask = np.flipud(mask).copy()
        if np.random.rand()>0.5:
            mask = np.rot90(mask, k=np.random.randint(1,4)).copy()
        return mask

# ------------------------
# UNet Model
# ------------------------
class ConvBlock(nn.Module):
    """Double convolution block with batch norm and ReLU, optional dropout."""
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if dropout>0:
            layers.insert(2, nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)
    def forward(self,x):
        return self.block(x)

class DenoisingUNet(nn.Module):
    """UNet for denoising binary masks with skip connections."""
    def __init__(self, in_channels=1, out_channels=1, dropout=0.2):
        super().__init__()
        self.enc1 = ConvBlock(in_channels,16)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(16,32,dropout)
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(32,64,dropout)

        self.up2 = nn.ConvTranspose2d(64,32,2,stride=2)
        self.dec2 = ConvBlock(64,32)
        self.up1 = nn.ConvTranspose2d(32,16,2,stride=2)
        self.dec1 = ConvBlock(32,16)

        self.output_conv = nn.Conv2d(16,out_channels,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))

        d2 = self.up2(b)
        d2 = torch.cat([d2,e2],dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1,e1],dim=1)
        d1 = self.dec1(d1)

        return self.sigmoid(self.output_conv(d1))

# ------------------------
# Loss Functions
# ------------------------
class DiceLoss(nn.Module):
    """Dice loss for segmentation stability."""
    def __init__(self,smooth=1e-6):
        super().__init__()
        self.smooth = smooth
    def forward(self,pred,target):
        inter = (pred*target).sum()
        dice = (2*inter+self.smooth)/(pred.sum()+target.sum()+self.smooth)
        return 1-dice

class CompositeLoss(nn.Module):
    """Weighted BCE + Dice loss for robust segmentation."""
    def __init__(self, alpha=0.7):
        super().__init__()
        self.alpha = alpha
        self.bce = nn.BCELoss()
        self.dice = DiceLoss()
    def forward(self,pred,target):
        return self.alpha*self.bce(pred,target) + (1-self.alpha)*self.dice(pred,target)

# ------------------------
# Metrics
# ------------------------
def calculate_iou(pred,target,threshold=0.5):
    pred_bin = (pred>threshold).float()
    target_bin = (target>threshold).float()
    inter = (pred_bin*target_bin).sum()
    union = pred_bin.sum()+target_bin.sum()-inter
    return 1.0 if union==0 else (inter/union).item()

def calculate_dice(pred,target,threshold=0.5,smooth=1e-6):
    pred_bin = (pred>threshold).float()
    target_bin = (target>threshold).float()
    inter = (pred_bin*target_bin).sum()
    if pred_bin.sum()==0 and target_bin.sum()==0:
        return 1.0
    return ((2*inter+smooth)/(pred_bin.sum()+target_bin.sum()+smooth)).item()

# ------------------------
# Training and Evaluation
# ------------------------
def train_model(model,train_loader,val_loader,criterion,optimizer,scheduler,device,epochs=100):
    model.to(device)
    history = {'train_losses':[],'val_losses':[],'val_ious':[],'val_dices':[]}
    best_loss = float('inf')
    best_weights = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x,y in train_loader:
            x,y = x.to(device),y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out,y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*x.size(0)
        train_loss/=len(train_loader.dataset)
        history['train_losses'].append(train_loss)

        val_metrics = evaluate_model(model,val_loader,criterion,device)
        history['val_losses'].append(val_metrics['loss'])
        history['val_ious'].append(val_metrics['iou'])
        history['val_dices'].append(val_metrics['dice'])

        scheduler.step(val_metrics['loss'])

        if val_metrics['loss']<best_loss:
            best_loss = val_metrics['loss']
            best_weights = model.state_dict().copy()

        if epoch<5 or (epoch+1)%10==0:
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{epochs} | LR={lr:.2e} | Train Loss={train_loss:.4f} | Val Loss={val_metrics['loss']:.4f} | IoU={val_metrics['iou']:.4f} | Dice={val_metrics['dice']:.4f}")

    if best_weights is not None:
        model.load_state_dict(best_weights)
    return model,history

def evaluate_model(model,val_loader,criterion,device):
    model.eval()
    total_loss,total_iou,total_dice,total_samples=0,0,0,0
    with torch.no_grad():
        for x,y in val_loader:
            x,y = x.to(device),y.to(device)
            pred = model(x)
            total_loss += criterion(pred,y).item()*x.size(0)
            total_iou += calculate_iou(pred,y)*x.size(0)
            total_dice += calculate_dice(pred,y)*x.size(0)
            total_samples += x.size(0)
    return {'loss':total_loss/total_samples,'iou':total_iou/total_samples,'dice':total_dice/total_samples}

# ------------------------
# Visualization
# ------------------------
def plot_training_progress(history):
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    axes[0,0].plot(history['train_losses'],label='Train')
    axes[0,0].plot(history['val_losses'],label='Val')
    axes[0,0].set_title('Loss'); axes[0,0].legend(); axes[0,0].grid(True)

    axes[0,1].plot(history['val_ious'],label='IoU',color='green')
    axes[0,1].set_title('Validation IoU'); axes[0,1].legend(); axes[0,1].grid(True)

    axes[1,0].plot(history['val_dices'],label='Dice',color='red')
    axes[1,0].set_title('Validation Dice'); axes[1,0].legend(); axes[1,0].grid(True)

    axes[1,1].plot(history['val_ious'],label='IoU',color='green')
    axes[1,1].plot(history['val_dices'],label='Dice',color='red')
    axes[1,1].set_title('IoU vs Dice'); axes[1,1].legend(); axes[1,1].grid(True)

    plt.tight_layout()
    plt.show()

def visualize_predictions(model,dataset,device,num_samples=5):
    """Display noisy input, ground truth, prediction, and error maps."""
    model.eval()
    results=[]
    with torch.no_grad():
        for i in range(min(num_samples,len(dataset))):
            x,y = dataset[i]
            pred = model(x.unsqueeze(0).to(device)).cpu().squeeze()
            iou = calculate_iou(pred,y)
            dice = calculate_dice(pred,y)
            bce = F.binary_cross_entropy(pred.unsqueeze(0), y).item()
            results.append({'input':x.squeeze().numpy(),'target':y.squeeze().numpy(),
                            'prediction':pred.numpy(),'iou':iou,'dice':dice,'bce':bce})
            print(f"Sample {i+1}: IoU={iou:.3f}, Dice={dice:.3f}, BCE={bce:.3f}")

    fig,axes=plt.subplots(num_samples,4,figsize=(15,4*num_samples))
    if num_samples==1: axes=axes.reshape(1,-1)
    for i in range(num_samples):
        axes[i,0].imshow(results[i]['input'],cmap='gray'); axes[i,0].set_title('Noisy Input')
        axes[i,1].imshow(results[i]['target'],cmap='gray'); axes[i,1].set_title('Ground Truth')
        axes[i,2].imshow(results[i]['prediction'],cmap='gray'); axes[i,2].set_title('Prediction')
        error = np.abs(results[i]['prediction']-results[i]['target'])
        axes[i,3].imshow(error,cmap='hot'); axes[i,3].set_title('Error Map')
    plt.tight_layout()
    plt.show()

# ------------------------
# Main Pipeline
# ------------------------
def main():
    # Hyperparameters
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 32
    img_size = 64
    epochs = 50
    learning_rate = 1e-3

    # Dataset
    dataset = SyntheticMaskDataset(num_samples=1000, img_size=img_size, augment=True)
    train_size = int(0.8*len(dataset))
    val_size = len(dataset)-train_size
    train_set, val_set = random_split(dataset,[train_size,val_size])
    train_loader = DataLoader(train_set,batch_size=batch_size,shuffle=True)
    val_loader = DataLoader(val_set,batch_size=batch_size,shuffle=False)

    # Model, Loss, Optimizer
    model = DenoisingUNet()
    criterion = CompositeLoss(alpha=0.7)
    optimizer = optim.Adam(model.parameters(),lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.5,patience=5)

    # Training
    model,history = train_model(model,train_loader,val_loader,criterion,optimizer,scheduler,device,epochs=epochs)

    # Visualize training metrics
    plot_training_progress(history)

    # Visualize predictions
    visualize_predictions(model,val_set,device,num_samples=5)

if __name__=="__main__":
    main()
