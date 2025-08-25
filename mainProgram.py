import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.optim import Adam
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from tqdm import tqdm
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassF1Score, MulticlassRecall, MulticlassAccuracy, MulticlassPrecision
import copy
import segmentation_models_pytorch as smp
from torchmetrics import ConfusionMatrix

# ------------------- Preparing the dataset for training -------------------
class SemanticSegmentationDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_paths = sorted([os.path.join(image_dir, img) for img in os.listdir(image_dir)])
        self.label_paths = sorted([os.path.join(label_dir, lbl) for lbl in os.listdir(label_dir)])
        self.class_colors = {
            (80, 80, 80): 0,        # LTE class
            (160, 160, 160): 1,     # 5G NR class
            (255, 255, 255): 2,     # Radar class
            (0, 0, 0): 3            # Noise class
        }
    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = cv2.imread(self.label_paths[idx])
        label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)

        label_mask = np.zeros(label.shape[:2], dtype=np.uint8)
        for rgb, idx in self.class_colors.items():
            label_mask[np.all(label == rgb, axis=-1)] = idx

        if self.transform:
            image = self.transform(image)
            label_mask = torch.from_numpy(label_mask).long()

        return image, label_mask

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor()
])


# Path to the dataset
train_dataset = SemanticSegmentationDataset(
    image_dir='C:\\Users\\Tai\\Desktop\\Tai\\datasetC03\\RadarComm_Spectrogram\\dataset\\train\\input',
    label_dir='C:\\Users\\Tai\\Desktop\\Tai\\datasetC03\\RadarComm_Spectrogram\\dataset\\train\\label',
    transform=train_transform
)
val_dataset = SemanticSegmentationDataset(
    image_dir='C:\\Users\\Tai\\Desktop\\Tai\\datasetC03\\RadarComm_Spectrogram\\dataset\\test\\input',
    label_dir='C:\\Users\\Tai\\Desktop\\Tai\\datasetC03\\RadarComm_Spectrogram\\dataset\\test\\label',
    transform=train_transform
)

# ------------------- Training model -------------------
def train_epoch(model, dataloader, criterion, optimizer, device, num_classes):
    model.train()
    running_loss = 0.0  
    accuracy_metric = MulticlassAccuracy(num_classes=num_classes).to(device)
    iou_metric = MulticlassJaccardIndex(num_classes=num_classes).to(device)
    precision_metric = MulticlassPrecision(num_classes=num_classes).to(device)
    f1_metric = MulticlassF1Score(num_classes=num_classes).to(device)
    confmat = ConfusionMatrix(task="multiclass", num_classes=num_classes).to(device)
    pbar = tqdm(dataloader, desc='Training', unit='batch')
    # -------------------- Model Quantization  --------------------
    scaler = torch.cuda.amp.GradScaler()
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():  
            outputs = model(images)
            loss = criterion(outputs, labels)
        # -------------------- EGR --------------------
        preds = torch.argmax(outputs, dim=1) 
        E = (preds == labels).float() 
        ones = torch.ones_like(E)
        loss_E = torch.nn.functional.mse_loss(E, ones)
        loss = (loss + loss_E)/2
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    # --------------------------------------------------------------
        running_loss += loss.item() * images.size(0)      
        preds = torch.argmax(outputs, dim=1)     
        accuracy_metric(preds, labels)
        iou_metric(preds, labels)
        precision_metric(preds, labels)
        f1_metric(preds, labels)
        confmat(preds, labels)  
        pbar.set_postfix({
            'Batch Loss': f'{loss.item():.4f}',
            'Mean Accuracy': f'{accuracy_metric.compute():.4f}',
            'Mean IoU': f'{iou_metric.compute():.4f}',
            'Mean Precision': f'{precision_metric.compute():.4f}',
            'Mean F1 Score': f'{f1_metric.compute():.4f}'
        }) 
    epoch_loss = running_loss / len(dataloader.dataset)  
    mean_accuracy = accuracy_metric.compute().cpu().numpy()
    mean_iou = iou_metric.compute().cpu().numpy()
    mean_precision = precision_metric.compute().cpu().numpy()
    mean_f1 = f1_metric.compute().cpu().numpy()

    cm = confmat.compute().cpu().numpy() 
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] 
    confmat.reset()
   
    return cm_normalized, epoch_loss, mean_accuracy, mean_iou, mean_precision, mean_f1

def evaluate(model, dataloader, criterion, device, num_classes):
    model.eval()
    model = model.half()
    running_loss = 0.0    
    accuracy_metric = MulticlassAccuracy(num_classes=num_classes).to(device)
    iou_metric = MulticlassJaccardIndex(num_classes=num_classes).to(device)
    precision_metric = MulticlassPrecision(num_classes=num_classes).to(device)
    f1_metric = MulticlassF1Score(num_classes=num_classes).to(device)
    confmat = ConfusionMatrix(task="multiclass", num_classes=num_classes).to(device)
    pbar = tqdm(dataloader, desc='Evaluating', unit='batch')
    with torch.no_grad():
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images.half()).float()
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            accuracy_metric(preds, labels)
            iou_metric(preds, labels)
            precision_metric(preds, labels)
            f1_metric(preds, labels)
            confmat(preds, labels)  
            pbar.set_postfix({
                'Batch Loss': f'{loss.item():.4f}',
                'Mean Accuracy': f'{accuracy_metric.compute():.4f}',
                'Mean IoU': f'{iou_metric.compute():.4f}',
                'Mean Precision': f'{precision_metric.compute():.4f}',
                'Mean F1 Score': f'{f1_metric.compute():.4f}'
            })
    
    epoch_loss = running_loss / len(dataloader.dataset)
    mean_accuracy = accuracy_metric.compute().cpu().numpy()
    mean_iou = iou_metric.compute().cpu().numpy()
    mean_precision = precision_metric.compute().cpu().numpy()
    mean_f1 = f1_metric.compute().cpu().numpy()
    cm = confmat.compute().cpu().numpy() 
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] 
    confmat.reset()
    model.float()
    return cm_normalized, epoch_loss, mean_accuracy, mean_iou, mean_precision, mean_f1

class myModel(nn.Module):
    def __init__(self, n_classes):
        super(myModel, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.upSampling = nn.Upsample(scale_factor=2, mode="bilinear") 
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=1)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1)

        self.bn1_2 = nn.BatchNorm2d(64)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1)
        self.bn2_2 = nn.BatchNorm2d(128)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1)
        self.bn3_1 = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1)
        self.bn3_2 = nn.BatchNorm2d(256)
               
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=1)
        self.bn4_1 = nn.BatchNorm2d(512)
        self.conv4_2 = nn.Conv2d(512, 256, kernel_size=3, padding=1, stride=1)
        self.bn4_2 = nn.BatchNorm2d(256)
        
        self.conv5_1 = nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1)
        self.bn5_1 = nn.BatchNorm2d(256)
        self.conv5_2 = nn.Conv2d(256, 128, kernel_size=3, padding=1, stride=1)
        self.bn5_2 = nn.BatchNorm2d(128)

        self.conv6_1 = nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1)
        self.bn6_1 = nn.BatchNorm2d(128)
        self.conv6_2 = nn.Conv2d(128, 64, kernel_size=3, padding=1, stride=1)
        self.bn6_2 = nn.BatchNorm2d(64)

        self.conv7_1 = nn.Conv2d(64, 16, kernel_size=5, padding=2, stride=1)
        self.bn7_1 = nn.BatchNorm2d(16)
        self.conv7_2 = nn.Conv2d(16, n_classes, kernel_size=5, padding=2, stride=1)
    def forward(self, x):
        # Encoder
        x1 = F.relu(self.bn1_1(self.conv1_1(x)))
        x1 = F.relu(self.bn1_2(self.conv1_2(x1)))
        out1, indices1 = F.max_pool2d(x1, kernel_size=2, stride=2, return_indices=True)
        x1 = self.maxpool(x1)

        x2 = F.relu(self.bn2_1(self.conv2_1(x1)))
        x2 = F.relu(self.bn2_2(self.conv2_2(x2)))
        out2, indices2 = F.max_pool2d(x2, kernel_size=2, stride=2, return_indices=True)
        x2 = self.maxpool(x2)

        x3 = F.relu(self.bn3_1(self.conv3_1(x2)))
        x3 = F.relu(self.bn3_2(self.conv3_2(x3)))
        out3, indices3 = F.max_pool2d(x3, kernel_size=2, stride=2, return_indices=True)
        x3 = self.maxpool(x3)

        x4 = F.relu(self.bn4_1(self.conv4_1(x3)))
        x4 = F.relu(self.bn4_2(self.conv4_2(x4)))

        # Decoder
        x5 = self.upSampling(x4)
        x5_ = F.max_unpool2d(out3, indices3, kernel_size=2, stride=2, output_size=x5.size())
        x5 = x5 + x5_
        x5 = F.relu(self.bn5_1(self.conv5_1(x5)))
        x5 = F.relu(self.bn5_2(self.conv5_2(x5)))       

        x6 = self.upSampling(x5)
        x6_ = F.max_unpool2d(out2, indices2, kernel_size=2, stride=2, output_size=x6.size())
        x6 = x6 + x6_
        x6 = F.relu(self.bn6_1(self.conv6_1(x6)))
        x6 = F.relu(self.bn6_2(self.conv6_2(x6))) 

        x7 = self.upSampling(x6) 
        x7_ = F.max_unpool2d(out1, indices1, kernel_size=2, stride=2, output_size=x7.size())
        x7 = x7 + x7_
        x7 = F.relu(self.bn7_1(self.conv7_1(x7)))
        x7 = self.conv7_2(x7) 
        
        return F.softmax(x7, dim=1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
print(device)
classes = 4  
model = myModel(classes)

def count_parameters(model):     
    return sum(p.numel() for p in model.parameters())
total_params = count_parameters(model)
print(f"Total parameters: {total_params}")

model.to(device)
model = nn.DataParallel(model)

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)

criterion = nn.CrossEntropyLoss()  
optimizer = Adam(model.parameters(), lr=0.00001)
num_epochs = 50

epoch_saved = 0
best_val_mAcc = 0.0  
best_model_state = None

for epoch in range(num_epochs):
    cm_normalized_train, epoch_loss_train, mAcc_train, mIoU_train, mPre_train, mF1_train = train_epoch(model, train_dataloader, criterion, optimizer, device, classes)
    cm_normalized_val, epoch_loss_val, mAcc_val, mIoU_val, mPre_val, mF1_val = evaluate(model, val_dataloader, criterion, device, classes)
    
    print(f"Epoch {epoch + 1}/{num_epochs}")
    print(f"Train Loss: {epoch_loss_train:.4f}, Mean Accuracy: {mAcc_train:.4f}, Mean IoU: {mIoU_train:.4f}, Mean Precision: {mPre_train:.4f}, Mean F1: {mF1_train:.4f}")
    print(f"Validation Loss: {epoch_loss_val:.4f}, Mean Accuracy: {mAcc_val:.4f}, Mean IoU: {mIoU_val:.4f}, Mean Precision: {mPre_val:.4f}, Mean F1: {mF1_val:.4f}")

    file = open('training.txt', 'w')
    file.write(f"Train Loss: {epoch_loss_train:.4f}, Mean Accuracy: {mAcc_train:.4f}, Mean IoU: {mIoU_train:.4f}, Mean Precision: {mPre_train:.4f}, Mean F1: {mF1_train:.4f}\n")
    file.write(f"Validation Loss: {epoch_loss_val:.4f}, Mean Accuracy: {mAcc_val:.4f}, Mean IoU: {mIoU_val:.4f}, Mean Precision: {mPre_val:.4f}, Mean F1: {mF1_val:.4f}\n")
    file.close()
    if mAcc_val >= best_val_mAcc:
        epoch_saved = epoch + 1 
        best_val_mAcc = mAcc_val
        best_model_state = copy.deepcopy(model.state_dict())
    
print("===================")
print(f"Best Model at epoch : {epoch_saved}")


model.load_state_dict(best_model_state)
if isinstance(model, torch.nn.DataParallel):
    model = model.module
model_save = torch.jit.script(model)
model_save.save("OSNet.pt")
