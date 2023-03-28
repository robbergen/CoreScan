from dataset import create_dataset
from dataset import create_patch_dataset
from train import train

path = '../img-rgb-50u'
label_path = '../img-clm-phy'

train_loader, val_loader = create_patch_dataset(path, label_path, 'chlorite',4) #Returns data loaders
model = train(train_loader, val_loader) #Train model
