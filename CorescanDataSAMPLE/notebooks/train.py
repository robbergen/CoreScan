import torch
import torch.nn as nn
from net import build_unet
import torch.optim as optim
from dataset import create_patch_dataloaders
epochs = 30

def train(train_dataloader, val_dataloader, mineral, class_dict, saved_model=None):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_unet()
    if saved_model:
        model.load_state_dict(torch.load(saved_model))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

    for i in range(epochs):
        train_iter = iter(train_dataloader)
        j = 0
        running_loss = 0
        while j < len(train_dataloader):
            if j % (len(train_dataloader)//5) == 0:
                print('\t Image ',j,'/',len(train_dataloader))
            img,mask = next(train_iter)
            patch_train_loader = create_patch_dataloaders(img, mask, mineral, class_dict, batch_size=4)
            patch_iter = iter(patch_train_loader)
            k=0
            while k < len(patch_train_loader):

                patch_img, patch_mask = next(patch_iter)
                
                if len(patch_img.shape) > 4:
                    patch_img = torch.squeeze(patch_img)

                if len(patch_img.shape) < 4:
                    patch_img = torch.unsqueeze(patch_img,0)
                
                patch_img = patch_img.to(device)
                patch_mask = patch_mask.to(device)
                

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(patch_img)
                loss = criterion(torch.squeeze(outputs), torch.squeeze(patch_mask))
                loss.backward()
                optimizer.step()
                k+= 1
                running_loss += loss.item()

            j+= 1

        model.eval()
        val_iter = iter(val_dataloader)
        val_loss=0
        j=0
        while j < len(val_dataloader):

            img,mask = next(val_iter)
            patch_train_loader = create_patch_dataloaders(img, mask, mineral, class_dict, batch_size=4)
            patch_iter = iter(patch_train_loader)
            k=0
            while k < len(patch_train_loader):

                patch_img, patch_mask = next(patch_iter)
                
                if len(patch_img.shape) > 4:
                    patch_img = torch.squeeze(patch_img)

                if len(patch_img.shape) < 4:
                    patch_img = torch.unsqueeze(patch_img,0)
                
                patch_img = patch_img.to(device)
                patch_mask = patch_mask.to(device)

                with torch.no_grad():
                    outputs = model(patch_img)
                    loss = criterion(torch.squeeze(outputs), torch.squeeze(patch_mask))
                k+= 1
                val_loss += loss.item()

            j+= 1
        model.train()
        # print statistics
        print('Epoch ',i+1,'/',epochs)
        print(f'loss: {running_loss / len(train_dataloader) / len(patch_train_loader):.3f}')
        print(f' val_loss: {val_loss / len(val_dataloader) / len(patch_train_loader):.3f}')
    
    torch.save(model.state_dict(), '../saved_models/patch_sericite.pt')
    return model