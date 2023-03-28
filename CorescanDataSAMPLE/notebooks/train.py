import torch
import torch.nn as nn
from net import build_unet
import torch.optim as optim

epochs = 1

def train(train_dataloader, val_dataloader):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_unet()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for i in range(epochs):
        running_loss = 0
        train_iter = iter(train_dataloader)
        j = 0
        while j < len(train_dataloader):
            j+= 1
            img,mask = next(train_iter)
            img = img.to(device)
            mask = mask.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(img)
            loss = criterion(torch.squeeze(outputs), torch.squeeze(mask))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        # Calculate validation loss
        val_iter = iter(val_dataloader)
        val_loss = 0
        j = 0
        model.eval()
        while j < len(val_dataloader):
            j+= 1
            img,mask = next(val_iter)
            img = img.to(device)
            mask = mask.to(device)

            with torch.no_grad():
                # forward
                outputs = model(img)
                loss = criterion(torch.squeeze(outputs), torch.squeeze(mask))
                val_loss += loss.item()    

        # print statistics
        print(f'[{i + 1}] loss: {running_loss / len(train_dataloader):.3f}')
        print(f'[{i + 1}] val_loss: {val_loss / len(val_dataloader):.3f}')
        model.train()

    return model