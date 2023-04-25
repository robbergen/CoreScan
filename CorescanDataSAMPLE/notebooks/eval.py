from scipy.spatial.distance import dice
from dataset import create_patch_dataloaders, create_mask
import numpy as np
import torch
import glymur
import cv2



def infer(modelpath, img, mineral, class_dict, mask=None):
    '''Classifies an image patch-wise, then combines the patches into an image mask.'''
    
    model = build_unet()
    model.load_state_dict(torch.load(modelpath))
    model = model.to(device)
    
    if mask==None:
        return_mask = False
        mask = img[0,:,:] #dummy mask for function
    else:
        return_mask = True
        
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

        with torch.no_grad():
            outputs = model(patch_img)
            if k==0:
                out = outputs
            else:
                out = torch.vstack([out,outputs])
            if return_mask:
                if k==0:
                    outmask = patch_mask
                else:
                    outmask = torch.vstack([outmask,patch_mask])
        k = k+1
    
    #Resize outputs into original shapes###
    out = out.reshape([32,2,64,64])
    outmask = outmask.reshape([32,2,64,64])
    emp = torch.empty([2048,128])

    for i in range(32):
        for j in range(2):
            emp[i*64:i*64+64,j*64:j*64+64] = out[i,j,:,:].transpose(1,0)

    out = emp

    emp = torch.empty([2048,128])

    for i in range(32):
        for j in range(2):
            emp[i*64:i*64+64,j*64:j*64+64] = outmask[i,j,:,:].transpose(1,0)

    outmask = emp
    ########################################

    outmask = outmask.cpu().detach().numpy().astype('float')
    out = out.cpu().detach().numpy().astype('float')
    
    return out, outmask

def dsc(im1,im2):
    return 1-dice(im1.flatten(),im2.flatten())

def IOU(im1,im2):
    return np.sum(im1*im2)/np.sum((im1+im2)>0)

def eval(loader, mineral_type,saved_model, class_min_dict):
    av_dsc = 0
    av_IOU = 0
    l_iter = iter(loader)

    for i in range(len(loader)):
        img, mask = next(l_iter)
        img = img.to(device)
        mask = mask.to(device)
        outputs, outmasks = infer(modelpath,img,mineral_type,class_min_dict,mask)
        av_dsc += dsc(outmasks,outputs)
        av_dsc += IOU(outmasks,outputs)
        
    print('Average dice score: {.3f}'.format(av_dsc/len(train_loader)))
    print('Average IoU score: {.3f}'.format(av_IOU/len(train_loader)))
    return


def register_imgs(img_path, label_path,mineral_type):
    '''Registers an image with its mask and returns both'''
    im1 = cv2.imread(img_path)
    im2 = cv2.imread(label_path)
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)

    j2k = glymur.Jp2k(img_path)
    j2k2 = glymur.Jp2k(label_path)

    rgb_tiepoint = j2k.box[3].data['ModelTiePoint'][3:6]
    cm_tiepoint = j2k2.box[3].data['ModelTiePoint'][3:6]
    rgb_res = j2k.box[3].data['ModelPixelScale'][0]
    cm_res = rgb_res*10

    rgb_box = (rgb_tiepoint,rgb_tiepoint+[im1.shape[1]*rgb_res,-im1.shape[0]*rgb_res,0])
    cm_box = (cm_tiepoint,cm_tiepoint+[im2.shape[1]*cm_res,-im2.shape[0]*cm_res,0])


    im1 = cv2.copyMakeBorder(im1, -min(int((rgb_box[0][1]-cm_box[0][1])/cm_res),0),
                              max(int((rgb_box[1][1]-cm_box[1][1])/cm_res),0),
                              max(int((rgb_box[0][0]-cm_box[0][0])/cm_res),0),
                              -min(int((rgb_box[1][0]-cm_box[1][0])/cm_res),0),
                             cv2.BORDER_CONSTANT)
    im2 = cv2.copyMakeBorder(im2, max(int((rgb_box[0][1]-cm_box[0][1])/cm_res),0),
                              -min(int((rgb_box[1][1]-cm_box[1][1])/cm_res),0),
                              -min(int((rgb_box[0][0]-cm_box[0][0])/cm_res),0),
                              max(int((rgb_box[1][0]-cm_box[1][0])/cm_res),0),
                             cv2.BORDER_CONSTANT)

    im1 = cv2.resize(im1, (128,2048))
    im2 = cv2.resize(im2, (128,2048), interpolation=cv2.INTER_NEAREST)


    mask = create_mask(np.transpose(im2,[2,0,1]),mineral_type,class_min_dict)
                     
    return im1, mask
                     
def plot_img(img):
                
    plt.figure(figsize=(10,10))
    plt.imshow(img)
    return

def plot_img_mask(img,mask):
    '''Overlay 1-channel mask on 3-channel image'''
    mask = np.expand_dims(mask,axis=-1)*255
    mask = np.concatenate([mask,mask,mask],axis=-1).astype('int')
    plt.figure(figsize=(10,10))
    dst = cv2.addWeighted(img,0.5,mask,0.7,0)
    plt.imshow(dst[:,:,:])
    return