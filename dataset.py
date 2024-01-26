import os
import PIL
from PIL import Image
from albumentations.augmentations.geometric.transforms import ShiftScaleRotate
from albumentations.core.composition import SomeOf
import numpy as np
import cv2
import random
from utils import get_image_list, make_test_label

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision import transforms
import albumentations
import albumentations.pytorch

class MVTecAD(Dataset):

    def __init__(self, image_list, label_list, transform):
        self.image_list = image_list
        self.label_list = label_list
        self.transform = transform

    def __getitem__(self, index):
        image = Image.open(self.image_list[index])
        label = self.label_list[index]
        return self.transform(image), label

    def __len__(self):
        return len(self.image_list)


class MVTecAD_Dataset(Dataset):

    def __init__(self, image_list, label_list, transform):
        self.image_list = image_list
        self.label_list = label_list
        self.transform = transform

    def __getitem__(self, index):
        image = cv2.imread(self.image_list[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.label_list[index][0]
        if self.transform is not None:
            augmented = self.transform(image=image, label=label)
            image = augmented['image']
            label = augmented['label']
        return image, (label, self.label_list[index][1], self.image_list[index])

    def __len__(self):
        return len(self.image_list)
    
def get_folder_names(directory):
    """ List all folder names in the specified directory. """
    return [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]

def MVTecAD_loader(image_dir, image_size, train_ratio=0.9, batch_size=1, num_workers=0, is_inference=False, seed=1234,contamination=False,Z_score=False):
    assert train_ratio >=0.0 and train_ratio <=1.0
    # automatically returns two dataloaders. train & valid depends on train_ratio
    random.seed(seed)
    
    transform_train = albumentations.Compose([
        albumentations.Resize(height=image_size[0], width=image_size[1]),
        albumentations.HorizontalFlip(),
        albumentations.core.composition.SomeOf([albumentations.ShiftScaleRotate(border_mode=cv2.BORDER_REPLICATE),
                                            albumentations.RandomRotate90(),
                                            albumentations.GaussNoise()], n=2),
        albumentations.Normalize(mean=[0, 0, 0], std=[1, 1, 1]), 
        albumentations.pytorch.transforms.ToTensorV2(transpose_mask=True)
    ])
    transform_infer = albumentations.Compose([
        albumentations.Resize(height=image_size[0], width=image_size[1]),
        albumentations.Normalize(mean=[0, 0, 0], std=[1, 1, 1]), 
        albumentations.pytorch.transforms.ToTensorV2(transpose_mask=True)
    ])

    train_imgdir = os.path.join(image_dir, os.path.join('train', 'good'))
    test_imgdir = os.path.join(image_dir, 'test')
    test_labdir = os.path.join(image_dir, 'ground_truth')

    train_image_list = get_image_list(train_imgdir)
    
    test_image_list = get_image_list(test_imgdir)
    
    ## add contam
    if contamination:
        #TODO procentual contamination
        anolist = get_folder_names(test_labdir)
        anos_for_trainset=[]     
        for anotype  in anolist :
            if anotype != "good":
                subset=[x for x in test_image_list if anotype in x]
                samples_subset=random.sample(subset, len(subset)//2 )
                anos_for_trainset.extend(samples_subset)
            
        train_image_list=train_image_list+anos_for_trainset   
        test_image_list=[x for x in test_image_list if x not in anos_for_trainset]
        
    test_label_list = [make_test_label(test_imgdir, test_labdir, x, image_size) for x in test_image_list]
    
    
    random.shuffle(train_image_list)
    train_image_list, valid_image_list = train_image_list[:int(len(train_image_list)*train_ratio)], train_image_list[int(len(train_image_list)*train_ratio):]
    train_label_list = [(np.zeros(image_size, dtype=np.uint8), 0)]*len(train_image_list)
    valid_label_list = [(np.zeros(image_size, dtype=np.uint8), 0)]*len(valid_image_list)



    #  TODO add part to insert contamination into the training data set : IDEA reomve from testset and add to trainset


    #print(train_image_list,)



    # training
    if not is_inference:
        
        
        if not Z_score:
            train_dataset = MVTecAD_Dataset(train_image_list, train_label_list, transform_train)
        else:
            train_dataset = MVTecAD_Dataset(train_image_list, train_label_list, transform_infer) # for calculation of the Z score we do not want any augmentation -> only for evaluation
            
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
        if train_ratio < 1.0:
            valid_dataset = MVTecAD_Dataset(valid_image_list, valid_label_list, transform_infer)
            valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=1, num_workers=0, shuffle=False)
        else:
            valid_dataloader = None
        return train_dataloader, valid_dataloader
    
    #inference
    else:
        train_dataset = MVTecAD_Dataset(train_image_list, train_label_list, transform_infer)
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
 
        infer_dataset = MVTecAD_Dataset(test_image_list, test_label_list, transform_infer)
        infer_dataloader = DataLoader(dataset=infer_dataset, batch_size=1, num_workers=0, shuffle=False)
        return infer_dataloader, train_dataloader

def imshow(x_0):
    for i in range(list(x_0.size())[0]):
        img = x_0[i].detach().cpu().numpy()
        img = img*255.0
        img = img.astype(np.uint8)
        img = np.moveaxis(img, 0, 2)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow('img', img)
        cv2.waitKey(0)


if __name__ == '__main__':
    image_dir = '/home/bule/projects/MVTec_Visualizer/data/mvtec_anomaly_detection/cable'
    image_size = (256, 256)
    data_loader, _ = MVTecAD_loader(image_dir, image_size, train_ratio=.9, batch_size=256, num_workers=24, is_inference=False, seed=1234,contamination=False)
    x_0, _ = next(iter(data_loader))
    
    #imshow(x_0)
    
    
    print(os.cpu_count())