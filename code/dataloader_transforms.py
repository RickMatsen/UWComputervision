import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
import torch
import math
from pad_pic import pad_to_square
from utils.preprocessing import resize_point

IMAGE_SIZE = 256 ### change parameter passing if it's not 256 in some cases

### augmentation functions and pipelines
### Takes a tuple (img, label) and outputs transformed (img, label)

def random_flip(img_labels):
    img, labels = img_labels
    # Randomly decide whether to horizontally flip the image
    if torch.rand(1) < 0.5:
        # Flip the image horizontally
        flipped_img = transforms.functional.hflip(img)
        
        # Adjust the labels
        adjusted_labels = labels.clone()
        adjusted_labels[0] = IMAGE_SIZE - adjusted_labels[0] - 1 # circle x
        for i in range(3, 13, 2):
            adjusted_labels[i] = IMAGE_SIZE -  adjusted_labels[i] - 1        
        return flipped_img, adjusted_labels
    else:
        # If not flipping, return the original image and labels
        return img, labels
    

def random_crop(img_labels):
    if torch.rand(1) >= 0.5:
        return img_labels

    img, labels = img_labels
    # Assume img is a torch Tensor of shape [C, H, W]

    # calculate image boundaries
    adjusted_labels = labels.clone()

    xlim_min, ylim_min = float("inf"), float("inf")
    xlim_max, ylim_max = -float("inf"), -float("inf")
    
    r = labels[2]  # radius
    
    xlim_min = min(labels[0]-r, xlim_min)
    xlim_max = max(labels[0]+r, xlim_max)

    ylim_min = min(labels[1]-r, ylim_min)
    ylim_max = max(labels[1]+r, ylim_max)
    
    for i in range(3, 13, 2):
        xlim_min = min(adjusted_labels[i], xlim_min)
        xlim_max = max(adjusted_labels[i], xlim_max)   

    for i in range(4, 13, 2):
        ylim_min = min(adjusted_labels[i], ylim_min)
        ylim_max = max(adjusted_labels[i], ylim_max)

    # sample a coordinate to the top left of the bounding rectangle

    pad_rec = 2

    xlim_min,xlim_max, ylim_min,ylim_max = map(lambda x: x.item(), [xlim_min,xlim_max, ylim_min,ylim_max])
    xlim_min = math.floor(xlim_min)
    ylim_min = math.floor(ylim_min)
    xlim_max = math.ceil(xlim_max)
    ylim_max = math.ceil(ylim_max)

    # print(xlim_min,xlim_max, ylim_min,ylim_max)

    x_tl = torch.randint(low=0, high = xlim_min+1, size=(1,)).item()
    y_tl = torch.randint(low=0, high = ylim_min+1, size=(1,)).item()
    x_tr = torch.randint(low = xlim_max, high = img.shape[-1], size=(1,)).item()
    y_tr = y_tl

    x_bl = x_tl
    y_bl = torch.randint(low = ylim_max, high = img.shape[-2], size=(1,)).item()

    x_br = x_tr
    y_br = y_bl

    adjusted_labels[0] = adjusted_labels[0] - x_tl
    adjusted_labels[1] =  adjusted_labels[1] - y_tl

    for i in range(3, 13, 2):
        adjusted_labels[i] = adjusted_labels[i]  - x_tl

    for i in range(4, 13, 2):
        adjusted_labels[i] = adjusted_labels[i]  - y_tl

    cropped_img = img[:, y_tl:y_bl+1, x_tl:x_tr+1] 
    _, h, w = cropped_img.shape

    diff = abs(h - w)
    padding = diff // 2
    if h > w:
        adjusted_labels[0] = adjusted_labels[0] + padding
        for i in range(3, 13, 2):
            adjusted_labels[i] = adjusted_labels[i]  + padding
    else:
        adjusted_labels[1] =  adjusted_labels[1] + padding
        for i in range(4, 13, 2):
            adjusted_labels[i] = adjusted_labels[i]  + padding
    
       


    cropped_img = pad_to_square( cropped_img  )
    
    rz = transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))

    point = (adjusted_labels[0], adjusted_labels[1])
    
    adjusted_labels[0], adjusted_labels[1] = resize_point(cropped_img, point, (IMAGE_SIZE, IMAGE_SIZE))
    
    for i in range(3, 13, 2):
        point = (adjusted_labels[i], adjusted_labels[i+1] )
        adjusted_labels[i], adjusted_labels[i+1] = resize_point(cropped_img, point, (IMAGE_SIZE, IMAGE_SIZE))

    adjusted_labels[2] = adjusted_labels[2] * IMAGE_SIZE / cropped_img.shape[-2]

    resized_img = rz(cropped_img)


    return resized_img, adjusted_labels



def random_noise(img_labels):
    img, labels = img_labels
    if torch.rand(1)<0.5:
        blurred_img = transforms.GaussianBlur(kernel_size=(5, 9))(img)
        return blurred_img, labels
    else:
        return img, labels


def random_vflip(img_labels):
    img, labels = img_labels
    # Randomly decide whether to horizontally flip the image
    if torch.rand(1) < 0.5:
        # Flip the image horizontally
        flipped_img = transforms.functional.vflip(img)
        
        # Adjust the labels
        adjusted_labels = labels.clone()
        adjusted_labels[1] = IMAGE_SIZE - adjusted_labels[1] -1 # circle y
        for i in range(4, 13, 2):
            adjusted_labels[i] = IMAGE_SIZE -  adjusted_labels[i]-1        
        return flipped_img, adjusted_labels
    else:
        # If not flipping, return the original image and labels
        return img, labels


### add new augmentation functions to the list 
transform_augument = transforms.Compose([random_flip, random_vflip, random_noise, random_crop])

transform_pipeline =  transforms.Compose([
     lambda x: x/255,
                               transforms.Normalize(
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
                                    ])

class XrayDataset(Dataset):
    def __init__(self, x, y ,transform=transform_pipeline, augmentation=transform_augument):
        self.labels = y
        self.transform = transform
        self.augmentation = augmentation
        self.x = x

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        image = self.x[idx]
        labels = self.labels[idx, :]
        if self.transform:
            if self.augmentation:
                image, labels = self.augmentation( (image, labels) )
            image = self.transform(image)
        return [image, labels]
    


