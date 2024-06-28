import torch
from torchvision import transforms, models
import torch.nn as nn
from matplotlib import pyplot as plt
import cv2
#import skimage
import numpy as np
from PIL import Image
from pad_pic import pad_to_square_from_path
from utils.xray_plot import draw_pred
from cv2 import cvtColor, COLOR_BGR2RGB
from utils.xray_plot import draw_pred_pil
import argparse
import os
import fnmatch
import pandas as pd
import xml.etree.ElementTree as ET




device = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_SIZE = 256

transform = transforms.Compose([
        transforms.ToTensor() # Converts image to tensor
    ])

# not robust with other resizes and flips
transform_pipeline =  transforms.Compose([pad_to_square_from_path,
                                          transforms.ToPILImage(),
                                          lambda img: cvtColor(np.array(img), COLOR_BGR2RGB),

                                          lambda img: cv2.resize(img, (256,256)),
                                          # lambda img: torch.from_numpy(img).permute(2, 1, 0),

                                          lambda img: torch.from_numpy(img).permute(2, 0, 1),
                              lambda x: x/255,
                               transforms.Normalize(
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
                                    ])
# mean=[0.485, 0.456, 0.406],
# std=[0.229, 0.224, 0.225])
invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),
                                lambda x: x*255
                               ])


def resnet34(outshape, source_domain):
    if source_domain == "ImageNet":
        resnet = models.resnet34(pretrained=True)
    elif not source_domain:
        resnet = models.resnet34(pretrained=False)
    else:
        raise NotImplementedError("Unknown source domain.")
    num_features = resnet.fc.in_features
    resnet.fc = nn.Linear(num_features, outshape)
    return resnet

class XrayPretrained(nn.Module):
    def __init__(self):
        super(XrayPretrained, self).__init__()
        self.resnet = pretrained_model # output size = resnet18_shape
        self.fc1=nn.Linear(resnet34_shape,  output_size)

    def forward(self, x):
        x = self.resnet(x)
        x = nn.functional.relu(x)
        x = self.fc1(x)
        return x


class XrayModel(nn.Module):
    def __init__(self, backbone):
        super(XrayModel, self).__init__()
        self.resnet = backbone # output size = resnet18_shape

    def forward(self, x):
        x = self.resnet(x)
        return x
    

def cal_retroversion(yhat):

    s_x, s_y = yhat[-2], yhat[-1]
    c_x, c_y = yhat[-4], yhat[-3]

    p_x, p_y = yhat[-6], yhat[-5]
    a_x, a_y = yhat[-8], yhat[-7]

    sc_x, sc_y = c_x-s_x, c_y-s_y

    pa_x, pa_y = a_x-p_x, a_y - p_y

    dot_prod = sc_x*pa_x + sc_y*pa_y

    denom = np.sqrt(sc_x**2 + sc_y**2) * np.sqrt(pa_x**2 + pa_y**2)

    angle_cos = dot_prod/denom

    return np.degrees(np.arccos(angle_cos)).item()

def cal_centering(yhat):
    s = (yhat[-2], yhat[-1])
    c = (yhat[-4], yhat[-3])

    x_s, y_s = s

    p = (yhat[-6], yhat[-5])
    a = (yhat[-8], yhat[-7])

    x = (yhat[0], yhat[1])
    r = yhat[2]
    
    x_a, y_a = a
    x_c, y_c = p

    x_b, y_b = (x_a+x_c)/2, (y_a+y_c)/2

    b = (x_b, y_b)

    x_x, y_x = x


    e_x_numer  = ((y_a-y_c)**2)*x_x + ((x_a-x_c)**2)*x_b + y_b*(x_a-x_c)*(y_a-y_c) - y_x*(x_a-x_c)*(y_a-y_c)
    e_x_denom = (y_a-y_c)**2 + (x_a-x_c)**2

    e_x = e_x_numer/e_x_denom

    e_y =   ( (y_a-y_c)/(x_a-x_c)     )  *(e_x-x_x) +y_x

    e = (e_x, e_y)

    len_ex = np.sqrt((e_x-x_x)**2 + (e_y-y_x)**2)

    len_ea =np.sqrt((e_x-x_a)**2 + (e_y-y_a)**2)

    len_es = np.sqrt((e_x-x_s)**2 + (e_y-y_s)**2)

    ratio = ((len_ex)/(2*r)).item()

    if ratio >=0.5:
        ratio = 0
    elif len_es<len_ea:
        ratio = - ratio
    

    return ratio



def find_images(directory, model, patterns=['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.tiff']):
    """
    Print all image file names found in the specified directory and its subdirectories.

    Args:
    - directory (str): The path to the directory to search.
    - patterns (list): A list of patterns to match against file names.
    """
    preds = []
    image_names = []
    measurements = []
    for root, dirs, files in os.walk(directory):
        for pattern in patterns:
            for filename in fnmatch.filter(files, pattern):
                pred = run_inference_xml(model, os.path.join(root, filename))
                preds.append(pred)
                measurements.append((cal_centering(pred), cal_retroversion(pred), pred[2]   ))
                image_names.append(filename)
    return image_names, preds, measurements



def load_model(model_path):
    # model = torch.load(model_path)
    model = XrayModel(resnet34(13, "ImageNet")).to(device)

    # cpu
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])

    return model

def run_inference(model, img_path):
    model.eval()
    with torch.no_grad():
        unlabeled_x = transform_pipeline(img_path).unsqueeze(0)
        outputs = model(unlabeled_x)
    inv_unlabeled_x = invTrans(unlabeled_x)
    return draw_pred_pil(inv_unlabeled_x, outputs, 0)


def run_inference_xml(model, img_path):
    model.eval()
    with torch.no_grad():
        unlabeled_x = transform_pipeline(img_path).unsqueeze(0)
        outputs = model(unlabeled_x)

    outputs = outputs.squeeze()
    # scale from 256*256 to the image size after padding
    padded_img = pad_to_square_from_path(img_path)
    
    # testing purpose

    outputs = outputs * (padded_img.shape[-1]/256)

    img = Image.open(img_path).convert('RGB') # Open and convert to RGB for consistency
    img_tensor = transform(img) # Convert image to tensor
    
    _, h, w = img_tensor.size()
    diff = abs(h - w)
    padding = diff // 2

    if h > w:
        pads = (padding, diff - padding, 0, 0)
        outputs[0] = outputs[0] - padding # circle x
        for i in range(3, 13, 2):
            outputs[i] = outputs[i] - padding
    else:
        pads = (0, 0, padding, diff - padding)
        outputs[1] = outputs[1] - padding  # circle y
        for i in range(4, 13, 2):
            outputs[i] = outputs[i] - padding
    return outputs


def infer_annot(dir_path, model_path, annot_path):
    
    model = load_model(model_path)
    
    results = find_images(dir_path, model)

    inferred = results[:2]
    filenames = results[0]
    measurements = results[2]

    zipped = zip(*inferred)

    mapping = {k:v for k,v in zipped}

    target_tree = ET.parse(annot_path)
    target_root = target_tree.getroot()


    for child in target_root:
        if child.tag!= "image":
            continue    
        img_name = child.attrib["name"].split("/")[-1]
        try:
            img_map = mapping[img_name]
        except:
            print(f"error in mapping file name: {img_name}")
            continue   

        # humeral head
        landmark = ET.SubElement(child, "ellipse")
        landmark.set("label",  'Circle fit to humeral articular surface')
        landmark.set("source", "manual")
        landmark.set("occluded", "0")
        landmark.set("cx", f"{img_map[0]}")
        landmark.set("cy", f"{img_map[1]}")
        landmark.set("rx", f"{img_map[2]}")
        landmark.set("ry", f"{img_map[2]}")

        landmark.set("z_order", "0")

        # notch
        landmark = ET.SubElement(child, "ellipse")
        landmark.set("label",  'Spinoglenoid notch')
        landmark.set("source", "manual")
        landmark.set("occluded", "0")
        landmark.set("cx", f"{img_map[3]}")
        landmark.set("cy", f"{img_map[4]}")
        landmark.set("rx", "2")
        landmark.set("ry", "2")
        landmark.set("z_order", "0")

        #apcs
        landmark = ET.SubElement(child, "points")
        landmark.set("label", "Anterior lip of glenoid fossa")
        landmark.set("source", "manual")
        landmark.set("occluded", "0")
        landmark.set("points", f"{img_map[5]},{img_map[6]}")
        landmark.set("z_order", "0")



        # p 
        landmark = ET.SubElement(child, "points")
        landmark.set("label", "Posterior lip of glenoid fossa")
        landmark.set("source", "manual")
        landmark.set("occluded", "0")
        landmark.set("points", f"{img_map[7]},{img_map[8]}")
        landmark.set("z_order", "0")

        # c
        landmark = ET.SubElement(child, "points")
        landmark.set("label", "Center of glenoid fossa")
        landmark.set("source", "manual")
        landmark.set("occluded", "0")
        landmark.set("points", f"{img_map[9]},{img_map[10]}")
        landmark.set("z_order", "0")

        # s
        landmark = ET.SubElement(child, "points")
        landmark.set("label", "Scapular body point")
        landmark.set("source", "manual")
        landmark.set("occluded", "0")
        landmark.set("points", f"{img_map[11]},{img_map[12]}")
        landmark.set("z_order", "0")



    # child = ET.SubElement(root_node, "user")
    # child.set("username","srquery")
    # group  = ET.SubElement(child,"group")
    # group.text = "fresher"
    # tree = ET.ElementTree(root_node)
    # tree.write("users.xml")

    target_tree.write(f"inferred_annotations.xml")

    measurements_df = pd.DataFrame(columns=["filename", "Centering", "Retroversion", "Radius"], index = np.arange(len(filenames)))
    measurements_df["filename"] = filenames
    measurements_df["Centering"] = [i[0] for i in measurements]
    measurements_df["Retroversion"] = [i[1] for i in measurements]
    measurements_df["Radius"] = [i[2].item() for i in measurements]

    try:
        measurements_df.to_csv("inferred_measurements.csv")
    except:
        print("Measurement csv dump failed.")
    


if __name__ == "__main__":
    pass
