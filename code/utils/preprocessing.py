import cv2
from tqdm import tqdm
from utils.parse_xml import xmlParser
import torch
import os
from PIL import Image
import torchvision.transforms as transforms

LABEL_ORDERS_DEFAULT = {'Circle fit to humeral articular surface':0,
 'Spinoglenoid notch':1,
 'Anterior lip of glenoid fossa':2,
 'Posterior lip of glenoid fossa':3,
 'Center of glenoid fossa':4,
 'Scapular body point':5}

LABEL_ORDERS_DEFAULT_LST = ['Circle fit to humeral articular surface',
 'Spinoglenoid notch',
 'Anterior lip of glenoid fossa',
 'Posterior lip of glenoid fossa',
 'Center of glenoid fossa',
 'Scapular body point'] 

def resize_point(img, point, target_size=(256, 256)):
    # point: tuple of (x, y)
    # img needs to be an image object

    y, x = img.shape[1:]
    scale_x = target_size[1] / x
    scale_y = target_size[0] / y
    return point[0]*scale_x, point[1]*scale_y

def resize_ellipse(img, ellipse, target_size=(256, 256)):
    # ellipse: quadruple (cx, cy, rx, ry)
    # img needs to be an image object

    y, x = img.shape[1:]
    # check y==x.

    scale_x = target_size[1] / x
    scale_y = target_size[0] / y

    return ellipse[0]*scale_x, ellipse[1]*scale_y, ellipse[2]*scale_x, ellipse[3]*scale_y


def annotation_to_tensor(data: xmlParser, annotations: dict, target_size: tuple, num_ell=2, num_pts=4, image_folder = ".") -> None:

    # image in pytorch:  [N, C, H, W] 
    X = torch.zeros(len(annotations), 3, target_size[1], target_size[0])
    
    _k = next(iter(annotations))
    # this should be changed
    # ellipse is actually one circle and one point 
    # yyshape = num_ell * 4 + num_pts * 2
    yyshape = num_ell * 4 + num_pts * 2 - 1 - 2
    target = torch.zeros(len(annotations), yyshape)

    # y columns in the order 
    # (cx_1, cy_1, rx_1, ry_1,...,cx_n, cy_n, rx_n, ry_n , x_1, y_1,..., x_n, y_n)

    # annotations
    img_names =  data.get_image_names()    
    for idx, sample_image_name in tqdm(enumerate(img_names)):
        sample_image = os.path.join(image_folder, sample_image_name)
        # img = cv2.imread(sample_image)
        # # print(sample_image_name)
        # resized_img = cv2.resize(img, target_size)

        # X[idx, :, :, :] = torch.from_numpy(resized_img).permute(2, 1, 0)



        tt = transforms.PILToTensor()
        rz = transforms.Resize(target_size)

        img = Image.open(sample_image).convert("RGB")
        img = tt(img)
        X[idx, :, :, :] = rz(img)





        ys = [float("inf") for _ in range(yyshape)]

        for ellipse in annotations[sample_image_name]['ellipse']:
            ell = tuple(map(float, (ellipse["cx"], ellipse["cy"], ellipse["rx"], ellipse["ry"])))

            cx, cy, rx, ry = resize_ellipse(img, ell, target_size)
            if ellipse["label"] == 'Circle fit to humeral articular surface':
                # start = LABEL_ORDERS_DEFAULT[ellipse["label"]]*4
                start = 0
                end = start + 3
                ys[start:end] = [cx, cy, (rx+ry)/2]
            if ellipse["label"] == 'Spinoglenoid notch':
                start = 3
                end = start + 2
                ys[start:end] = [cx, cy]

        for keypoints in annotations[sample_image_name]['keypoints']:
            x, y = map(float, keypoints["points"].split(","))
            x, y = resize_point( img, (x, y), target_size)

            # start = num_ell*4 + (LABEL_ORDERS_DEFAULT[keypoints["label"]]-num_ell)*2
            # starting index of point + point specfic starting position
            start = 5 + (LABEL_ORDERS_DEFAULT[keypoints["label"]]-num_ell)*2
            end = start + 2

            ys[start:end] = [x,y]
        try:
            assert len(ys) == yyshape
        except:
            print("**********")
            print(f"Bad annotation on {sample_image_name}")
            if len(ys)<yyshape:
                # pad ys to length yyshape
                ys.extend([0]*(yyshape-len(ys)))
                print("Missing points")
            else:
                # truncate ys to length yyshape
                ys = ys[:yyshape]
                print("Too many points")
            print("**********")
            
            

        target[idx, :] = torch.Tensor(ys)
    return X, target
