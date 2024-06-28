from utils.parse_xml import xmlParser
from utils.helper import set_seed
from utils.preprocessing import *
import argparse
import os

parser = argparse.ArgumentParser(
                    prog='extract_annotations',
                    description='Extract tensors from annotations')


parser.add_argument('--xml_path', type=str, default='./data_padded/annotations/annotations2_3.xml', action = "store")
parser.add_argument('--save_path', type=str, default='.', action = "store")
parser.add_argument('--tensor_suffix', type=str, default="2_3", action = "store")
parser.add_argument("--img_folder", type=str, default="./data_padded", action = "store")
# for third chunk, use "./data_padded/", and put corresponding photos into 3rd file, just to match annotation part. 
parser.add_argument("--img_size", type=tuple, default=(256,256), action = "store")
## starts parsing ....

args = parser.parse_args()

# final variables
COLOR_CYC = ['b','g','r','c','m','y']

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


xml_path = args.xml_path
save_path = args.save_path
data = xmlParser(xml_path)
annotations = data.get_annotations()
image_folder = args.img_folder
# sample_image = os.path.join(image_folder, sample_image_name)


X, y = annotation_to_tensor(data, annotations, args.img_size, image_folder = image_folder)

suf = args.tensor_suffix

try:
    torch.save(X, os.path.join(save_path, f'X_{suf}.pt') )
    torch.save(y, os.path.join(save_path, f'y_{suf}.pt'))
except:
    print('Error saving files')