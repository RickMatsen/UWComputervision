import xml.etree.ElementTree as ET
import numpy as np

class xmlParser:

    def __init__(self, xml_path):
        self.tree = ET.parse(xml_path)
        self.root = self.tree.getroot()
        self.labels = []
        self.images = {}
        self.images_names = []
        self.annotations = {}

        # set label_names
        self._set_label_names()
        
        # set images and their names
        self._set_image()
        self._set_image_names()
    
    def _set_label_names(self):
        xml_labels = self.root.find('.//labels') # return element <labels>
        for label in xml_labels:
            xml_label = label.find('.//name')
            if xml_label is not None:
                self.labels.append(xml_label.text)

    def _set_image(self): # get the all the annotation
        xml_images = self.root.findall('.//image')
        for xml_image in xml_images:
            # Get image attributes dict: id, name, width, height
            image_attrib = xml_image.attrib
            image_name = (image_attrib['name'])
            self.images[image_name] = image_attrib

            self.annotations[image_name] = {
            'ellipse': [],
            'keypoints': []
        }   
            w, h = float(self.images[image_name]['width']), float(self.images[image_name]['height'])
            diff = abs(h - w)
            padding = diff // 2
            
            if h > w:
                for item in xml_image:
                    if item.tag == 'ellipse':
                        temp = item.attrib         
                        temp["cx"] = str(padding+float(item.attrib["cx"]))
                        # ell = tuple(map(float, (ellipse["cx"], ellipse["cy"], ellipse["rx"], ellipse["ry"])))
                        self.annotations[image_name]['ellipse'].append(temp)
                    elif item.tag == 'points':
                        temp = item.attrib
                        x, y =  map(float, temp["points"].split(","))
                        x = x+padding
                        temp["points"] = str(x) + "," + str(y)
                        self.annotations[image_name]['keypoints'].append(temp)
                    else:
                        raise NotImplementedError('Unimplemented annotation type: {}'.format(item.tag))

            else:
                for item in xml_image:
                    if item.tag == 'ellipse':
                        temp = item.attrib         
                        temp["cy"] = str(padding+float(item.attrib["cy"]))
                        self.annotations[image_name]['ellipse'].append(temp)
                    elif item.tag == 'points':
                        temp = item.attrib
                        x, y =  map(float, temp["points"].split(","))
                        y = y+padding
                        temp["points"] = str(x) + "," + str(y)
                        self.annotations[image_name]['keypoints'].append(temp)
                    else:
                        raise NotImplementedError('Unimplemented annotation type: {}'.format(item.tag))

    def _set_image_names(self):
        for image in self.images.values():
            self.images_names.append(image['name'])

    def get_image_sizes(self) -> np.ndarray:
        image_sizes = []
        for name in self.images_names:
            image_sizes.append([self.images[name]['width'], self.images[name]['height']])
        return np.array(image_sizes)

    def get_annotations(self) -> dict:
        return self.annotations
    
    def get_image_names(self) -> list:
        return self.images_names
    
    def get_labels(self) -> list:
        return self.labels
    

   
                