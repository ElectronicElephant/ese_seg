import cv2 as cv
import numpy as np
from PIL import Image
import os
import pickle
from tqdm import tqdm
from lxml.etree import Element, SubElement, tostring
from xml.dom.minidom import parseString

# for loading mat
from scipy.io import loadmat

root = "/home/tutian/dataset/coco_to_voc/val"
instance_dir = os.path.join(root, "instance_labels")
sem_dir = os.path.join(root, "class_labels")

COCO_CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                'scissors', 'teddy bear', 'hair drier', 'toothbrush')

COCO_LABEL_MAP = { 1:  1,  2:  2,  3:  3,  4:  4,  5:  5,  6:  6,  7:  7,  8:  8,
                   9:  9, 10: 10, 11: 11, 13: 12, 14: 13, 15: 14, 16: 15, 17: 16,
                  18: 17, 19: 18, 20: 19, 21: 20, 22: 21, 23: 22, 24: 23, 25: 24,
                  27: 25, 28: 26, 31: 27, 32: 28, 33: 29, 34: 30, 35: 31, 36: 32,
                  37: 33, 38: 34, 39: 35, 40: 36, 41: 37, 42: 38, 43: 39, 44: 40,
                  46: 41, 47: 42, 48: 43, 49: 44, 50: 45, 51: 46, 52: 47, 53: 48,
                  54: 49, 55: 50, 56: 51, 57: 52, 58: 53, 59: 54, 60: 55, 61: 56,
                  62: 57, 63: 58, 64: 59, 65: 60, 67: 61, 70: 62, 72: 63, 73: 64,
                  74: 65, 75: 66, 76: 67, 77: 68, 78: 69, 79: 70, 80: 71, 81: 72,
                  82: 73, 84: 74, 85: 75, 86: 76, 87: 77, 88: 78, 89: 79, 90: 80}

def save_xml(img_name, cat_list, pointsList, save_dir, width, height, channel):
    has_objects = False
    node_root = Element('annotation')
    node_folder = SubElement(node_root, 'folder')
    node_folder.text = 'JPEGImages'

    node_filename = SubElement(node_root, 'filename')
    node_filename.text = img_name

    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = '%s' % width

    node_height = SubElement(node_size, 'height')
    node_height.text = '%s' % height

    node_depth = SubElement(node_size, 'depth')
    node_depth.text = '%s' % channel

    count = 0
    for points in pointsList:
        bbox_center_x, bbox_center_y = points[3], points[4]
        bbox_w, bbox_h = points[5], points[6]
        bbox_xmin, bbox_ymin = bbox_center_x - bbox_w / 2.0, bbox_center_y - bbox_h / 2.0
        bbox_xmax, bbox_ymax = bbox_center_x + bbox_w / 2.0, bbox_center_y + bbox_h / 2.0
        coef_center_x, coef_center_y = points[7], points[8]

        coef = points[9:]
        coef_str = str(points[9:])
        coef_str = coef_str[1:-1]
        node_object = SubElement(node_root, 'object')
        node_name = SubElement(node_object, 'name')
        node_name.text = COCO_CLASSES[cat_list[count] - 1]
        node_difficult = SubElement(node_object, 'difficult')
        node_difficult.text = '0'
        node_bndbox = SubElement(node_object, 'bndbox')
        node_xmin = SubElement(node_bndbox, 'xmin')
        node_xmin.text = '%s' % bbox_xmin
        node_ymin = SubElement(node_bndbox, 'ymin')
        node_ymin.text = '%s' % bbox_ymin
        node_xmax = SubElement(node_bndbox, 'xmax')
        node_xmax.text = '%s' % bbox_xmax
        node_ymax = SubElement(node_bndbox, 'ymax')
        node_ymax.text = '%s' % bbox_ymax
        # node_coef_center_x = SubElement(node_bndbox, 'coef_center_x')
        # node_coef_center_x.text = '%s' % coef_center_x
        # node_coef_center_y = SubElement(node_bndbox, 'coef_center_y')
        # node_coef_center_y.text = '%s' % coef_center_y
        node_polygon = SubElement(node_object, 'coef')
        node_polygon.text = '%s' % coef_str
        count += 1
        has_objects = True
    xml = tostring(node_root, pretty_print=True)
    dom = parseString(xml)

    if has_objects:
        with open(os.path.join(root, 'coef_8_success.txt'), 'a') as f:
            f.write(img_name + '\n')
    save_xml = os.path.join(save_dir, img_name + '.xml')
    with open(save_xml, 'wb') as f:
        f.write(xml)

def getBoundingBox(mask):
    '''
    By WenQiang
    :param mask:
    :return:
    '''
    coords = np.transpose(np.nonzero(mask))
    x, y, w, h = cv.boundingRect(coords)
    return x, y, w, h


def runOneImage(img_path):
    instance_mask = Image.open(os.path.join(instance_dir, img_path))  # PIL
    # instance_mat = loadmat(img_path)
    # instance_mask = instance_mat['GTinst'][0, 0]['Segmentation']
    instance_mask = np.array(instance_mask)
    instance_ids = np.unique(instance_mask)
    semantic_mask = np.array(Image.open(os.path.join(sem_dir, img_path)))
    # sem_mat = loadmat(img_path.replace("inst", "cls"))
    # semantic_mask = sem_mat['GTcls'][0, 0]['Segmentation']
    img_name = img_path.split('/')[-1]

    img_height, img_width = instance_mask.shape
    img_info_dict = []
    for instance_id in instance_ids:
        objects_info = {}
        if instance_id == 0 or instance_id == 255:  # background or edge, pass
            continue
        # extract instance
        temp = np.zeros(instance_mask.shape)
        temp.fill(instance_id)
        tempMask = (instance_mask == temp)
        cat_id = np.max(semantic_mask * tempMask)  # semantic category of this instance
        instance = instance_mask * tempMask

        # BoundingBox
        x, y, w, h = getBoundingBox(instance)
        assert x+w <= img_height and y+h <=img_width

        # Crop the mask and get the coeffs
        instance_mask_ = instance[x:x + w, y:y + h].astype(np.bool) * 255
        instance_mask_ = Image.fromarray(instance_mask_.astype(np.uint8)).resize((64, 64), Image.NEAREST)
        instance_mask_ = np.reshape(instance_mask_, (-1, 64 * 64))
        coeffs = dico.transform(instance_mask_).astype('float64')  # Just put the raw coef into XML

        # Here x, y is the center
        x += w/2
        y += h/2
        objects_info['label'] = COCO_LABEL_MAP[cat_id]  # Convert from 1-90 to 1-80
        objects_info['bbox'] = (y, x, h, w)  # TO BE CAREFUL
        objects_info['img_wh'] = (img_width, img_height)
        # objects_info['center'] = (center_x,center_y)
        # No need for center at all
        objects_info['coeffs'] = coeffs
        img_info_dict.append(objects_info)

    info_txt = np.zeros((len(img_info_dict), 9 + n_components))
    for i in range(len(img_info_dict)):
        info_txt[i][0] = img_info_dict[i]['label']
        info_txt[i][1:3] = img_info_dict[i]['img_wh']
        info_txt[i][3:7] = img_info_dict[i]['bbox']
        # info_txt[i][7:9] = img_info_dict[i]['center']
        info_txt[i][9:] = img_info_dict[i]['coeffs']
    # np.savetxt(os.path.join(label_dir_txt, img_name[:-4] + '.txt'), info_txt)
    img_info = np.reshape(info_txt, (-1, 9 + n_components))
    cat_list = []  # Cat list in one img
    for i in range(len(img_info)):
        cat_id = int(img_info[i][0])
        cat_list.append(cat_id)

    points_list = img_info
    width = img_info[0][1]
    height = img_info[0][2]
    channel = 3

    save_xml(img_name[:-4], cat_list, points_list, save_dir, width, height, channel)


if __name__ == "__main__":
    inst_list = os.listdir(instance_dir)

    # XML
    save_dir = os.path.join(root, 'bases_' + str(50) + '_xml_raw_coef')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    model_path = '/home/tutian/dataset/coco_to_voc/coco_all_50_1.sklearnmodel'
    n_components = 50
    dico = pickle.load(open(f'{model_path}', 'rb'))
    # dico is treated as the global variable

    for i in tqdm(range(len(inst_list))):
        runOneImage(inst_list[i])
