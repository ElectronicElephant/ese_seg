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

root = "../sbd"
instance_dir = os.path.join(root, "inst")
sem_dir = os.path.join(root, "cls")

labels = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
          "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

x_mean = [-10372.72573, -65.62571687, 8.113603703, -4.129944152, -70.272901, 48.39311715, 13.54921555, -6.204910281, -82.93014464, -13.49012528, -1.377779259, -0.359250444, -9.937065122, 3.458047501, -0.637840469, 6.447263647, -0.159122537, -4.013595629, 0.368631004, -0.798153475, -0.675555162, -0.64375462, -1.876287186, -5.29987036, -3.081862721, -1.205230327, 1.611716191, -1.447915821, -1.008998948, 2.062282999, -0.366824452, -0.76531215, -5.657952825, -0.702878769, 4.139859116, 3.660853075, -4.365368841, -3.759996972, 0.10982376, -0.901409142, -0.914115701, -0.287375188, 0.673038067, -1.64012666, 0.983785635, 0.369574124, -0.080789953, 1.392399963, 1.066113083, -1.677959563]
sqrt_var = [2803.685003, 2161.867545, 2087.585027, 1877.30127, 1616.864336, 1459.067858, 1325.125214, 1166.808235, 1078.785935, 954.0199264, 941.2122393, 877.5677226, 822.7808391, 767.6920133, 742.2590864, 709.1605303, 707.289721, 671.4039817, 625.9553238, 595.3469511, 586.0426203, 571.3110848, 560.1314211, 535.6616464, 534.0023283, 508.0441017, 489.9292073, 485.8909739, 474.3963334, 466.0055539, 449.4863225, 444.2908041, 437.5323292, 432.4269679, 409.8391652, 406.6126991, 400.0697815, 396.4458867, 391.1772833, 384.5596921, 377.9070695, 371.8721299, 366.0097442, 350.6612788, 350.3757641, 347.7540654, 343.1649469, 335.2165795, 332.2271304, 327.4941672]


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
        coef = points[9:]
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
        node_name.text = labels[cat_list[count] - 1]
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
        node_coef_center_x = SubElement(node_bndbox, 'coef_center_x')
        node_coef_center_x.text = '%s' % coef_center_x
        node_coef_center_y = SubElement(node_bndbox, 'coef_center_y')
        node_coef_center_y.text = '%s' % coef_center_y
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
    # instance_mask = Image.open(img_path)  # PIL
    instance_mat = loadmat(img_path)
    instance_mask = instance_mat['GTinst'][0, 0]['Segmentation']
    instance_mask = np.array(instance_mask)
    instance_ids = np.unique(instance_mask)
    # semantic_mask = np.array(Image.open(img_path.replace("inst", "cls")))
    sem_mat = loadmat(img_path.replace("inst", "cls"))
    semantic_mask = sem_mat['GTcls'][0, 0]['Segmentation']
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
        coeffs = dico.transform(instance_mask_).astype('float64')
        # np.clip(coeffs, -2500, 2500, coeffs)
        # coeffs = coeffs / 5000  # clip to -0.5 to 0.5
        # print(f'{coeffs.shape}, {coeffs[0]}')
        # assert (np.max(coeffs) <= 2500 and np.min(coeffs) >= -2500)
        # if coeffs.max() == coeffs.min():
        #     print(f'coeffs.max == min occurred on img {img_path} instance id {instance_id}')
        #     continue
        # print(coeffs.shape)
        assert coeffs.shape == (1, 50)
        # coeffs = 2 * (coeffs - coeffs.min()) / (coeffs.max() - coeffs.min()) - 1
        coeffs = (coeffs - x_mean) / sqrt_var
        # print(np.unique(coeffs))
        # assert (np.max(coeffs) == 1 and np.min(coeffs) == -1)

        # Here x, y is the center
        x += w/2
        y += h/2
        objects_info['label'] = cat_id
        objects_info['bbox'] = (y, x, h, w)  # TO BE CAREFUL
        objects_info['img_wh'] = (img_width, img_height)
        # objects_info['center'] = (center_x,center_y)
        # No need for center at all
        objects_info['coeffs'] = coeffs
        img_info_dict.append(objects_info)
    # with open(os.path.join(label_dir_pkl, img_name[:-4] + '.pkl'), 'wb') as fpkl:
    #     pickle.dump(img_info_dict, fpkl)
    info_txt = np.zeros((len(img_info_dict), 9 + n_components))
    for i in range(len(img_info_dict)):
        info_txt[i][0] = img_info_dict[i]['label']
        info_txt[i][1:3] = img_info_dict[i]['img_wh']
        info_txt[i][3:7] = img_info_dict[i]['bbox']
        # info_txt[i][7:9] = img_info_dict[i]['center']
        info_txt[i][9:] = img_info_dict[i]['coeffs']
    # np.savetxt(os.path.join(label_dir_txt, img_name[:-4] + '.txt'), info_txt)
    img_info = np.reshape(info_txt, (-1, 59))
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
    save_dir = os.path.join(root, 'bases_' + str(50) + '_xml_each_var')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    path = '/disk1/home/tutian/ese_seg/label_utils'
    n_components = 50
    n_iter = 1
    dico = pickle.load(open(f'{path}/all_{n_components}_{n_iter}.sklearnmodel', 'rb'))
    # dico is treated as the global variable

    for i in tqdm(range(len(inst_list))):
        runOneImage(os.path.join(instance_dir, inst_list[i]))
