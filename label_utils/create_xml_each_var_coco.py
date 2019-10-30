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

root = "/home/tutian/dataset/coco_to_voc/train"
instance_dir = os.path.join(root, "instance_labels")
sem_dir = os.path.join(root, "class_labels")

# labels = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
#           "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

x_min =  [-15408.068104448881, -6893.558054798728, -7003.406866817996, -7173.151488944284, -8880.702237736832, -5105.870172246976, -5765.5587195891485, -5024.227379613461, -5711.952435731431, -5495.081529198267, -5833.850420273756, -4434.37549020221, -5849.216285241527, -4148.2654407091895, -3569.2531463158916, -4339.357174902734, -3655.7764618342203, -3823.3819004419747, -3141.4357750292143, -4225.954414632274, -4508.907524652018, -2985.9986722598996, -3351.4766979792385, -3542.6383142662216, -3208.1730852282417, -3276.2051016720184, -2778.240479008936, -2687.1807642675817, -2864.3521512732636, -2667.346488961604, -2679.78247499033, -2778.1530493300193, -2615.297232543604, -2887.83922977382, -2814.11271273744, -2665.593586967864, -2244.208215546852, -2604.715325774133, -2555.901894909533, -3023.0542016462905, -3120.604337844805, -2276.2895359281847, -2105.2348396526972, -2107.14859953116, -4062.8254106434965, -2053.622120297776, -2197.4795855647635, -2042.3037948693445, -2467.5308906646937, -2245.5552141163903]
x_max =  [0.0, 6832.446298223013, 7426.165815379417, 6974.701596658017, 4716.901065835743, 8131.608870119551, 5740.872699165772, 4581.338796015798, 5217.3107185273375, 5434.597380283167, 5576.999587107373, 4287.165831371201, 4963.129599067099, 4621.02114880624, 3682.6609034386593, 4353.761120273803, 4174.824769494295, 3994.883741475415, 3283.721646183678, 3798.4092325829133, 4347.6387582645475, 3372.640698902529, 3295.0094768303293, 2926.3658864426816, 3499.712903749524, 3039.4470982219764, 2473.9809720368858, 2405.556357232199, 3184.463910105855, 2784.1799697475394, 2284.209254236527, 2625.629675147772, 2336.795159840813, 2528.887489215271, 2782.44841959135, 2342.962374129638, 2477.479578295029, 2332.187232909927, 2459.4770586568147, 2794.3178970248023, 2505.2624769384856, 2767.461569799445, 1918.2837463541125, 2050.6555855719203, 2690.2851498377295, 2887.8565628719634, 2263.3678542969415, 1798.6753995660308, 2160.58798020158, 2092.1122966365115]
x_mean =  [-9806.334230601844, -0.1265930578759492, -44.70213815499062, 4.1016068564528485, 34.85025642973737, -3.515908079075314, 33.660171096323424, 130.83580930988637, 0.21492417056751245, 2.8112355899964174, 18.833675030236837, 0.626437650033731, -3.2008816942056932, 0.016458105852027838, 8.394310893579835, -5.059975166016848, 0.3082644590455881, 1.5217574906226543, -0.018611740148539873, 0.7879045499805826, -0.24098315206080123, -0.8808304685364998, -0.7913288600067822, -3.8891420056181145, 6.353012221300202, -0.4225753767008447, 0.27977828714261016, 0.08870383388150666, -5.0118744432067786, 0.48268046843874046, -18.893481918065138, 0.7532384238847303, -5.311672820484189, -6.17895441522754, -0.356883920263817, -0.38091052476647386, 0.08936253734500309, 1.2569901866919777, 1.4373361126170598, -3.279811354042419, -2.068920651918281, 0.060461234684045725, 0.6868672104607721, 0.03698304732462165, -2.532655293110934, -0.1347230399250139, 1.5058533210691571, 0.09911752586840517, -0.012458458813556523, 2.3168010192166624]
sqrt_var =  [3178.8067937849487, 2108.5508769810863, 1994.220493314925, 1938.7995338537069, 1639.3960276470855, 1432.760749474181, 1288.8778997753777, 1173.580613711433, 1122.0012218569218, 928.4866017001266, 921.4166617204439, 856.0066864535319, 827.1107657788409, 801.1389228102764, 747.0316068891458, 738.6573541712918, 713.7510960451755, 656.5846272310993, 644.8258954808872, 606.805027464974, 596.0518795953916, 588.3050234920775, 586.3892172394093, 554.8030689406621, 543.5063777522089, 503.09736918051834, 496.38691611492146, 488.43183601616107, 487.21877068107796, 476.8424720930743, 459.66215884985763, 442.3700766285788, 435.6704169622154, 429.36612409375545, 410.33755900022, 408.4439272034049, 404.64446380132824, 394.1204334571845, 393.73104227507173, 389.3877034575619, 381.867970587664, 372.6268526542834, 358.85596109586754, 357.9271102515216, 352.67275163779277, 348.4594642007308, 344.12421096240666, 343.61001513303694, 331.4995424542531, 326.8226821028282]

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
        coeffs = dico.transform(instance_mask_).astype('float64')
        assert coeffs.shape == (1, 50)
        coeffs = (coeffs - x_mean) / sqrt_var
        # coeffs = (coeffs - x_min) / (x_max - x_min)

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
    save_dir = os.path.join(root, 'bases_' + str(50) + '_xml_each_var')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    model_path = '/home/tutian/dataset/model/coco_all_50_1.sklearnmodel'
    n_components = 50
    dico = pickle.load(open(f'{model_path}', 'rb'))
    # dico is treated as the global variable

    for i in tqdm(range(len(inst_list))):
        # runOneImage(os.path.join(instance_dir, inst_list[i]))
        runOneImage(inst_list[i])
