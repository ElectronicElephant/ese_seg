"""Pascal VOC object detection dataset."""
from __future__ import absolute_import
from __future__ import division
import os
import logging
import warnings
import numpy as np
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import mxnet as mx
from ..base import VisionDataset


class VOCDetection(VisionDataset):
    """Pascal VOC detection Dataset.
    The train Dataset of Our Model, has the Coefficients label, not Polygon Points label
    You should origanize your dataset as pascal voc format first.
    Parameters
    ----------
    root : str, default '~/mxnet/datasets/voc'
        Path to folder storing the dataset.
    splits : list of tuples, default ((2007, 'trainval'), (2012, 'trainval'))
        List of combinations of (year, name)
        For years, candidates can be: 2007, 2012.
        For names, candidates can be: 'train', 'val', 'trainval', 'test'.
    transform : callable, default None
        A function that takes data and label and transforms them. Refer to
        :doc:`./transforms` for examples.

        A transform function for object detection should take label into consideration,
        because any geometric modification will require label to be modified.
    index_map : dict, default None
        In default, the 20 classes are mapped into indices from 0 to 19. We can
        customize it by providing a str to int dict specifying how to map class
        names to indices. Use by advanced users only, when you want to swap the orders
        of class labels.
    preload_label : bool, default True
        If True, then parse and load all labels into memory during
        initialization. It often accelerate speed but require more memory
        usage. Typical preloaded labels took tens of MB. You only need to disable it
        when your dataset is extremely large.
    """
    CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
               'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')

    def __init__(self, root='/home/tutian/dataset',
                 # splits=((2007, 'trainval'), (2012, 'trainval')),
                 splits=((2012, 'train')),
                 transform=None, index_map=None, preload_label=True):
        super(VOCDetection, self).__init__(root)
        self._im_shapes = {}
        self._root = root
        self._transform = transform
        self._splits = splits
        self._items = self._load_items(splits)
        self._anno_path = os.path.join('{}', './bases_50_xml_each_var', '{}.xml')
        self._image_path = os.path.join('{}', 'img', '{}.jpg')
        self.index_map = index_map or dict(zip(self.classes, range(self.num_class)))
        self._label_cache = self._preload_labels() if preload_label else None

    def __str__(self):
        detail = ','.join([str(s[0]) + s[1] for s in self._splits])
        return self.__class__.__name__ + '(' + detail + ')'

    @property
    def classes(self):
        """Category names."""
        try:
            self._validate_class_names(self.CLASSES)
        except AssertionError as e:
            raise RuntimeError("Class names must not contain {}".format(e))
        return type(self).CLASSES

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        img_id = self._items[idx]
        img_path = self._image_path.format(*img_id)
        label = self._label_cache[idx] if self._label_cache else self._load_label(idx)
        img = mx.image.imread(img_path, 1)
        if self._transform is not None:
            return self._transform(img, label)
        return img, label

    def _load_items(self, splits):
        """Load individual image indices from splits."""
        ids = []
        for year, name in splits:
            root = os.path.join(self._root, 'VOC' + str(year))
            lf = os.path.join(root, 'ImageSets', 'Segmentation', name + '.txt')
            with open(lf, 'r') as f:
                ids += [(root, line.strip()) for line in f.readlines()]
        return ids

    def _load_label(self, idx):
        """Parse xml file and return labels."""
        img_id = self._items[idx]
        anno_path = self._anno_path.format(*img_id)
        root = ET.parse(anno_path).getroot()
        size = root.find('size')
        width = float(size.find('width').text)
        height = float(size.find('height').text)
        if idx not in self._im_shapes:
            # store the shapes for later usage
            self._im_shapes[idx] = (width, height)
        label = []
        for obj in root.iter('object'):
            obj_label_info = []
            difficult = int(obj.find('difficult').text)
            cls_name = obj.find('name').text.strip().lower()
            if cls_name not in self.classes:
                continue
            cls_id = self.index_map[cls_name]
            xml_box = obj.find('bndbox')
            xmin = (float(xml_box.find('xmin').text))
            ymin = (float(xml_box.find('ymin').text))
            xmax = (float(xml_box.find('xmax').text))
            ymax = (float(xml_box.find('ymax').text))
            xml_coef = obj.find('coef').text
            xml_coef = xml_coef.split()
            coef = [float(xml_coef[i]) for i in range(len(xml_coef))]
            obj_label_info.append(xmin)
            obj_label_info.append(ymin)
            obj_label_info.append(xmax)
            obj_label_info.append(ymax)
            for i in range(len(coef)):
                obj_label_info.append(coef[i])
            obj_label_info.append(cls_id)
            obj_label_info.append(difficult)
            obj_label_info.append(width)
            obj_label_info.append(height)
            try:
                self._validate_label(xmin, ymin, xmax, ymax, width, height)
            except AssertionError as e:
                raise RuntimeError("Invalid label at {}, {}".format(anno_path, e))
            label.append(obj_label_info)
        return np.array(label)

    def _validate_label(self, xmin, ymin, xmax, ymax, width, height):
        """Validate labels."""
        assert 0 <= xmin < width, "xmin must in [0, {}), given {}".format(width, xmin)
        assert 0 <= ymin < height, "ymin must in [0, {}), given {}".format(height, ymin)
        assert xmin < xmax <= width, "xmax must in (xmin, {}], given {}".format(width, xmax)
        assert ymin < ymax <= height, "ymax must in (ymin, {}], given {}".format(height, ymax)

    def _validate_class_names(self, class_list):
        """Validate class names."""
        assert all(c.islower() for c in class_list), "uppercase characters"
        stripped = [c for c in class_list if c.strip() != c]
        if stripped:
            warnings.warn('white space removed for {}'.format(stripped))

    def _preload_labels(self):
        """Preload all labels into memory."""
        logging.debug("Preloading %s labels into memory...", str(self))
        return [self._load_label(idx) for idx in range(len(self))]


class coco_pretrain_Detection(VisionDataset):
    """coco pretrain detection Dataset.
    The pretrain Dataset of Our Model, only train bbox (has the Coefficients label, not Polygon Points label)
    Parameters
    ----------
    root : str, default '~/mxnet/datasets/voc'
        Path to folder storing the dataset.
    splits : list of tuples, default ((2007, 'trainval'), (2012, 'trainval'))
        List of combinations of (year, name)
        For years, candidates can be: 2007, 2012.
        For names, candidates can be: 'train', 'val', 'trainval', 'test'.
    transform : callable, default None
        A function that takes data and label and transforms them. Refer to
        :doc:`./transforms` for examples.

        A transform function for object detection should take label into consideration,
        because any geometric modification will require label to be modified.
    index_map : dict, default None
        In default, the 20 classes are mapped into indices from 0 to 19. We can
        customize it by providing a str to int dict specifying how to map class
        names to indices. Use by advanced users only, when you want to swap the orders
        of class labels.
    preload_label : bool, default True
        If True, then parse and load all labels into memory during
        initialization. It often accelerate speed but require more memory
        usage. Typical preloaded labels took tens of MB. You only need to disable it
        when your dataset is extremely large.
    """
    CLASSES = ('airplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'dining table', 'dog', 'horse', 'motorcycle',
               'person', 'potted plant', 'sheep', 'couch', 'train', 'tv')

    def __init__(self, root='/home/wenqiang/tutian_temp/coco_ese_seg',
                 # splits=((2007, 'trainval'), (2012, 'trainval')),
                 splits=((2012, 'train')),
                 transform=None, index_map=None, preload_label=True):
        super(coco_pretrain_Detection, self).__init__(root)
        self._im_shapes = {}
        self._root = root
        self._transform = transform
        self._splits = splits
        self._items = self._load_items(splits)
        self._anno_path = os.path.join('{}', './che_coef_anno/n8_xml_2_bboxwh', '{}.xml')
        self._image_path = os.path.join('{}', 'JPEGImages', '{}.jpg')
        self.index_map = index_map or dict(zip(self.classes, range(self.num_class)))
        self._label_cache = self._preload_labels() if preload_label else None

    def __str__(self):
        detail = ','.join([str(s[0]) + s[1] for s in self._splits])
        return self.__class__.__name__ + '(' + detail + ')'

    @property
    def classes(self):
        """Category names."""
        try:
            self._validate_class_names(self.CLASSES)
        except AssertionError as e:
            raise RuntimeError("Class names must not contain {}".format(e))
        return type(self).CLASSES

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        img_id = self._items[idx]
        img_path = self._image_path.format(*img_id)
        label = self._label_cache[idx] if self._label_cache else self._load_label(idx)
        img = mx.image.imread(img_path, 1)
        if self._transform is not None:
            return self._transform(img, label)
        return img, label

    def _load_items(self, splits):
        """Load individual image indices from splits."""
        ids = []
        for year, name in splits:
            root = os.path.join(self._root, 'VOC' + str(year))
            lf = os.path.join(root, 'ImageSets', 'Segmentation', name + '.txt')
            with open(lf, 'r') as f:
                ids += [(root, line.strip()) for line in f.readlines()]
        return ids

    def _load_label(self, idx):
        """Parse xml file and return labels."""
        img_id = self._items[idx]
        anno_path = self._anno_path.format(*img_id)
        root = ET.parse(anno_path).getroot()
        size = root.find('size')
        width = float(size.find('width').text)
        height = float(size.find('height').text)
        if idx not in self._im_shapes:
            # store the shapes for later usage
            self._im_shapes[idx] = (width, height)
        label = []
        for obj in root.iter('object'):
            obj_label_info = []
            difficult = int(obj.find('difficult').text)
            cls_name = obj.find('name').text.strip().lower()
            if cls_name not in self.classes:
                continue
            cls_id = self.index_map[cls_name]
            xml_box = obj.find('bndbox')
            xmin = (float(xml_box.find('xmin').text))
            ymin = (float(xml_box.find('ymin').text))
            xmax = (float(xml_box.find('xmax').text))
            ymax = (float(xml_box.find('ymax').text))
            obj_label_info.append(xmin)
            obj_label_info.append(ymin)
            obj_label_info.append(xmax)
            obj_label_info.append(ymax)
            for _ in range(50):
                obj_label_info.append(0)
            obj_label_info.append(cls_id)
            obj_label_info.append(difficult)
            obj_label_info.append(width)
            obj_label_info.append(height)
            try:
                self._validate_label(xmin, ymin, xmax, ymax, width, height)
            except AssertionError as e:
                raise RuntimeError("Invalid label at {}, {}".format(anno_path, e))
            label.append(obj_label_info)
        return np.array(label)

    def _validate_label(self, xmin, ymin, xmax, ymax, width, height):
        """Validate labels."""
        assert 0 <= xmin < width, "xmin must in [0, {}), given {}".format(width, xmin)
        assert 0 <= ymin < height, "ymin must in [0, {}), given {}".format(height, ymin)
        assert xmin < xmax <= width, "xmax must in (xmin, {}], given {}".format(width, xmax)
        assert ymin < ymax <= height, "ymax must in (ymin, {}], given {}".format(height, ymax)

    def _validate_class_names(self, class_list):
        """Validate class names."""
        assert all(c.islower() for c in class_list), "uppercase characters"
        stripped = [c for c in class_list if c.strip() != c]
        if stripped:
            warnings.warn('white space removed for {}'.format(stripped))

    def _preload_labels(self):
        """Preload all labels into memory."""
        logging.debug("Preloading %s labels into memory...", str(self))
        return [self._load_label(idx) for idx in range(len(self))]


class VOC_Val_Detection(VisionDataset):
    """Pascal VOC Val detection Dataset.
    The Val DataSet of Our model, has Polygon Points(360) label, not Coefficient Model

    Parameters
    ----------
    root : str, default '~/mxnet/datasets/voc'
        Path to folder storing the dataset.
    splits : list of tuples, default ((2007, 'trainval'), (2012, 'trainval'))
        List of combinations of (year, name)
        For years, candidates can be: 2007, 2012.
        For names, candidates can be: 'train', 'val', 'trainval', 'test'.
    transform : callable, default None
        A function that takes data and label and transforms them. Refer to
        :doc:`./transforms` for examples.

        A transform function for object detection should take label into consideration,
        because any geometric modification will require label to be modified.
    index_map : dict, default None
        In default, the 20 classes are mapped into indices from 0 to 19. We can
        customize it by providing a str to int dict specifying how to map class
        names to indices. Use by advanced users only, when you want to swap the orders
        of class labels.
    preload_label : bool, default True
        If True, then parse and load all labels into memory during
        initialization. It often accelerate speed but require more memory
        usage. Typical preloaded labels took tens of MB. You only need to disable it
        when your dataset is extremely large.
    """
    CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
               'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')

    def __init__(self, root='/home/tutian/dataset/',
                 # ori    splits=((2007, 'trainval'), (2012, 'trainval')),
                 splits=((2012, 'train')),
                 transform=None, index_map=None, preload_label=True):
        super(VOC_Val_Detection, self).__init__(root)
        self._im_shapes = {}
        self._root = root
        self._transform = transform
        self._splits = splits
        self._items = self._load_items(splits)
        self._coef_anno_path = os.path.join('{}', './bases_50_xml_each_uniform', '{}.xml')
        self._anno_path = os.path.join('{}', './label_polygon_360_xml', '{}.xml')
        self._image_path = os.path.join('{}', 'img', '{}.jpg')
        self.index_map = index_map or dict(zip(self.classes, range(self.num_class)))
        self._label_cache = self._preload_labels() if preload_label else None

    def __str__(self):
        detail = ','.join([str(s[0]) + s[1] for s in self._splits])
        return self.__class__.__name__ + '(' + detail + ')'

    @property
    def classes(self):
        """Category names."""
        try:
            self._validate_class_names(self.CLASSES)
        except AssertionError as e:
            raise RuntimeError("Class names must not contain {}".format(e))
        return type(self).CLASSES

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        img_id = self._items[idx]
        img_path = self._image_path.format(*img_id)
        label = self._label_cache[idx] if self._label_cache else self._load_label(idx)
        img = mx.image.imread(img_path, 1)
        if self._transform is not None:
            return self._transform(img, label)
        return img, label

    def _load_items(self, splits):
        """Load individual image indices from splits."""
        ids = []
        for year, name in splits:
            root = os.path.join(self._root, 'VOC' + str(year))
            lf = os.path.join(root, 'ImageSets', 'Segmentation', name + '.txt')
            with open(lf, 'r') as f:
                ids += [(root, line.strip()) for line in f.readlines()]
        return ids

    def _load_label(self, idx):
        """Parse xml file and return labels."""
        img_id = self._items[idx]
        anno_path = self._anno_path.format(*img_id)
        coef_anno_path = self._coef_anno_path.format(*img_id)
        root = ET.parse(anno_path).getroot()
        coef_root = ET.parse(coef_anno_path).getroot()
        size = root.find('size')
        width = float(size.find('width').text)
        height = float(size.find('height').text)
        if idx not in self._im_shapes:
            # store the shapes for later usage
            self._im_shapes[idx] = (width, height)
        label = []
        for obj, coef_obj in zip(root.iter('object'), coef_root.iter('object')):
            obj_label_info = []
            difficult = int(obj.find('difficult').text)
            cls_name = obj.find('name').text.strip().lower()
            if cls_name not in self.classes:
                continue
            cls_id = self.index_map[cls_name]
            xml_box = obj.find('bndbox')
            xmin = (float(xml_box.find('xmin').text))
            ymin = (float(xml_box.find('ymin').text))
            xmax = (float(xml_box.find('xmax').text))
            ymax = (float(xml_box.find('ymax').text))
            xml_points_x = obj.find('points_x').text
            xml_points_x = xml_points_x.split()
            points_x = [float(xml_points_x[i]) for i in range(len(xml_points_x))]
            xml_points_y = obj.find('points_y').text
            xml_points_y = xml_points_y.split()
            points_y = [float(xml_points_y[i]) for i in range(len(xml_points_y))]
            obj_label_info.append(xmin)
            obj_label_info.append(ymin)
            obj_label_info.append(xmax)
            obj_label_info.append(ymax)
            for i in range(len(points_x)):
                obj_label_info.append(points_x[i])
            for i in range(len(points_y)):
                obj_label_info.append(points_y[i])
            obj_label_info.append(cls_id)
            obj_label_info.append(difficult)
            obj_label_info.append(width)
            obj_label_info.append(height)

            # Added for WenQiang analysis
            obj_label_info.append(int(img_id[1]))  # + 1
            # print(img_id)
            xml_coef = coef_obj.find('coef').text
            xml_coef = xml_coef.split()
            coef = [float(xml_coef[i]) for i in range(len(xml_coef))]
            for i in range(len(coef)):  # + 50
                obj_label_info.append(coef[i])

            try:
                self._validate_label(xmin, ymin, xmax, ymax, width, height)
            except AssertionError as e:
                raise RuntimeError("Invalid label at {}, {}".format(anno_path, e))
            label.append(obj_label_info)
        return np.array(label, dtype=np.float64)

    def _validate_label(self, xmin, ymin, xmax, ymax, width, height):
        """Validate labels."""
        assert 0 <= xmin < width, "xmin must in [0, {}), given {}".format(width, xmin)
        assert 0 <= ymin < height, "ymin must in [0, {}), given {}".format(height, ymin)
        assert xmin < xmax <= width, "xmax must in (xmin, {}], given {}".format(width, xmax)
        assert ymin < ymax <= height, "ymax must in (ymin, {}], given {}".format(height, ymax)

    def _validate_class_names(self, class_list):
        """Validate class names."""
        assert all(c.islower() for c in class_list), "uppercase characters"
        stripped = [c for c in class_list if c.strip() != c]
        if stripped:
            warnings.warn('white space removed for {}'.format(stripped))

    def _preload_labels(self):
        """Preload all labels into memory."""
        logging.debug("Preloading %s labels into memory...", str(self))
        return [self._load_label(idx) for idx in range(len(self))]


class cocoDetection(VisionDataset):
    """Pascal VOC detection Dataset.
    The train Dataset of Our Model, has the Coefficients label, not Polygon Points label
    You should origanize your dataset as pascal voc format first.
    Parameters
    ----------
    root : str, default '~/mxnet/datasets/voc'
        Path to folder storing the dataset.
    splits : list of tuples, default ((2007, 'trainval'), (2012, 'trainval'))
        List of combinations of (year, name)
        For years, candidates can be: 2007, 2012.
        For names, candidates can be: 'train', 'val', 'trainval', 'test'.
    transform : callable, default None
        A function that takes data and label and transforms them. Refer to
        :doc:`./transforms` for examples.

        A transform function for object detection should take label into consideration,
        because any geometric modification will require label to be modified.
    index_map : dict, default None
        In default, the 20 classes are mapped into indices from 0 to 19. We can
        customize it by providing a str to int dict specifying how to map class
        names to indices. Use by advanced users only, when you want to swap the orders
        of class labels.
    preload_label : bool, default True
        If True, then parse and load all labels into memory during
        initialization. It often accelerate speed but require more memory
        usage. Typical preloaded labels took tens of MB. You only need to disable it
        when your dataset is extremely large.
    """
    CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
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
    
    # The cat_id in label is already for training. i.e., 1 to 80
    # There is no splits in coco
    def __init__(self, root='', transform=None, index_map=None, preload_label=True, subfolder=''):
        super(cocoDetection, self).__init__(root)
        self._im_shapes = {}
        self._root = root
        assert root
        self._transform = transform
        self._items = self._load_items()
        if subfolder == '':
            print("Subfolder not defined!")
        self._anno_path = os.path.join('{}', subfolder, '{}.xml')
        self._image_path = os.path.join('{}', 'img', '{}.jpg')
        self.index_map = index_map or dict(zip(self.classes, range(self.num_class)))
        self._label_cache = self._preload_labels() if preload_label else None

    def __str__(self):
        # detail = ','.join([str(s[0]) + s[1] for s in self._splits])
        # return self.__class__.__name__ + '(' + detail + ')'
        return self._root

    @property
    def classes(self):
        """Category names."""
        try:
            self._validate_class_names(self.CLASSES)
        except AssertionError as e:
            raise RuntimeError("Class names must not contain {}".format(e))
        return type(self).CLASSES

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        img_id = self._items[idx]
        img_path = self._image_path.format(img_id[0], img_id[1].zfill(12))
        label = self._label_cache[idx] if self._label_cache else self._load_label(idx)
        img = mx.image.imread(img_path, 1)
        if self._transform is not None:
            return self._transform(img, label)
        return img, label

    def _load_items(self):
        """Load individual image names from txt"""
        ids = []
        root = self._root
        lf = os.path.join(root, 'images_ids.txt')
        with open(lf, 'r') as f:
            ids += [(root, line.strip()) for line in f.readlines()]
        return ids

    def _load_label(self, idx):
        """Parse xml file and return labels."""
        img_id = self._items[idx]
        anno_path = self._anno_path.format(*img_id)
        root = ET.parse(anno_path).getroot()
        size = root.find('size')
        width = float(size.find('width').text)
        height = float(size.find('height').text)
        if idx not in self._im_shapes:
            # store the shapes for later usage
            self._im_shapes[idx] = (width, height)
        label = []
        for obj in root.iter('object'):
            obj_label_info = []
            difficult = int(obj.find('difficult').text)
            cls_name = obj.find('name').text.strip().lower()
            if cls_name not in self.classes:
                continue
            cls_id = self.index_map[cls_name]
            xml_box = obj.find('bndbox')
            xmin = (float(xml_box.find('xmin').text))
            ymin = (float(xml_box.find('ymin').text))
            xmax = (float(xml_box.find('xmax').text))
            ymax = (float(xml_box.find('ymax').text))
            # inst_id = (float(obj.find('inst_id').text))
            xml_coef = obj.find('coef').text.split()
            # coef = [float(xml_coef[i]) for i in range(len(xml_coef))]
            coef = [float(i) for i in xml_coef]
            obj_label_info.append(xmin) # 0
            obj_label_info.append(ymin) # 1
            obj_label_info.append(xmax) # 2
            obj_label_info.append(ymax) # 3
            # for i in range(len(coef)):
            #     obj_label_info.append(coef[i])
            obj_label_info += coef  # 4 - 50+4
            obj_label_info.append(cls_id)  # 50+4 - 50+5
            obj_label_info.append(difficult) # 50+5 - 50+6
            obj_label_info.append(width)  # 50+6 - 50+7
            obj_label_info.append(height) # 50+7 - 50+8
            obj_label_info.append(int(img_id[1]))  # 50+8 - 50+9
            # obj_label_info.append(int(inst_id)) # 50+9 - 50+10
            try:
                self._validate_label(xmin, ymin, xmax, ymax, width, height)
            except AssertionError as e:
                raise RuntimeError("Invalid label at {}, {}".format(anno_path, e))
            label.append(obj_label_info)
        return np.array(label)

    def _validate_label(self, xmin, ymin, xmax, ymax, width, height):
        """Validate labels."""
        assert 0 <= xmin < width, "xmin must in [0, {}), given {}".format(width, xmin)
        assert 0 <= ymin < height, "ymin must in [0, {}), given {}".format(height, ymin)
        assert xmin < xmax <= width, "xmax must in (xmin, {}], given {}".format(width, xmax)
        assert ymin < ymax <= height, "ymax must in (ymin, {}], given {}".format(height, ymax)

    def _validate_class_names(self, class_list):
        """Validate class names."""
        assert all(c.islower() for c in class_list), "uppercase characters"
        stripped = [c for c in class_list if c.strip() != c]
        if stripped:
            warnings.warn('white space removed for {}'.format(stripped))

    def _preload_labels(self):
        """Preload all labels into memory."""
        logging.debug("Preloading %s labels into memory...", str(self))
        return [self._load_label(idx) for idx in range(len(self))]

