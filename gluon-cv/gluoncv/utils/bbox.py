"""Calculate Intersection-Over-Union(IOU) of two bounding boxes."""
from __future__ import division

import numpy as np
import cv2 as cv
import shapely
from shapely.geometry import Polygon, MultiPoint
import numpy.polynomial.chebyshev as chebyshev
from sklearn.metrics import jaccard_score

from PIL import Image  # vis, just for debug
from time import time

x_mean = [-10372.72573, -65.62571687, 8.113603703, -4.129944152, -70.272901, 48.39311715, 13.54921555, -6.204910281, -82.93014464, -13.49012528, -1.377779259, -0.359250444, -9.937065122, 3.458047501, -0.637840469, 6.447263647, -0.159122537, -4.013595629, 0.368631004, -0.798153475, -0.675555162, -0.64375462, -1.876287186, -5.29987036, -3.081862721, -1.205230327, 1.611716191, -1.447915821, -1.008998948, 2.062282999, -0.366824452, -0.76531215, -5.657952825, -0.702878769, 4.139859116, 3.660853075, -4.365368841, -3.759996972, 0.10982376, -0.901409142, -0.914115701, -0.287375188, 0.673038067, -1.64012666, 0.983785635, 0.369574124, -0.080789953, 1.392399963, 1.066113083, -1.677959563]
sqrt_var = [2803.685003, 2161.867545, 2087.585027, 1877.30127, 1616.864336, 1459.067858, 1325.125214, 1166.808235, 1078.785935, 954.0199264, 941.2122393, 877.5677226, 822.7808391, 767.6920133, 742.2590864, 709.1605303, 707.289721, 671.4039817, 625.9553238, 595.3469511, 586.0426203, 571.3110848, 560.1314211, 535.6616464, 534.0023283, 508.0441017, 489.9292073, 485.8909739, 474.3963334, 466.0055539, 449.4863225, 444.2908041, 437.5323292, 432.4269679, 409.8391652, 406.6126991, 400.0697815, 396.4458867, 391.1772833, 384.5596921, 377.9070695, 371.8721299, 366.0097442, 350.6612788, 350.3757641, 347.7540654, 343.1649469, 335.2165795, 332.2271304, 327.4941672]
x_min = np.array([-15482.42389, -6890.276516, -5538.785075, -5816.060271, -4558.030574, -6838.889213, -4845.916472, -4227.809116, -3826.952649, -3513.548843, -4829.720262, -3443.692333, -4599.026508, -3632.84294, -2830.503294, -2743.853048, -2821.758048, -2892.371713, -2783.273599, -2328.833622, -2287.968436, -2414.126732, -2567.267731, -2730.10522, -2273.68255, -2323.956728, -1884.060547, -2001.761208, -2243.889445, -2216.714357, -2104.139639, -2422.256762, -1817.853336, -2194.244812, -1718.128955, -1841.351373, -1889.84415, -1951.221188, -1691.402656, -1864.771442, -1685.0357, -1469.929736, -1399.324855, -1524.768124, -1847.479326, -1464.607373, -1438.142198, -1467.812184, -1486.837391, -1468.122325])
x_max = np.array([0, 6975.697248, 5453.263858, 5710.089781, 8459.724464, 5007.860651, 4914.459004, 4938.654403, 3904.488484, 4878.943755, 4372.214577, 3314.96113, 3958.25269, 3224.153406, 2789.40399, 3242.741783, 2877.455503, 2805.222196, 2610.172046, 2595.920655, 2711.131864, 2382.223149, 2464.276953, 2649.856643, 2540.557773, 3242.252329, 2196.179273, 2306.631054, 1858.970692, 2254.906552, 1963.849169, 2242.1793, 1795.529975, 2562.035894, 1748.565462, 1918.170148, 1555.71773, 2739.393537, 1662.911931, 1877.164972, 1832.516479, 1596.010903, 1562.090407, 1526.694717, 1544.801876, 1441.254865, 1445.42314, 1397.396689, 1378.251306, 1448.646948])


def bbox_iou(bbox_a, bbox_b, offset=0):
    """Calculate Intersection-Over-Union(IOU) of two bounding boxes.

    Parameters
    ----------
    bbox_a : numpy.ndarray
        An ndarray with shape :math:`(N, 4)`.
    bbox_b : numpy.ndarray
        An ndarray with shape :math:`(M, 4)`.
    offset : float or int, default is 0
        The ``offset`` is used to control the whether the width(or height) is computed as
        (right - left + ``offset``).
        Note that the offset must be 0 for normalized bboxes, whose ranges are in ``[0, 1]``.

    Returns
    -------
    numpy.ndarray
        An ndarray with shape :math:`(N, M)` indicates IOU between each pairs of
        bounding boxes in `bbox_a` and `bbox_b`.

    """
    if bbox_a.shape[1] < 4 or bbox_b.shape[1] < 4:
        raise IndexError("Bounding boxes axis 1 must have at least length 4")

    tl = np.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
    br = np.minimum(bbox_a[:, None, 2:4], bbox_b[:, 2:4])

    area_i = np.prod(br - tl + offset, axis=2) * (tl < br).all(axis=2)
    area_a = np.prod(bbox_a[:, 2:4] - bbox_a[:, :2] + offset, axis=1)
    area_b = np.prod(bbox_b[:, 2:4] - bbox_b[:, :2] + offset, axis=1)
    
    return area_i / (area_a[:, None] + area_b - area_i)


def polygon_iou(polygon_as, polygon_bs, offset=0):
    """Calculate Intersection-Over-Union(IOU) of two polygons
    
    Parameters
    ----------
    polygon_as : numpy.ndarray
         An ndarray with shape :math'(N, polygon_a_nums, 2)
    polygon_bs : numpy.ndarray
         An ndarray with shape :math'(M, polygon_b_nums, 2)
    offset : float or int, default is 0
        The ``offset`` is used to control the whether the width(or height) is computed as
        (right - left + ``offset``).
        Note that the offset must be 0 for normalized bboxes, whose ranges are in ``[0, 1]``.
    
    Returns
    ------
    numpy.ndarray
        An An ndarray with shape :math:`(N, M)` indicates IOU between each pairs of
        polygons in `polygon_as` and `polygon_bs`.
    This way is not need the points_num is equal 
    """
    try:    
        N = polygon_as.shape[0]
        M = polygon_bs.shape[0]
    except:
        N = len(polygon_as)
        M = len(polygon_bs)    
    polygon_ious = np.zeros((N, M))
    for n in range(N):
        polygon_a = polygon_as[n]
        polya = Polygon(polygon_a).convex_hull
        for m in range(M):
            polygon_b = polygon_bs[m]
            polyb = Polygon(polygon_b).convex_hull
            try:
                inter_area = polya.intersection(polyb).area
                union_poly = np.concatenate((polygon_a,polygon_b))
                union_area = MultiPoint(union_poly).convex_hull.area
                if union_area == 0 or inter_area == 0:
                    iou = 0
                else:
                    iou = float(inter_area) / union_area
                polygon_ious[n][m] = iou
            except shapely.geos.TopologicalError:
                print("shapely.geos.TopologicalError occured, iou set to 0")
                polygon_ious[n][m] = 0
                continue
    return polygon_ious


def cheby(coef):
    """
    coef numpy.addary with shape (N , 2*deg+2) such as (N,18), (N,26)
    theta nuumpy.addary with shape (360,)    [-1,1]

    Return numpy.array woth shape (N,360)
    """
    theta = np.linspace(-1, 1, 360)
    coef = coef.T
    r = chebyshev.chebval(theta, coef)
    
    return r


def coef_iou(coef_as, coef_bs, bbox_as, bbox_bs, center_as, center_bs):
    """Calculate Intersection-Over-Union(IOU) of two coefs
    Parameters
    ----------
    coef_as : numpy.ndarray
         An ndarray with shape :math'(N,2*deg+2)
    coef_bs : numpy.ndarray
         An ndarray with shape :math'(M,2*deg+2)
    bbox_as : numpy.ndarray
         An ndarray with shape :math'(N,4)  x1y1x2y2
    bbox_bs : numpy.ndarray
         An ndarray with shape :math'(M,4)  x1y1x2y2
    center_as : numpy.ndarray
         An ndarray with shape :math'(N,2)  xy
    center_bs : numpy.ndarray
         An ndarray with shape :math'(M,2)  xy

    Returns
    ------
    numpy.ndarray
        An An ndarray with shape :math:`(N, M)` indicates IOU between each pairs of
        polygons in `coef_as` and coef_bs`.
    """

    polygon_as = coef_trans_polygon(coef_as, bbox_as, center_as)
    polygon_bs = coef_trans_polygon(coef_bs, bbox_bs, center_bs)
    iou = polygon_iou(polygon_as, polygon_bs)
    
    return iou


def new_mask_iou(coefs, bboxs, bases, polygon_gts):
    # Here the bbox stands for the predicted bbox
    bboxs_x1 = bboxs[:, 0].reshape(-1, 1)  # N,1
    bboxs_x2 = bboxs[:, 2].reshape(-1, 1)  # N,1
    bboxs_y1 = bboxs[:, 1].reshape(-1, 1)  # N,1
    bboxs_y2 = bboxs[:, 3].reshape(-1, 1)  # N,1
    # bboxsw = np.abs(bboxs_x2 - bboxs_x1)  # N,1
    # bboxsh = np.abs(bboxs_y2 - bboxs_y1)  # N,1
    # Tutian doesn't think it necessary to add np.abs() - May ask Haiyang

    # uniform
    # coefs = np.clip(coefs, 0, 1)  # No difference
    coefs = coefs * (x_max-x_min) + x_min

    # var
    # coefs = coefs * sqrt_var + x_mean
    # print(np.unique(coefs))
    masks = np.dot(coefs, bases)

    # masks = (masks >= ((masks.max()+masks.min())/2)).astype(np.uint8)  # the threshold
    # masks *= 255  # Can be simplified

    masks_pred = masks.reshape(-1, 64, 64)
    N = masks_pred.shape[0]
    M = polygon_gts.shape[0]

    ious = np.zeros((N, M))
    for n in range(N):
        mask_pd = masks_pred[n]  # 64 by 64
        # print(np.unique(coefs[n]))
        for m in range(M):
            x1, x2, y1, y2 = int(bboxs_x1[n]), int(bboxs_x2[n]), int(bboxs_y1[n]), int(bboxs_y2[n])
            w, h = x2 - x1, y2 - y1
            assert(w >= 0 and h >=0)
            # No original image size info. Instead, just create a board that is large enough to hold the gtbbox and pdbbox
            board_x = max(x1, x2, np.max(polygon_gts[:,0]))
            board_y = max(y1, y2, np.max(polygon_gts[:,1]))
            board_x = int(board_x) + 1  # + 1 is for safety
            board_y = int(board_y) + 1
            board_pd = np.zeros((board_y, board_x))
            board_gt = np.zeros((board_y, board_x))

            # Drawing the predicted mask to the board
            # (w, h) need to be carefully checked. 
            # resized = cv.resize(mask_pd, (w, h), interpolation = cv.INTER_NEAREST)
            resized = cv.resize(mask_pd, (w, h))

            # First resize then binarize
            resized = (resized >= ((resized.max()+resized.min())/2)).astype(np.uint8)  # the threshold
            # masks *= 255  # Can be simplified

            board_pd[y1:y2, x1:x2] = np.array(resized)

            # Drawing the gt to the board
            contour = np.expand_dims(polygon_gts[m], axis=1)
            contour = np.expand_dims(contour, axis=0)
            # print(contour.shape)
            cv.drawContours(board_gt, contour.astype(np.int32), -1, (1), -1)

            iou = jaccard_score(board_pd.flatten(), board_gt.flatten())

            ious[n][m] = iou

    return ious

def coef_polygon_iou(pred_coef_l, bases, pred_bbox_l, gt_points_xs_l, gt_points_ys_l):
    """Calculate Intersection-Over-Union(IOU) of pred coefs(Reconstructed) and gt polygon points
    Parameters
    ----------
    pred_coef_l : numpy.ndarray
         An ndarray with shape :math'(N,2*deg+2)
    pred_bbox_l : numpy.ndarray
         An ndarray with shape :math'(N,4)  x1y1x2y2
    pred_center_l : numpy.ndarray
         An ndarray with shape :math'(N,2)  xy
    gt_points_xs_l : numpy.ndarray
         An ndarray with shape :math'(M,360)  x1, x2, x3,..., x360
    gt_points_ys_l : numpy.ndarray
         An ndarray with shape :math'(M,360)  y1, y2, y3,..., y360
    
    Returns
    ------
    numpy.ndarray
        An ndarray with shape :math:`(N, M)` indicates IOU between each pairs of
        polygons in `pred coef` and gt polygon points`.
    """

    gt_points_xs_l = gt_points_xs_l.reshape(-1, 360, 1)  # M, 360 ,1
    gt_points_ys_l = gt_points_ys_l.reshape(-1, 360, 1)  # M, 360 ,1
    polygon_bs = np.concatenate((gt_points_xs_l,gt_points_ys_l), axis=-1)  # M, 360 ,2
    # mask_pred = coef_trans_mask(pred_coef_l, pred_bbox_l, bases)
    # iou = mask_iou(mask_pred, polygon_bs)

    iou = new_mask_iou(pred_coef_l, pred_bbox_l, bases, polygon_bs)
    
    return iou

def bbox_xywh_to_xyxy(xywh):
    """Convert bounding boxes from format (x, y, w, h) to (xmin, ymin, xmax, ymax)

    Parameters
    ----------
    xywh : list, tuple or numpy.ndarray
        The bbox in format (x, y, w, h).
        If numpy.ndarray is provided, we expect multiple bounding boxes with
        shape `(N, 4)`.

    Returns
    -------
    tuple or numpy.ndarray
        The converted bboxes in format (xmin, ymin, xmax, ymax).
        If input is numpy.ndarray, return is numpy.ndarray correspondingly.

    """
    if isinstance(xywh, (tuple, list)):
        if not len(xywh) == 4:
            raise IndexError(
                "Bounding boxes must have 4 elements, given {}".format(len(xywh)))
        w, h = np.maximum(xywh[2] - 1, 0), np.maximum(xywh[3] - 1, 0)
        return (xywh[0], xywh[1], xywh[0] + w, xywh[1] + h)
    elif isinstance(xywh, np.ndarray):
        if not xywh.size % 4 == 0:
            raise IndexError(
                "Bounding boxes must have n * 4 elements, given {}".format(xywh.shape))
        xyxy = np.hstack((xywh[:, :2], xywh[:, :2] + np.maximum(0, xywh[:, 2:4] - 1)))
        return xyxy
    else:
        raise TypeError(
            'Expect input xywh a list, tuple or numpy.ndarray, given {}'.format(type(xywh)))

def bbox_xyxy_to_xywh(xyxy):
    """Convert bounding boxes from format (xmin, ymin, xmax, ymax) to (x, y, w, h).

    Parameters
    ----------
    xyxy : list, tuple or numpy.ndarray
        The bbox in format (xmin, ymin, xmax, ymax).
        If numpy.ndarray is provided, we expect multiple bounding boxes with
        shape `(N, 4)`.

    Returns
    -------
    tuple or numpy.ndarray
        The converted bboxes in format (x, y, w, h).
        If input is numpy.ndarray, return is numpy.ndarray correspondingly.

    """
    if isinstance(xyxy, (tuple, list)):
        if not len(xyxy) == 4:
            raise IndexError(
                "Bounding boxes must have 4 elements, given {}".format(len(xyxy)))
        x1, y1 = xyxy[0], xyxy[1]
        w, h = xyxy[2] - x1 + 1, xyxy[3] - y1 + 1
        return (x1, y1, w, h)
    elif isinstance(xyxy, np.ndarray):
        if not xyxy.size % 4 == 0:
            raise IndexError(
                "Bounding boxes must have n * 4 elements, given {}".format(xyxy.shape))
        return np.hstack((xyxy[:, :2], xyxy[:, 2:4] - xyxy[:, :2] + 1))
    else:
        raise TypeError(
            'Expect input xywh a list, tuple or numpy.ndarray, given {}'.format(type(xyxy)))

def bbox_clip_xyxy(xyxy, width, height):
    """Clip bounding box with format (xmin, ymin, xmax, ymax) to specified boundary.

    All bounding boxes will be clipped to the new region `(0, 0, width, height)`.

    Parameters
    ----------
    xyxy : list, tuple or numpy.ndarray
        The bbox in format (xmin, ymin, xmax, ymax).
        If numpy.ndarray is provided, we expect multiple bounding boxes with
        shape `(N, 4)`.
    width : int or float
        Boundary width.
    height : int or float
        Boundary height.

    Returns
    -------
    type
        Description of returned object.

    """
    if isinstance(xyxy, (tuple, list)):
        if not len(xyxy) == 4:
            raise IndexError(
                "Bounding boxes must have 4 elements, given {}".format(len(xyxy)))
        x1 = np.minimum(width - 1, np.maximum(0, xyxy[0]))
        y1 = np.minimum(height - 1, np.maximum(0, xyxy[1]))
        x2 = np.minimum(width - 1, np.maximum(0, xyxy[2]))
        y2 = np.minimum(height - 1, np.maximum(0, xyxy[3]))
        return (x1, y1, x2, y2)
    elif isinstance(xyxy, np.ndarray):
        if not xyxy.size % 4 == 0:
            raise IndexError(
                "Bounding boxes must have n * 4 elements, given {}".format(xyxy.shape))
        x1 = np.minimum(width - 1, np.maximum(0, xyxy[:, 0]))
        y1 = np.minimum(height - 1, np.maximum(0, xyxy[:, 1]))
        x2 = np.minimum(width - 1, np.maximum(0, xyxy[:, 2]))
        y2 = np.minimum(height - 1, np.maximum(0, xyxy[:, 3]))
        return np.hstack((x1, y1, x2, y2))
    else:
        raise TypeError(
            'Expect input xywh a list, tuple or numpy.ndarray, given {}'.format(type(xyxy)))
