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

# For sbd
# x_mean = [-10372.72573, -65.62571687, 8.113603703, -4.129944152, -70.272901, 48.39311715, 13.54921555, -6.204910281, -82.93014464, -13.49012528, -1.377779259, -0.359250444, -9.937065122, 3.458047501, -0.637840469, 6.447263647, -0.159122537, -4.013595629, 0.368631004, -0.798153475, -0.675555162, -0.64375462, -1.876287186, -5.29987036, -3.081862721, -1.205230327, 1.611716191, -1.447915821, -1.008998948, 2.062282999, -0.366824452, -0.76531215, -5.657952825, -0.702878769, 4.139859116, 3.660853075, -4.365368841, -3.759996972, 0.10982376, -0.901409142, -0.914115701, -0.287375188, 0.673038067, -1.64012666, 0.983785635, 0.369574124, -0.080789953, 1.392399963, 1.066113083, -1.677959563]
# sqrt_var = [2803.685003, 2161.867545, 2087.585027, 1877.30127, 1616.864336, 1459.067858, 1325.125214, 1166.808235, 1078.785935, 954.0199264, 941.2122393, 877.5677226, 822.7808391, 767.6920133, 742.2590864, 709.1605303, 707.289721, 671.4039817, 625.9553238, 595.3469511, 586.0426203, 571.3110848, 560.1314211, 535.6616464, 534.0023283, 508.0441017, 489.9292073, 485.8909739, 474.3963334, 466.0055539, 449.4863225, 444.2908041, 437.5323292, 432.4269679, 409.8391652, 406.6126991, 400.0697815, 396.4458867, 391.1772833, 384.5596921, 377.9070695, 371.8721299, 366.0097442, 350.6612788, 350.3757641, 347.7540654, 343.1649469, 335.2165795, 332.2271304, 327.4941672]
# x_min = np.array([-15482.42389, -6890.276516, -5538.785075, -5816.060271, -4558.030574, -6838.889213, -4845.916472, -4227.809116, -3826.952649, -3513.548843, -4829.720262, -3443.692333, -4599.026508, -3632.84294, -2830.503294, -2743.853048, -2821.758048, -2892.371713, -2783.273599, -2328.833622, -2287.968436, -2414.126732, -2567.267731, -2730.10522, -2273.68255, -2323.956728, -1884.060547, -2001.761208, -2243.889445, -2216.714357, -2104.139639, -2422.256762, -1817.853336, -2194.244812, -1718.128955, -1841.351373, -1889.84415, -1951.221188, -1691.402656, -1864.771442, -1685.0357, -1469.929736, -1399.324855, -1524.768124, -1847.479326, -1464.607373, -1438.142198, -1467.812184, -1486.837391, -1468.122325])
# x_max = np.array([0, 6975.697248, 5453.263858, 5710.089781, 8459.724464, 5007.860651, 4914.459004, 4938.654403, 3904.488484, 4878.943755, 4372.214577, 3314.96113, 3958.25269, 3224.153406, 2789.40399, 3242.741783, 2877.455503, 2805.222196, 2610.172046, 2595.920655, 2711.131864, 2382.223149, 2464.276953, 2649.856643, 2540.557773, 3242.252329, 2196.179273, 2306.631054, 1858.970692, 2254.906552, 1963.849169, 2242.1793, 1795.529975, 2562.035894, 1748.565462, 1918.170148, 1555.71773, 2739.393537, 1662.911931, 1877.164972, 1832.516479, 1596.010903, 1562.090407, 1526.694717, 1544.801876, 1441.254865, 1445.42314, 1397.396689, 1378.251306, 1448.646948])

# For coco
x_min = np.array([-15408.068104448881, -6893.558054798728, -7003.406866817996, -7173.151488944284, -8880.702237736832, -5105.870172246976, -5765.5587195891485, -5024.227379613461, -5711.952435731431, -5495.081529198267, -5833.850420273756, -4434.37549020221, -5849.216285241527, -4148.2654407091895, -3569.2531463158916, -4339.357174902734, -3655.7764618342203, -3823.3819004419747, -3141.4357750292143, -4225.954414632274, -4508.907524652018, -2985.9986722598996, -3351.4766979792385, -3542.6383142662216, -3208.1730852282417, -3276.2051016720184, -2778.240479008936, -2687.1807642675817, -2864.3521512732636, -2667.346488961604, -2679.78247499033, -2778.1530493300193, -2615.297232543604, -2887.83922977382, -2814.11271273744, -2665.593586967864, -2244.208215546852, -2604.715325774133, -2555.901894909533, -3023.0542016462905, -3120.604337844805, -2276.2895359281847, -2105.2348396526972, -2107.14859953116, -4062.8254106434965, -2053.622120297776, -2197.4795855647635, -2042.3037948693445, -2467.5308906646937, -2245.5552141163903])
x_max = np.array([0.0, 6832.446298223013, 7426.165815379417, 6974.701596658017, 4716.901065835743, 8131.608870119551, 5740.872699165772, 4581.338796015798, 5217.3107185273375, 5434.597380283167, 5576.999587107373, 4287.165831371201, 4963.129599067099, 4621.02114880624, 3682.6609034386593, 4353.761120273803, 4174.824769494295, 3994.883741475415, 3283.721646183678, 3798.4092325829133, 4347.6387582645475, 3372.640698902529, 3295.0094768303293, 2926.3658864426816, 3499.712903749524, 3039.4470982219764, 2473.9809720368858, 2405.556357232199, 3184.463910105855, 2784.1799697475394, 2284.209254236527, 2625.629675147772, 2336.795159840813, 2528.887489215271, 2782.44841959135, 2342.962374129638, 2477.479578295029, 2332.187232909927, 2459.4770586568147, 2794.3178970248023, 2505.2624769384856, 2767.461569799445, 1918.2837463541125, 2050.6555855719203, 2690.2851498377295, 2887.8565628719634, 2263.3678542969415, 1798.6753995660308, 2160.58798020158, 2092.1122966365115])
x_mean = np.array([-9806.334230601844, -0.1265930578759492, -44.70213815499062, 4.1016068564528485, 34.85025642973737, -3.515908079075314, 33.660171096323424, 130.83580930988637, 0.21492417056751245, 2.8112355899964174, 18.833675030236837, 0.626437650033731, -3.2008816942056932, 0.016458105852027838, 8.394310893579835, -5.059975166016848, 0.3082644590455881, 1.5217574906226543, -0.018611740148539873, 0.7879045499805826, -0.24098315206080123, -0.8808304685364998, -0.7913288600067822, -3.8891420056181145, 6.353012221300202, -0.4225753767008447, 0.27977828714261016, 0.08870383388150666, -5.0118744432067786, 0.48268046843874046, -18.893481918065138, 0.7532384238847303, -5.311672820484189, -6.17895441522754, -0.356883920263817, -0.38091052476647386, 0.08936253734500309, 1.2569901866919777, 1.4373361126170598, -3.279811354042419, -2.068920651918281, 0.060461234684045725, 0.6868672104607721, 0.03698304732462165, -2.532655293110934, -0.1347230399250139, 1.5058533210691571, 0.09911752586840517, -0.012458458813556523, 2.3168010192166624])
sqrt_var = np.array([3178.8067937849487, 2108.5508769810863, 1994.220493314925, 1938.7995338537069, 1639.3960276470855, 1432.760749474181, 1288.8778997753777, 1173.580613711433, 1122.0012218569218, 928.4866017001266, 921.4166617204439, 856.0066864535319, 827.1107657788409, 801.1389228102764, 747.0316068891458, 738.6573541712918, 713.7510960451755, 656.5846272310993, 644.8258954808872, 606.805027464974, 596.0518795953916, 588.3050234920775, 586.3892172394093, 554.8030689406621, 543.5063777522089, 503.09736918051834, 496.38691611492146, 488.43183601616107, 487.21877068107796, 476.8424720930743, 459.66215884985763, 442.3700766285788, 435.6704169622154, 429.36612409375545, 410.33755900022, 408.4439272034049, 404.64446380132824, 394.1204334571845, 393.73104227507173, 389.3877034575619, 381.867970587664, 372.6268526542834, 358.85596109586754, 357.9271102515216, 352.67275163779277, 348.4594642007308, 344.12421096240666, 343.61001513303694, 331.4995424542531, 326.8226821028282])


def cal_iou(mask1, mask2):
    mask1 = mask1.astype(np.bool)
    mask2 = mask2.astype(np.bool)
    intersection = np.sum(mask1 * mask2)
    union = np.sum((mask1 + mask2).astype(np.bool))
    return intersection / union


def cal_iou2(mask1, mask2):
    # Does not exam the data type. Be careful.
    intersection = np.sum(mask1 * mask2)
    union = np.sum((mask1 + mask2))
    return intersection / union


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


def new_mask_iou(coefs, bboxs, bases, polygon_gts):
    # Here the bbox stands for the predicted bbox
    bboxs_x1 = bboxs[:, 0].reshape(-1, 1)  # N,1
    bboxs_x2 = bboxs[:, 2].reshape(-1, 1)  # N,1
    bboxs_y1 = bboxs[:, 1].reshape(-1, 1)  # N,1
    bboxs_y2 = bboxs[:, 3].reshape(-1, 1)  # N,1
    # bboxsw = np.abs(bboxs_x2 - bboxs_x1)  # N,1
    # bboxsh = np.abs(bboxs_y2 - bboxs_y1)  # N,1
    # Tutian doesn't think it necessary to add np.abs() - May ask Haiyang

    coefs[:,20:] = 0  # First 20 coefs

    # uniform
    # coefs = coefs * (x_max-x_min) + x_min

    # var
    coefs = coefs * sqrt_var + x_mean

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

            ious[n][m] = iou  # N is for predicted, M is for gts

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

def new_iou(coefs, bases, bboxs, gt_masks):
    """Calculate Intersection-Over-Union(IOU) of pred coefs(Reconstructed) and gt polygon points
    Parameters
    ----------
    pred_coef_l : numpy.ndarray
         An ndarray with shape :math'(N,2*deg+2)
    pred_bbox_l : numpy.ndarray
         An ndarray with shape :math'(N,4)  x1y1x2y2
    pred_center_l : numpy.ndarray
         An ndarray with shape :math'(N,2)  xy
    
    Returns
    ------
    numpy.ndarray
        An ndarray with shape :math:`(N, M)` indicates IOU between each pairs of
        polygons in `pred coef` and gt polygon points`.
    """

    # Here the bbox stands for the predicted bbox
    bboxs_x1 = bboxs[:, 0].reshape(-1, 1)  # N,1
    bboxs_x2 = bboxs[:, 2].reshape(-1, 1)  # N,1
    bboxs_y1 = bboxs[:, 1].reshape(-1, 1)  # N,1
    bboxs_y2 = bboxs[:, 3].reshape(-1, 1)  # N,1

    # coefs[:,20:] = 0  # First 20 coefs

    coefs = coefs * (x_max-x_min) + x_min  # uniform
    # coefs = coefs * sqrt_var + x_mean      # var

    masks = np.dot(coefs, bases)
    masks_pred = masks.reshape((-1, 64, 64))
    N = masks_pred.shape[0]
    M = gt_masks.shape[0]

    ious = np.zeros((N, M))
    for n in range(N):
        mask_pd = masks_pred[n]  # 64 by 64
        for m in range(M):
            x1, x2, y1, y2 = int(bboxs_x1[n]), int(bboxs_x2[n]), int(bboxs_y1[n]), int(bboxs_y2[n])
            w, h = x2 - x1, y2 - y1
            assert(w >= 0 and h >=0 and x2 <= 416 and y2 <= 416)
            board_pd = np.zeros((416, 416))

            # First resize then binarize
            resized = cv.resize(mask_pd, (w, h))
            resized = (resized >= ((resized.max()+resized.min())/2))  # the threshold
            board_pd[y1:y2, x1:x2] = np.array(resized)

            iou = cal_iou(board_pd.flatten(), gt_masks[m].flatten())

            ious[n][m] = iou  # N is for predicted, M is for gts

    return ious

