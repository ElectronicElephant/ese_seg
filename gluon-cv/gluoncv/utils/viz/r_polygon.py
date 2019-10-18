"""Bounding box visualization functions."""
from __future__ import absolute_import, division

import random
import mxnet as mx
from .image import plot_image
import numpy as np
import cv2 as cv
import numpy.polynomial.chebyshev as chebyshev
from matplotlib import pyplot as plt


x_mean = [-10372.72573, -65.62571687, 8.113603703, -4.129944152, -70.272901, 48.39311715, 13.54921555, -6.204910281, -82.93014464, -13.49012528, -1.377779259, -0.359250444, -9.937065122, 3.458047501, -0.637840469, 6.447263647, -0.159122537, -4.013595629, 0.368631004, -0.798153475, -0.675555162, -0.64375462, -1.876287186, -5.29987036, -3.081862721, -1.205230327, 1.611716191, -1.447915821, -1.008998948, 2.062282999, -0.366824452, -0.76531215, -5.657952825, -0.702878769, 4.139859116, 3.660853075, -4.365368841, -3.759996972, 0.10982376, -0.901409142, -0.914115701, -0.287375188, 0.673038067, -1.64012666, 0.983785635, 0.369574124, -0.080789953, 1.392399963, 1.066113083, -1.677959563]
sqrt_var = [2803.685003, 2161.867545, 2087.585027, 1877.30127, 1616.864336, 1459.067858, 1325.125214, 1166.808235, 1078.785935, 954.0199264, 941.2122393, 877.5677226, 822.7808391, 767.6920133, 742.2590864, 709.1605303, 707.289721, 671.4039817, 625.9553238, 595.3469511, 586.0426203, 571.3110848, 560.1314211, 535.6616464, 534.0023283, 508.0441017, 489.9292073, 485.8909739, 474.3963334, 466.0055539, 449.4863225, 444.2908041, 437.5323292, 432.4269679, 409.8391652, 406.6126991, 400.0697815, 396.4458867, 391.1772833, 384.5596921, 377.9070695, 371.8721299, 366.0097442, 350.6612788, 350.3757641, 347.7540654, 343.1649469, 335.2165795, 332.2271304, 327.4941672]
x_min = np.array([-15482.42389, -6890.276516, -5538.785075, -5816.060271, -4558.030574, -6838.889213, -4845.916472, -4227.809116, -3826.952649, -3513.548843, -4829.720262, -3443.692333, -4599.026508, -3632.84294, -2830.503294, -2743.853048, -2821.758048, -2892.371713, -2783.273599, -2328.833622, -2287.968436, -2414.126732, -2567.267731, -2730.10522, -2273.68255, -2323.956728, -1884.060547, -2001.761208, -2243.889445, -2216.714357, -2104.139639, -2422.256762, -1817.853336, -2194.244812, -1718.128955, -1841.351373, -1889.84415, -1951.221188, -1691.402656, -1864.771442, -1685.0357, -1469.929736, -1399.324855, -1524.768124, -1847.479326, -1464.607373, -1438.142198, -1467.812184, -1486.837391, -1468.122325])
x_max = np.array([0, 6975.697248, 5453.263858, 5710.089781, 8459.724464, 5007.860651, 4914.459004, 4938.654403, 3904.488484, 4878.943755, 4372.214577, 3314.96113, 3958.25269, 3224.153406, 2789.40399, 3242.741783, 2877.455503, 2805.222196, 2610.172046, 2595.920655, 2711.131864, 2382.223149, 2464.276953, 2649.856643, 2540.557773, 3242.252329, 2196.179273, 2306.631054, 1858.970692, 2254.906552, 1963.849169, 2242.1793, 1795.529975, 2562.035894, 1748.565462, 1918.170148, 1555.71773, 2739.393537, 1662.911931, 1877.164972, 1832.516479, 1596.010903, 1562.090407, 1526.694717, 1544.801876, 1441.254865, 1445.42314, 1397.396689, 1378.251306, 1448.646948])


def cheby(coef):
    """
    coef numpy.array with shape (N , 2*deg+2) such as (N,18), (N,26)
    theta nuumpy.array with shape (360,)    [-1,1]

    Return numpy.array object shape with r (N,360)
    """
    theta = np.linspace(-1, 1, 360)
    coef = coef.T
    r = chebyshev.chebval(theta, coef)

    return r

def plot_r_polygon(img, bboxes, coefs, img_w, img_h, scores=None, labels=None, thresh=0.5,
              class_names=None, colors=None, ax=None,
              reverse_rgb=False, absolute_coordinates=True, num_bases = 50):
    """Visualize bounding boxes and Object Mask ( Object shape ).

    Parameters
    ----------
    img : numpy.ndarray or mxnet.nd.NDArray
        Image with shape `H, W, 3`.
    bboxes : numpy.ndarray or mxnet.nd.NDArray
        Bounding boxes with shape `N, 4`. Where `N` is the number of boxes.
    abpoints : numpy.ndarray or mxnet.nd.NDarray
        shape N,2
    coef : shape N , 2*deg+2
    scores : numpy.ndarray or mxnet.nd.NDArray, optional
        Confidence scores of the provided `bboxes` with shape `N`.
    labels : numpy.ndarray or mxnet.nd.NDArray, optional
        Class labels of the provided `bboxes` with shape `N`.
    thresh : float, optional, default 0.5
        Display threshold if `scores` is provided. Scores with less than `thresh`
        will be ignored in display, this is visually more elegant if you have
        a large number of bounding boxes with very small scores.
    class_names : list of str, optional
        Description of parameter `class_names`.
    colors : dict, optional
        You can provide desired colors as {0: (255, 0, 0), 1:(0, 255, 0), ...}, otherwise
        random colors will be substituted.
    ax : matplotlib axes, optional
        You can reuse previous axes if provided.
    reverse_rgb : bool, optional
        Reverse RGB<->BGR orders if `True`.
    absolute_coordinates : bool
        If `True`, absolute coordinates will be considered, otherwise coordinates
        are interpreted as in range(0, 1).

    Returns
    -------
    matplotlib axes
        The ploted axes.

    """

    if labels is not None and not len(bboxes) == len(labels):
        raise ValueError('The length of labels and bboxes mismatch, {} vs {}'
                         .format(len(labels), len(bboxes)))
    if scores is not None and not len(bboxes) == len(scores):
        raise ValueError('The length of scores and bboxes mismatch, {} vs {}'
                         .format(len(scores), len(bboxes)))

    ax = plot_image(img, ax=ax, reverse_rgb=reverse_rgb)

    if len(bboxes) < 1:
        return ax

    if isinstance(bboxes, mx.nd.NDArray):
        bboxes = bboxes.asnumpy()
    if isinstance(coefs, mx.nd.NDArray):
        coefs = coefs.asnumpy()
    if isinstance(img_w, mx.nd.NDArray):
        img_w = img_w.asnumpy()
    if isinstance(img_h, mx.nd.NDArray):
        img_h = img_h.asnumpy()
    if isinstance(labels, mx.nd.NDArray):
        labels = labels.asnumpy()
    if isinstance(scores, mx.nd.NDArray):
        scores = scores.asnumpy()

    if not absolute_coordinates:
        # convert to absolute coordinates using image shape
        height = img.shape[0]
        width = img.shape[1]
        bboxes[:, (0, 2)] *= width
        bboxes[:, (1, 3)] *= height

    # use random colors if None is provided
    if colors is None:
        colors = dict()

    bases = np.load('/disk1/home/tutian/ese_seg/sbd/all_50_1.npy')
    # coefs = np.tanh(coefs)
    # masks = np.zeros((img_h, img_w))

    for i, bbox in enumerate(bboxes):
        if scores is not None and scores.flat[i] < thresh:
            continue
        if labels is not None and labels.flat[i] < 0:
            continue
        cls_id = int(labels.flat[i]) if labels is not None else -1
        if cls_id not in colors:
            if class_names is not None:
                colors[cls_id] = plt.get_cmap('hsv')(cls_id / len(class_names))
            else:
                colors[cls_id] = (random.random(), random.random(), random.random())
        xmin, ymin, xmax, ymax = [int(x) for x in bbox]
        rect = plt.Rectangle((xmin, ymin), xmax - xmin,
                             ymax - ymin, fill=False,
                             edgecolor=colors[cls_id],
                             linewidth=3.5)
        ax.add_patch(rect)
        if class_names is not None and cls_id < len(class_names):
            class_name = class_names[cls_id]
        else:
            class_name = str(cls_id) if cls_id >= 0 else ''
        score = '{:.3f}'.format(scores.flat[i]) if scores is not None else ''
        if class_name or score:
            ax.text(xmin, ymin - 2,
                    '{:s} {:s}'.format(class_name, score),
                    bbox=dict(facecolor=colors[cls_id], alpha=0.5),
                    fontsize=12, color='white')
        # demo
        # coef = coefs[i].reshape(1,num_bases)
        # if coefs[i][0] >= 0.999 or coefs[i][0] <= -0.999:
        #     coefs[i][0] *= 2

        # coeffs = (coeffs - x_mean) / sqrt_var
        coefs_single = coefs[i] * sqrt_var + x_mean
        # coeffs = (coeffs - x_min) / (x_max - x_min)
        # coefs_single = coefs[i] * (x_max-x_min) + x_min

        mask_single = np.dot(coefs_single, bases)
        mask_single = np.reshape(mask_single, (64, 64))
        theta = (mask_single.max() + mask_single.min()) / 2
        mask_single = (mask_single > theta) * 255

        # r_all = cheby(coef)  # (1,360)
        bboxw = xmax - xmin
        bboxh = ymax - ymin 
        
        board = np.zeros((max(ymax, img_h), max(xmax, img_w), 4))
        resized = cv.resize(mask_single, (bboxw, bboxh), interpolation = cv.INTER_NEAREST)
        # print(board.shape, xmin, xmax, ymin, ymax, bboxw, bboxh, np.array(resized).shape)
        if (ymin<0):
            resized = resized[-ymin:,:]
            ymin = 0
        if (xmin<0):
            resized = resized[:, -xmin:]
            xmin = 0
        for i in range(4):
            board[ymin:ymax,xmin:xmax, i] = resized*colors[cls_id][i] / 255
        # board = np.dot(np.array(colors[cls_id])[:3], board)

        ax.imshow(board, alpha=0.5)

        # r_all_real = r_all * np.sqrt(bboxw*bboxw+bboxh*bboxh)
        # r_all_real = r_all_real.astype(np.float32).reshape(360,)
        # theta_list = np.arange(359 , -1 ,-1)
        # theta_list = theta_list.astype(np.float32)
        # x, y = cv.polarToCart(r_all_real, theta_list, angleInDegrees=True)
        # x = x + float(abpoints[i][0])
        # y = y + float(abpoints[i][1])
        # x = np.clip(x, xmin, xmax)
        # y = np.clip(y, ymin, ymax)
        # polygon = [[int(x[j]), int(y[j])] for j in range(360)]
        # polygon = np.array(polygon).reshape((360, 2))

        # pgon = plt.Polygon(polygon, fill=False,
        #                      edgecolor=colors[cls_id],
        #                      linewidth=3.5)
        # ax.add_patch(pgon)
    return ax
