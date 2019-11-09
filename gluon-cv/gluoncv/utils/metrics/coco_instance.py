"""MS COCO Instance Segmentation Evaluate Metrics."""
from __future__ import absolute_import

import sys
import io
import os
from os import path as osp
import warnings
import numpy as np
import mxnet as mx
from ...data.mscoco.utils import try_import_pycocotools

import cv2 as cv

x_min = np.array([-15408.068104448881, -6893.558054798728, -7003.406866817996, -7173.151488944284, -8880.702237736832, -5105.870172246976, -5765.5587195891485, -5024.227379613461, -5711.952435731431, -5495.081529198267, -5833.850420273756, -4434.37549020221, -5849.216285241527, -4148.2654407091895, -3569.2531463158916, -4339.357174902734, -3655.7764618342203, -3823.3819004419747, -3141.4357750292143, -4225.954414632274, -4508.907524652018, -2985.9986722598996, -3351.4766979792385, -3542.6383142662216, -3208.1730852282417, -3276.2051016720184, -2778.240479008936, -2687.1807642675817, -2864.3521512732636, -2667.346488961604, -2679.78247499033, -2778.1530493300193, -2615.297232543604, -2887.83922977382, -2814.11271273744, -2665.593586967864, -2244.208215546852, -2604.715325774133, -2555.901894909533, -3023.0542016462905, -3120.604337844805, -2276.2895359281847, -2105.2348396526972, -2107.14859953116, -4062.8254106434965, -2053.622120297776, -2197.4795855647635, -2042.3037948693445, -2467.5308906646937, -2245.5552141163903])
x_max = np.array([0.0, 6832.446298223013, 7426.165815379417, 6974.701596658017, 4716.901065835743, 8131.608870119551, 5740.872699165772, 4581.338796015798, 5217.3107185273375, 5434.597380283167, 5576.999587107373, 4287.165831371201, 4963.129599067099, 4621.02114880624, 3682.6609034386593, 4353.761120273803, 4174.824769494295, 3994.883741475415, 3283.721646183678, 3798.4092325829133, 4347.6387582645475, 3372.640698902529, 3295.0094768303293, 2926.3658864426816, 3499.712903749524, 3039.4470982219764, 2473.9809720368858, 2405.556357232199, 3184.463910105855, 2784.1799697475394, 2284.209254236527, 2625.629675147772, 2336.795159840813, 2528.887489215271, 2782.44841959135, 2342.962374129638, 2477.479578295029, 2332.187232909927, 2459.4770586568147, 2794.3178970248023, 2505.2624769384856, 2767.461569799445, 1918.2837463541125, 2050.6555855719203, 2690.2851498377295, 2887.8565628719634, 2263.3678542969415, 1798.6753995660308, 2160.58798020158, 2092.1122966365115])
x_mean = np.array([-9806.334230601844, -0.1265930578759492, -44.70213815499062, 4.1016068564528485, 34.85025642973737, -3.515908079075314, 33.660171096323424, 130.83580930988637, 0.21492417056751245, 2.8112355899964174, 18.833675030236837, 0.626437650033731, -3.2008816942056932, 0.016458105852027838, 8.394310893579835, -5.059975166016848, 0.3082644590455881, 1.5217574906226543, -0.018611740148539873, 0.7879045499805826, -0.24098315206080123, -0.8808304685364998, -0.7913288600067822, -3.8891420056181145, 6.353012221300202, -0.4225753767008447, 0.27977828714261016, 0.08870383388150666, -5.0118744432067786, 0.48268046843874046, -18.893481918065138, 0.7532384238847303, -5.311672820484189, -6.17895441522754, -0.356883920263817, -0.38091052476647386, 0.08936253734500309, 1.2569901866919777, 1.4373361126170598, -3.279811354042419, -2.068920651918281, 0.060461234684045725, 0.6868672104607721, 0.03698304732462165, -2.532655293110934, -0.1347230399250139, 1.5058533210691571, 0.09911752586840517, -0.012458458813556523, 2.3168010192166624])
sqrt_var = np.array([3178.8067937849487, 2108.5508769810863, 1994.220493314925, 1938.7995338537069, 1639.3960276470855, 1432.760749474181, 1288.8778997753777, 1173.580613711433, 1122.0012218569218, 928.4866017001266, 921.4166617204439, 856.0066864535319, 827.1107657788409, 801.1389228102764, 747.0316068891458, 738.6573541712918, 713.7510960451755, 656.5846272310993, 644.8258954808872, 606.805027464974, 596.0518795953916, 588.3050234920775, 586.3892172394093, 554.8030689406621, 543.5063777522089, 503.09736918051834, 496.38691611492146, 488.43183601616107, 487.21877068107796, 476.8424720930743, 459.66215884985763, 442.3700766285788, 435.6704169622154, 429.36612409375545, 410.33755900022, 408.4439272034049, 404.64446380132824, 394.1204334571845, 393.73104227507173, 389.3877034575619, 381.867970587664, 372.6268526542834, 358.85596109586754, 357.9271102515216, 352.67275163779277, 348.4594642007308, 344.12421096240666, 343.61001513303694, 331.4995424542531, 326.8226821028282])

class COCOInstanceMetric(mx.metric.EvalMetric):
    """Instance segmentation metric for COCO bbox and segm task.
    Will return box summary, box metric, seg summary and seg metric.

    Parameters
    ----------
    dataset : instance of gluoncv.data.COCOInstance
        The validation dataset.
    save_prefix : str
        Prefix for the saved JSON results.
    use_time : bool
        Append unique datetime string to created JSON file name if ``True``.
    cleanup : bool
        Remove created JSON file if ``True``.
    score_thresh : float
        Detection results with confident scores smaller than ``score_thresh`` will
        be discarded before saving to results.

    """
    def __init__(self, dataset, save_prefix, use_time=True, cleanup=False, score_thresh=1e-3,
                 method='' , bases_path='/home/tutian/dataset/coco_to_voc/coco_all_50_1.npy'):
        super(COCOInstanceMetric, self).__init__('COCOInstance')
        self.dataset = dataset
        self._img_ids = sorted(dataset.coco.getImgIds())
        # print(self._img_ids)
        self._current_id = 0
        self._cleanup = cleanup
        self._results = []
        self._score_thresh = score_thresh
        
        assert(method in ['var', 'uniform'])
        print(f"COCOInstanceMetric is loading {bases_path}")
        print(f"Method: {method}")
        self._method = method
        self._bases = np.load(bases_path)

        try_import_pycocotools()
        import pycocotools.mask as cocomask
        self._cocomask = cocomask

        if use_time:
            import datetime
            t = datetime.datetime.now().strftime('_%Y_%m_%d_%H_%M_%S')
        else:
            t = ''
        self._filename = osp.abspath(osp.expanduser(save_prefix) + t + '.json')
        try:
            f = open(self._filename, 'w')
        except IOError as e:
            raise RuntimeError("Unable to open json file to dump. What(): {}".format(str(e)))
        else:
            f.close()

    def __del__(self):
        if self._cleanup:
            try:
                os.remove(self._filename)
            except IOError as err:
                warnings.warn(str(err))

    def reset(self):
        self._current_id = 0
        self._results = []

    def _dump_json(self):
        """Write coco json file"""
        if not self._current_id == len(self._img_ids):
            warnings.warn(
                'Recorded {} out of {} validation images, incomplete results'.format(
                    self._current_id, len(self._img_ids)))
        import json
        try:
            with open(self._filename, 'w') as f:
                json.dump(self._results, f)
        except IOError as e:
            raise RuntimeError("Unable to dump json file, ignored. What(): {}".format(str(e)))

    def _get_ap(self, coco_eval):
        """Return the default AP from coco_eval."""
        # Metric printing adapted from detectron/json_dataset_evaluator.
        def _get_thr_ind(coco_eval, thr):
            ind = np.where((coco_eval.params.iouThrs > thr - 1e-5) &
                           (coco_eval.params.iouThrs < thr + 1e-5))[0][0]
            iou_thr = coco_eval.params.iouThrs[ind]
            assert np.isclose(iou_thr, thr)
            return ind

        IoU_lo_thresh = 0.5
        IoU_hi_thresh = 0.95
        ind_lo = _get_thr_ind(coco_eval, IoU_lo_thresh)
        ind_hi = _get_thr_ind(coco_eval, IoU_hi_thresh)
        # precision has dims (iou, recall, cls, area range, max dets)
        # area range index 0: all area ranges
        # max dets index 2: 100 per image
        precision = coco_eval.eval['precision'][ind_lo:(ind_hi + 1), :, :, 0, 2]
        ap_default = np.mean(precision[precision > -1])
        return ap_default

    def _update(self, annType='bbox'):
        """Use coco to get real scores. """
        pred = self.dataset.coco.loadRes(self._filename)
        gt = self.dataset.coco
        # lazy import pycocotools
        try_import_pycocotools()
        from pycocotools.cocoeval import COCOeval
        coco_eval = COCOeval(gt, pred, annType)
        coco_eval.evaluate()
        coco_eval.accumulate()
        names, values = [], []
        names.append('~~~~ Summary {} metrics ~~~~\n'.format(annType))
        # catch coco print string, don't want directly print here
        _stdout = sys.stdout
        sys.stdout = io.StringIO()
        coco_eval.summarize()
        coco_summary = sys.stdout.getvalue()
        sys.stdout = _stdout
        values.append(str(coco_summary).strip())
        names.append('~~~~ Mean AP for {} ~~~~\n'.format(annType))
        values.append('{:.1f}'.format(100 * self._get_ap(coco_eval)))
        return names, values

    def get(self):
        """Get evaluation metrics. """
        self._dump_json()
        bbox_names, bbox_values = self._update('bbox')
        mask_names, mask_values = self._update('segm')
        names = bbox_names + mask_names
        values = bbox_values + mask_values
        return names, values

    def _encode_mask(self, mask):
        """Convert mask to coco rle"""
        rle = self._cocomask.encode(np.array(mask[:, :, np.newaxis], order='F'))[0]
        rle['counts'] = rle['counts'].decode('ascii')
        return rle

    # pylint: disable=arguments-differ, unused-argument
    def update(self, pred_bboxes, pred_labels, pred_scores, pred_coefs, im_height, im_width, *args, **kwargs):
        """Update internal buffer with latest predictions.
        Note that the statistics are not available until you call self.get() to return
        the metrics.

        Parameters
        ----------
        pred_bboxes : mxnet.NDArray or numpy.ndarray
            Prediction bounding boxes with shape `B, N, 4`.
            Where B is the size of mini-batch, N is the number of bboxes.
        pred_labels : mxnet.NDArray or numpy.ndarray
            Prediction bounding boxes labels with shape `B, N`.
        pred_scores : mxnet.NDArray or numpy.ndarray
            Prediction bounding boxes scores with shape `B, N`.
        pred_masks: mxnet.NDArray or numpy.ndarray
            Prediction masks with *original* shape `H, W`.

        """
        def as_numpy(a):
            """Convert a (list of) mx.NDArray into numpy.ndarray"""
            if isinstance(a, mx.nd.NDArray):
                a = a.asnumpy()
            return a

        # mask must be the same as image shape, so no batch dimension is supported
        pred_bbox, pred_label, pred_score, pred_coef = [
            as_numpy(x) for x in [pred_bboxes, pred_labels, pred_scores, pred_coefs]]
        # filter out padded detection & low confidence detections
        valid_pred = np.where((pred_label >= 0) & (pred_score >= self._score_thresh))[0]
        pred_bbox = pred_bbox[valid_pred].astype('float32')
        pred_label = pred_label.flat[valid_pred].astype('int32')
        pred_score = pred_score.flat[valid_pred].astype('float32')
        pred_coef = pred_coef[valid_pred].astype('float32')

        imgid = self._img_ids[self._current_id]
        self._current_id += 1
        # print(imgid)
        # for each bbox detection in each image
        for bbox, label, score, coef in zip(pred_bbox, pred_label, pred_score, pred_coef):
            if label not in self.dataset.contiguous_id_to_json:
                # ignore non-exist class
                continue
            if score < self._score_thresh:
                continue
            category_id = self.dataset.contiguous_id_to_json[label]

            # Reconstruct the mask
            x1, x2, y1, y2 = int(bbox[0]), int(bbox[2]), int(bbox[1]), int(bbox[3])
            w, h = x2 - x1, y2 - y1
            if(w==0 or h==0):
                print(f'w {w} h {h}')
                continue
            if(self._method == 'var'):
                coef = coef * sqrt_var + x_mean
            else:  # uniform
                coef = coef * (x_max - x_min) + x_min
            pred_mask = np.dot(coef, self._bases).reshape((64, 64))
            resized = cv.resize(pred_mask, (w, h))
            resized = (resized >= ((pred_mask.max() + pred_mask.min())/2))  # bool
            mask = np.zeros((im_height, im_width), dtype=np.uint8)  # Has to be uint8 instead of bool
            mask[y1:y2, x1:x2] = np.array(resized)

            # convert [xmin, ymin, xmax, ymax]  to [xmin, ymin, w, h]
            bbox[2:4] -= bbox[:2]
            # coco format full image mask to rle
            rle = self._encode_mask(mask.astype(np.uint8))
            self._results.append({'image_id': imgid,
                                  'category_id': category_id,
                                  'bbox': list(map(lambda x: float(round(x, 2)), bbox[:4])),
                                  'score': float(round(score, 3)),
                                  'segmentation': rle})


class COCOInstanceMetric_BAK(mx.metric.EvalMetric):
    """Instance segmentation metric for COCO bbox and segm task.
    Will return box summary, box metric, seg summary and seg metric.

    Parameters
    ----------
    dataset : instance of gluoncv.data.COCOInstance
        The validation dataset.
    save_prefix : str
        Prefix for the saved JSON results.
    use_time : bool
        Append unique datetime string to created JSON file name if ``True``.
    cleanup : bool
        Remove created JSON file if ``True``.
    score_thresh : float
        Detection results with confident scores smaller than ``score_thresh`` will
        be discarded before saving to results.

    """
    def __init__(self, dataset, save_prefix, use_time=True, cleanup=False, score_thresh=1e-3, 
                 method='' , bases_path='/home/tutian/dataset/coco_to_voc/coco_all_50_1.npy'):
        super(COCOInstanceMetric, self).__init__('COCOInstance')
        self.dataset = dataset
        self._img_ids = sorted(dataset.coco.getImgIds())
        self._current_id = 0
        self._cleanup = cleanup
        self._results = []
        self._score_thresh = score_thresh
        assert(method in ['var', 'uniform'])
        print(f"COCOInstanceMetric is loading {bases_path}")
        print(f"Method: {method}")
        self._method = method
        self._bases = np.load(bases_path)

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
        self._train_to_anno_id = {j : i for i, j in COCO_LABEL_MAP.items()}

        try_import_pycocotools()
        import pycocotools.mask as cocomask
        self._cocomask = cocomask

        if use_time:
            import datetime
            t = datetime.datetime.now().strftime('_%Y_%m_%d_%H_%M_%S')
        else:
            t = ''
        self._filename = osp.abspath(osp.expanduser(save_prefix) + t + '.json')
        try:
            f = open(self._filename, 'w')
        except IOError as e:
            raise RuntimeError("Unable to open json file to dump. What(): {}".format(str(e)))
        else:
            f.close()

    def __del__(self):
        if self._cleanup:
            try:
                os.remove(self._filename)
            except IOError as err:
                warnings.warn(str(err))

    def reset(self):
        self._current_id = 0
        self._results = []

    def _dump_json(self):
        """Write coco json file"""
        if not self._current_id == len(self._img_ids):
            warnings.warn(
                'Recorded {} out of {} validation images, incomplete results'.format(
                    self._current_id, len(self._img_ids)))
        import json
        try:
            with open(self._filename, 'w') as f:
                json.dump(self._results, f)
        except IOError as e:
            raise RuntimeError("Unable to dump json file, ignored. What(): {}".format(str(e)))

    def _get_ap(self, coco_eval):
        """Return the default AP from coco_eval."""
        # Metric printing adapted from detectron/json_dataset_evaluator.
        def _get_thr_ind(coco_eval, thr):
            ind = np.where((coco_eval.params.iouThrs > thr - 1e-5) &
                           (coco_eval.params.iouThrs < thr + 1e-5))[0][0]
            iou_thr = coco_eval.params.iouThrs[ind]
            assert np.isclose(iou_thr, thr)
            return ind

        IoU_lo_thresh = 0.5
        IoU_hi_thresh = 0.95
        ind_lo = _get_thr_ind(coco_eval, IoU_lo_thresh)
        ind_hi = _get_thr_ind(coco_eval, IoU_hi_thresh)
        # precision has dims (iou, recall, cls, area range, max dets)
        # area range index 0: all area ranges
        # max dets index 2: 100 per image
        precision = coco_eval.eval['precision'][ind_lo:(ind_hi + 1), :, :, 0, 2]
        ap_default = np.mean(precision[precision > -1])
        return ap_default

    def _update(self, annType='bbox'):
        """Use coco to get real scores. """
        pred = self.dataset.coco.loadRes(self._filename)
        gt = self.dataset.coco
        # lazy import pycocotools
        try_import_pycocotools()
        from pycocotools.cocoeval import COCOeval
        coco_eval = COCOeval(gt, pred, annType)
        coco_eval.evaluate()
        coco_eval.accumulate()
        names, values = [], []
        names.append('~~~~ Summary {} metrics ~~~~\n'.format(annType))
        # catch coco print string, don't want directly print here
        _stdout = sys.stdout
        sys.stdout = io.StringIO()
        coco_eval.summarize()
        coco_summary = sys.stdout.getvalue()
        sys.stdout = _stdout
        values.append(str(coco_summary).strip())
        names.append('~~~~ Mean AP for {} ~~~~\n'.format(annType))
        values.append('{:.1f}'.format(100 * self._get_ap(coco_eval)))
        return names, values

    def get(self):
        """Get evaluation metrics. """
        self._dump_json()
        bbox_names, bbox_values = self._update('bbox')
        mask_names, mask_values = self._update('segm')
        names = bbox_names + mask_names
        values = bbox_values + mask_values
        return names, values

    def _encode_mask(self, mask):
        """Convert mask to coco rle"""
        rle = self._cocomask.encode(np.array(mask[:, :, np.newaxis], order='F'))[0]
        rle['counts'] = rle['counts'].decode('ascii')
        return rle

    # pylint: disable=arguments-differ, unused-argument
    def update(self, pred_bboxes, pred_labels, pred_scores, pred_coefs, imgids, widths, heights, *args, **kwargs):
        """Update internal buffer with latest predictions.
        Note that the statistics are not available until you call self.get() to return
        the metrics.

        Parameters
        ----------
        pred_bboxes : mxnet.NDArray or numpy.ndarray
            Prediction bounding boxes with shape `B, N, 4`.
            Where B is the size of mini-batch, N is the number of bboxes.
        pred_labels : mxnet.NDArray or numpy.ndarray
            Prediction bounding boxes labels with shape `B, N`.
        pred_scores : mxnet.NDArray or numpy.ndarray
            Prediction bounding boxes scores with shape `B, N`.
        pred_masks: mxnet.NDArray or numpy.ndarray
            Prediction masks with *original* shape `H, W`.

        """
        def as_numpy(a):
            """Convert a (list of) mx.NDArray into numpy.ndarray"""
            if isinstance(a, (list, tuple)):
                out = [x.asnumpy() if isinstance(x, mx.nd.NDArray) else x for x in a]
                try:
                    out = np.concatenate(out, axis=0)
                except ValueError:
                    out = np.array(out)
                return out
            elif isinstance(a, mx.nd.NDArray):
                a = a.asnumpy()
            return a

        for pred_bbox, pred_coef, pred_label, pred_score, gt_width, gt_height, img_id  in zip(
                *[as_numpy(x) for x in [pred_bboxes, pred_coefs, pred_labels, pred_scores,
                                        widths, heights, imgids]]):
            # valid_pred = np.where(pred_label.flat >= 0)[0]
            valid_pred = np.where(pred_label.flat >= 0 & (pred_score.flat >= self._score_thresh))[0]
            pred_bbox = pred_bbox[valid_pred, :]
            # pred_center = pred_center[valid_pred, :]
            pred_coef = pred_coef[valid_pred, :]
            pred_label = pred_label.flat[valid_pred].astype(int)
            pred_score = pred_score.flat[valid_pred]

            valid_gt = np.where(gt_width.flat >= 0)[0]
            img_id = int(np.unique(img_id[valid_gt, :]))
            gt_width = int(np.unique(gt_width[valid_gt, :]))
            gt_height = int(np.unique(gt_height[valid_gt, :]))

            # Construct the predicted masks
            assert(len(pred_bbox)==len(pred_coef))
            for i in range(len(pred_bbox)):
                x1, x2, y1, y2 = pred_bbox[i][0], pred_bbox[i][2], pred_bbox[i][1], pred_bbox[i][3]
                x1 = int(x1 * gt_width / 416.0)
                x2 = int(x2 * gt_width / 416.0)
                y1 = int(y1 * gt_height / 416.0)
                y2 = int(y2 * gt_height / 416.0)
                w, h = x2 - x1, y2 - y1

                if(self._method == 'var'):
                    coef = pred_coef[i] * sqrt_var + x_mean
                else:  # uniform
                    coef = pred_coef[i] * (x_max - x_min) + x_min
                
                pred_mask = np.dot(coef, self._bases).reshape((64, 64))
                resized = cv.resize(pred_mask, (w, h))
                resized = (resized >= ((pred_mask.max() + pred_mask.min())/2))  # bool

                pred_mask = np.zeros((gt_height, gt_width), dtype=np.bool)
                pred_mask[y1:y2, x1:x2] = np.array(resized)
                rle = self._encode_mask(pred_mask)
                
                # Some work for safety
                assert(pred_score[i] >= self._score_thresh)  # I don't know why the original code has something like this
                # convert [xmin, ymin, xmax, ymax]  to [xmin, ymin, w, h]
                pred_bbox_wh = pred_bbox[i]
                pred_bbox_wh[2:4] -= pred_bbox_wh[:2]

                self._results.append({'image_id': imgid,
                                      'category_id': self._train_to_anno_id[pred_label[i]],
                                      'bbox': list(map(lambda x: float(round(x, 2)), pred_bbox_wh[:4])),
                                      'score': float(round(pred_score[i], 3)),
                                      'segmentation': rle})
