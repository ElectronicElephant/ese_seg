# pylint: disable=arguments-differ
"""Custom losses.
Losses are subclasses of gluon.loss.Loss which is a HybridBlock actually.
"""
from __future__ import absolute_import
from mxnet import gluon
from mxnet import nd
from mxnet.gluon.loss import Loss, _apply_weighting, _reshape_like

__all__ = ['FocalLoss', 'SSDMultiBoxLoss', 'YOLOV3Loss',
           'MixSoftmaxCrossEntropyLoss', 'MixSoftmaxCrossEntropyOHEMLoss']

# For sbd
# x_mean = [-10372.72573, -65.62571687, 8.113603703, -4.129944152, -70.272901, 48.39311715, 13.54921555, -6.204910281, -82.93014464, -13.49012528, -1.377779259, -0.359250444, -9.937065122, 3.458047501, -0.637840469, 6.447263647, -0.159122537, -4.013595629, 0.368631004, -0.798153475, -0.675555162, -0.64375462, -1.876287186, -5.29987036, -3.081862721, -1.205230327, 1.611716191, -1.447915821, -1.008998948, 2.062282999, -0.366824452, -0.76531215, -5.657952825, -0.702878769, 4.139859116, 3.660853075, -4.365368841, -3.759996972, 0.10982376, -0.901409142, -0.914115701, -0.287375188, 0.673038067, -1.64012666, 0.983785635, 0.369574124, -0.080789953, 1.392399963, 1.066113083, -1.677959563]
# sqrt_var = [2803.685003, 2161.867545, 2087.585027, 1877.30127, 1616.864336, 1459.067858, 1325.125214, 1166.808235, 1078.785935, 954.0199264, 941.2122393, 877.5677226, 822.7808391, 767.6920133, 742.2590864, 709.1605303, 707.289721, 671.4039817, 625.9553238, 595.3469511, 586.0426203, 571.3110848, 560.1314211, 535.6616464, 534.0023283, 508.0441017, 489.9292073, 485.8909739, 474.3963334, 466.0055539, 449.4863225, 444.2908041, 437.5323292, 432.4269679, 409.8391652, 406.6126991, 400.0697815, 396.4458867, 391.1772833, 384.5596921, 377.9070695, 371.8721299, 366.0097442, 350.6612788, 350.3757641, 347.7540654, 343.1649469, 335.2165795, 332.2271304, 327.4941672]
# x_min = nd.array([-15482.42389, -6890.276516, -5538.785075, -5816.060271, -4558.030574, -6838.889213, -4845.916472, -4227.809116, -3826.952649, -3513.548843, -4829.720262, -3443.692333, -4599.026508, -3632.84294, -2830.503294, -2743.853048, -2821.758048, -2892.371713, -2783.273599, -2328.833622, -2287.968436, -2414.126732, -2567.267731, -2730.10522, -2273.68255, -2323.956728, -1884.060547, -2001.761208, -2243.889445, -2216.714357, -2104.139639, -2422.256762, -1817.853336, -2194.244812, -1718.128955, -1841.351373, -1889.84415, -1951.221188, -1691.402656, -1864.771442, -1685.0357, -1469.929736, -1399.324855, -1524.768124, -1847.479326, -1464.607373, -1438.142198, -1467.812184, -1486.837391, -1468.122325])
# x_max = nd.array([0, 6975.697248, 5453.263858, 5710.089781, 8459.724464, 5007.860651, 4914.459004, 4938.654403, 3904.488484, 4878.943755, 4372.214577, 3314.96113, 3958.25269, 3224.153406, 2789.40399, 3242.741783, 2877.455503, 2805.222196, 2610.172046, 2595.920655, 2711.131864, 2382.223149, 2464.276953, 2649.856643, 2540.557773, 3242.252329, 2196.179273, 2306.631054, 1858.970692, 2254.906552, 1963.849169, 2242.1793, 1795.529975, 2562.035894, 1748.565462, 1918.170148, 1555.71773, 2739.393537, 1662.911931, 1877.164972, 1832.516479, 1596.010903, 1562.090407, 1526.694717, 1544.801876, 1441.254865, 1445.42314, 1397.396689, 1378.251306, 1448.646948])

# For coco
x_min =  [-15408.068104448881, -6893.558054798728, -7003.406866817996, -7173.151488944284, -8880.702237736832, -5105.870172246976, -5765.5587195891485, -5024.227379613461, -5711.952435731431, -5495.081529198267, -5833.850420273756, -4434.37549020221, -5849.216285241527, -4148.2654407091895, -3569.2531463158916, -4339.357174902734, -3655.7764618342203, -3823.3819004419747, -3141.4357750292143, -4225.954414632274, -4508.907524652018, -2985.9986722598996, -3351.4766979792385, -3542.6383142662216, -3208.1730852282417, -3276.2051016720184, -2778.240479008936, -2687.1807642675817, -2864.3521512732636, -2667.346488961604, -2679.78247499033, -2778.1530493300193, -2615.297232543604, -2887.83922977382, -2814.11271273744, -2665.593586967864, -2244.208215546852, -2604.715325774133, -2555.901894909533, -3023.0542016462905, -3120.604337844805, -2276.2895359281847, -2105.2348396526972, -2107.14859953116, -4062.8254106434965, -2053.622120297776, -2197.4795855647635, -2042.3037948693445, -2467.5308906646937, -2245.5552141163903]
x_max =  [0.0, 6832.446298223013, 7426.165815379417, 6974.701596658017, 4716.901065835743, 8131.608870119551, 5740.872699165772, 4581.338796015798, 5217.3107185273375, 5434.597380283167, 5576.999587107373, 4287.165831371201, 4963.129599067099, 4621.02114880624, 3682.6609034386593, 4353.761120273803, 4174.824769494295, 3994.883741475415, 3283.721646183678, 3798.4092325829133, 4347.6387582645475, 3372.640698902529, 3295.0094768303293, 2926.3658864426816, 3499.712903749524, 3039.4470982219764, 2473.9809720368858, 2405.556357232199, 3184.463910105855, 2784.1799697475394, 2284.209254236527, 2625.629675147772, 2336.795159840813, 2528.887489215271, 2782.44841959135, 2342.962374129638, 2477.479578295029, 2332.187232909927, 2459.4770586568147, 2794.3178970248023, 2505.2624769384856, 2767.461569799445, 1918.2837463541125, 2050.6555855719203, 2690.2851498377295, 2887.8565628719634, 2263.3678542969415, 1798.6753995660308, 2160.58798020158, 2092.1122966365115]
x_mean =  [-9806.334230601844, -0.1265930578759492, -44.70213815499062, 4.1016068564528485, 34.85025642973737, -3.515908079075314, 33.660171096323424, 130.83580930988637, 0.21492417056751245, 2.8112355899964174, 18.833675030236837, 0.626437650033731, -3.2008816942056932, 0.016458105852027838, 8.394310893579835, -5.059975166016848, 0.3082644590455881, 1.5217574906226543, -0.018611740148539873, 0.7879045499805826, -0.24098315206080123, -0.8808304685364998, -0.7913288600067822, -3.8891420056181145, 6.353012221300202, -0.4225753767008447, 0.27977828714261016, 0.08870383388150666, -5.0118744432067786, 0.48268046843874046, -18.893481918065138, 0.7532384238847303, -5.311672820484189, -6.17895441522754, -0.356883920263817, -0.38091052476647386, 0.08936253734500309, 1.2569901866919777, 1.4373361126170598, -3.279811354042419, -2.068920651918281, 0.060461234684045725, 0.6868672104607721, 0.03698304732462165, -2.532655293110934, -0.1347230399250139, 1.5058533210691571, 0.09911752586840517, -0.012458458813556523, 2.3168010192166624]
sqrt_var =  [3178.8067937849487, 2108.5508769810863, 1994.220493314925, 1938.7995338537069, 1639.3960276470855, 1432.760749474181, 1288.8778997753777, 1173.580613711433, 1122.0012218569218, 928.4866017001266, 921.4166617204439, 856.0066864535319, 827.1107657788409, 801.1389228102764, 747.0316068891458, 738.6573541712918, 713.7510960451755, 656.5846272310993, 644.8258954808872, 606.805027464974, 596.0518795953916, 588.3050234920775, 586.3892172394093, 554.8030689406621, 543.5063777522089, 503.09736918051834, 496.38691611492146, 488.43183601616107, 487.21877068107796, 476.8424720930743, 459.66215884985763, 442.3700766285788, 435.6704169622154, 429.36612409375545, 410.33755900022, 408.4439272034049, 404.64446380132824, 394.1204334571845, 393.73104227507173, 389.3877034575619, 381.867970587664, 372.6268526542834, 358.85596109586754, 357.9271102515216, 352.67275163779277, 348.4594642007308, 344.12421096240666, 343.61001513303694, 331.4995424542531, 326.8226821028282]


class FocalLoss(gluon.loss.Loss):
    """Focal Loss for imbalanced classification.
    Focal loss was described in https://arxiv.org/abs/1708.02002

    Parameters
    ----------
    axis : int, default -1
        The axis to sum over when computing softmax and entropy.
    alpha : float, default 0.25
        The alpha which controls loss curve.
    gamma : float, default 2
        The gamma which controls loss curve.
    sparse_label : bool, default True
        Whether label is an integer array instead of probability distribution.
    from_logits : bool, default False
        Whether input is a log probability (usually from log_softmax) instead.
    batch_axis : int, default 0
        The axis that represents mini-batch.
    weight : float or None
        Global scalar weight for loss.
    num_class : int
        Number of classification categories. It is required is `sparse_label` is `True`.
    eps : float
        Eps to avoid numerical issue.
    size_average : bool, default True
        If `True`, will take mean of the output loss on every axis except `batch_axis`.

    Inputs:
        - **pred**: the prediction tensor, where the `batch_axis` dimension
          ranges over batch size and `axis` dimension ranges over the number
          of classes.
        - **label**: the truth tensor. When `sparse_label` is True, `label`'s
          shape should be `pred`'s shape with the `axis` dimension removed.
          i.e. for `pred` with shape (1,2,3,4) and `axis = 2`, `label`'s shape
          should be (1,2,4) and values should be integers between 0 and 2. If
          `sparse_label` is False, `label`'s shape must be the same as `pred`
          and values should be floats in the range `[0, 1]`.
        - **sample_weight**: element-wise weighting tensor. Must be broadcastable
          to the same shape as label. For example, if label has shape (64, 10)
          and you want to weigh each sample in the batch separately,
          sample_weight should have shape (64, 1).
    Outputs:
        - **loss**: loss tensor with shape (batch_size,). Dimensions other than
          batch_axis are averaged out.
    """
    def __init__(self, axis=-1, alpha=0.25, gamma=2, sparse_label=True,
                 from_logits=False, batch_axis=0, weight=None, num_class=None,
                 eps=1e-12, size_average=True, **kwargs):
        super(FocalLoss, self).__init__(weight, batch_axis, **kwargs)
        self._axis = axis
        self._alpha = alpha
        self._gamma = gamma
        self._sparse_label = sparse_label
        if sparse_label and (not isinstance(num_class, int) or (num_class < 1)):
            raise ValueError("Number of class > 0 must be provided if sparse label is used.")
        self._num_class = num_class
        self._from_logits = from_logits
        self._eps = eps
        self._size_average = size_average

    def hybrid_forward(self, F, pred, label, sample_weight=None):
        """Loss forward"""
        if not self._from_logits:
            pred = F.sigmoid(pred)
        if self._sparse_label:
            one_hot = F.one_hot(label, self._num_class)
        else:
            one_hot = label > 0
        pt = F.where(one_hot, pred, 1 - pred)
        t = F.ones_like(one_hot)
        alpha = F.where(one_hot, self._alpha * t, (1 - self._alpha) * t)
        loss = -alpha * ((1 - pt) ** self._gamma) * F.log(F.minimum(pt + self._eps, 1))
        loss = _apply_weighting(F, loss, self._weight, sample_weight)
        if self._size_average:
            return F.mean(loss, axis=self._batch_axis, exclude=True)
        else:
            return F.sum(loss, axis=self._batch_axis, exclude=True)

def _as_list(arr):
    """Make sure input is a list of mxnet NDArray"""
    if not isinstance(arr, (list, tuple)):
        return [arr]
    return arr


class SSDMultiBoxLoss(gluon.Block):
    r"""Single-Shot Multibox Object Detection Loss.

    .. note::

        Since cross device synchronization is required to compute batch-wise statistics,
        it is slightly sub-optimal compared with non-sync version. However, we find this
        is better for converged model performance.

    Parameters
    ----------
    negative_mining_ratio : float, default is 3
        Ratio of negative vs. positive samples.
    rho : float, default is 1.0
        Threshold for trimmed mean estimator. This is the smooth parameter for the
        L1-L2 transition.
    lambd : float, default is 1.0
        Relative weight between classification and box regression loss.
        The overall loss is computed as :math:`L = loss_{class} + \lambda \times loss_{loc}`.

    """
    def __init__(self, negative_mining_ratio=3, rho=1.0, lambd=1.0, **kwargs):
        super(SSDMultiBoxLoss, self).__init__(**kwargs)
        self._negative_mining_ratio = max(0, negative_mining_ratio)
        self._rho = rho
        self._lambd = lambd

    def forward(self, cls_pred, box_pred, cls_target, box_target):
        """Compute loss in entire batch across devices."""
        # require results across different devices at this time
        cls_pred, box_pred, cls_target, box_target = [_as_list(x) \
            for x in (cls_pred, box_pred, cls_target, box_target)]
        # cross device reduction to obtain positive samples in entire batch
        num_pos = []
        for cp, bp, ct, bt in zip(*[cls_pred, box_pred, cls_target, box_target]):
            pos_samples = (ct > 0)
            num_pos.append(pos_samples.sum())
        num_pos_all = sum([p.asscalar() for p in num_pos])
        if num_pos_all < 1:
            # no positive samples found, return dummy losses
            return nd.zeros((1,)), nd.zeros((1,)), nd.zeros((1,))

        # compute element-wise cross entropy loss and sort, then perform negative mining
        cls_losses = []
        box_losses = []
        sum_losses = []
        for cp, bp, ct, bt in zip(*[cls_pred, box_pred, cls_target, box_target]):
            pred = nd.log_softmax(cp, axis=-1)
            pos = ct > 0
            cls_loss = -nd.pick(pred, ct, axis=-1, keepdims=False)
            rank = (cls_loss * (pos - 1)).argsort(axis=1).argsort(axis=1)
            hard_negative = rank < (pos.sum(axis=1) * self._negative_mining_ratio).expand_dims(-1)
            # mask out if not positive or negative
            cls_loss = nd.where((pos + hard_negative) > 0, cls_loss, nd.zeros_like(cls_loss))
            cls_losses.append(nd.sum(cls_loss, axis=0, exclude=True) / num_pos_all)

            bp = _reshape_like(nd, bp, bt)
            box_loss = nd.abs(bp - bt)
            box_loss = nd.where(box_loss > self._rho, box_loss - 0.5 * self._rho,
                                (0.5 / self._rho) * nd.square(box_loss))
            # box loss only apply to positive samples
            box_loss = box_loss * pos.expand_dims(axis=-1)
            box_losses.append(nd.sum(box_loss, axis=0, exclude=True) / num_pos_all)
            sum_losses.append(cls_losses[-1] + self._lambd * box_losses[-1])

        return sum_losses, cls_losses, box_losses


class YOLOV3Loss(gluon.loss.Loss):
    """Losses of YOLO v3.

    Parameters
    ----------
    batch_axis : int, default 0
        The axis that represents mini-batch.
    weight : float or None
        Global scalar weight for loss.

    """
    def __init__(self, batch_axis=0, weight=None, num_bases = 50, **kwargs):
        super(YOLOV3Loss, self).__init__(weight, batch_axis, **kwargs)
        self._sigmoid_ce = gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=False)
        self._l1_loss = gluon.loss.L1Loss()
        self._smoothl1_loss = gluon.loss.HuberLoss()
        # self._sigmoid_bce = gluon.loss.SigmoidBCELoss()
        # self._focal_loss = FocalLoss(sparse_label=False)
        self._num_bases = num_bases

    def hybrid_forward(self, F, objness, box_centers, box_scales, coef, cls_preds,
                       objness_t, center_t, scale_t, coef_t, weight_t, class_t, class_mask):
        """Compute ESE-Seg based on YOLOv3 losses.

        Parameters
        ----------
        objness : mxnet.nd.NDArray
            Predicted objectness (B, N), range (0, 1).  B * N * 1 - Tutian
        box_centers : mxnet.nd.NDArray
            Predicted box centers (x, y) (B, N, 2), range (0, 1).
        box_scales : mxnet.nd.NDArray
            Predicted box scales (width, height) (B, N, 2).
        cls_preds : mxnet.nd.NDArray
            Predicted class predictions (B, N, num_class), range (0, 1).
        objness_t : mxnet.nd.NDArray
            Objectness target, (B, N), 0 for negative 1 for positive, -1 for ignore.
        center_t : mxnet.nd.NDArray
            Center (x, y) targets (B, N, 2).
        scale_t : mxnet.nd.NDArray
            Scale (width, height) targets (B, N, 2).
        coef_center_t : mxnet.nd.NDArray
            Coefficient Center (x, y) targets (B, N, 2).
        coef_t : mxnet.nd.NDArray
            Coefficient targets (B, N, 2*deg+2).
        weight_t : mxnet.nd.NDArray
            Loss Multipliers for center and scale targets (B, N, 2).
        class_t : mxnet.nd.NDArray
            Class targets (B, N, num_class).
            It's relaxed one-hot vector, i.e., (1, 0, 1, 0, 0).
            It can contain more than one positive class.
        class_mask : mxnet.nd.NDArray
            0 or 1 mask array to mask out ignored samples (B, N, num_class).

        Returns
        -------
        tuple of NDArrays
            obj_loss: sum of objectness logistic loss
            center_loss: sum of box center logistic regression loss
            scale_loss: sum of box scale l1 loss
            coef_center_loss: sum of coef_center l1 loss
            coef_loss: sum of coefficient l1 loss
            cls_loss: sum of per class logistic loss
        """
        # compute some normalization count, except batch-size
        denorm = F.cast(
            F.shape_array(objness_t).slice_axis(axis=0, begin=1, end=None).prod(), 'float32')
        weight_t = F.broadcast_mul(weight_t, objness_t)

        # Weights of coefs
        coef_weight_t = weight_t * 2
        for i in range(int((self._num_bases - 2) / 2)):
            coef_weight_t = F.Concat(coef_weight_t, weight_t * (1 + 1/(i+1.1)), dim=-1)

        hard_objness_t = F.where(objness_t > 0, F.ones_like(objness_t), objness_t)
        new_objness_mask = F.where(objness_t > 0, objness_t, objness_t >= 0)

        obj_loss = F.broadcast_mul(
            self._sigmoid_ce(objness, hard_objness_t, new_objness_mask), denorm)
        center_loss = F.broadcast_mul(self._sigmoid_ce(box_centers, center_t, weight_t), denorm * 2)
        scale_loss = F.broadcast_mul(self._l1_loss(box_scales, scale_t, weight_t), denorm * 2)
        # New dataset - normalized according to each coef's var or uniform
        # coef_loss = F.broadcast_mul(self._smoothl1_loss(F.tanh(coef), F.tanh(coef_t), coef_weight_t), denorm * (self._num_bases))
        coef_loss = F.broadcast_mul(self._smoothl1_loss(coef, coef_t, coef_weight_t), denorm * (self._num_bases))
        
        denorm_class = F.cast(
            F.shape_array(class_t).slice_axis(axis=0, begin=1, end=None).prod(), 'float32')
        class_mask = F.broadcast_mul(class_mask, objness_t)
        cls_loss = F.broadcast_mul(self._sigmoid_ce(cls_preds, class_t, class_mask), denorm_class)

        return obj_loss, center_loss, scale_loss, coef_loss, cls_loss


class SoftmaxCrossEntropyLoss(Loss):
    r"""SoftmaxCrossEntropyLoss with ignore labels

    Parameters
    ----------
    axis : int, default -1
        The axis to sum over when computing softmax and entropy.
    sparse_label : bool, default True
        Whether label is an integer array instead of probability distribution.
    from_logits : bool, default False
        Whether input is a log probability (usually from log_softmax) instead
        of unnormalized numbers.
    weight : float or None
        Global scalar weight for loss.
    batch_axis : int, default 0
        The axis that represents mini-batch.
    ignore_label : int, default -1
        The label to ignore.
    size_average : bool, default False
        Whether to re-scale loss with regard to ignored labels.
    """
    def __init__(self, sparse_label=True, batch_axis=0, ignore_label=-1,
                 size_average=True, **kwargs):
        super(SoftmaxCrossEntropyLoss, self).__init__(None, batch_axis, **kwargs)
        self._sparse_label = sparse_label
        self._ignore_label = ignore_label
        self._size_average = size_average

    def hybrid_forward(self, F, pred, label):
        """Compute loss"""
        softmaxout = F.SoftmaxOutput(
            pred, label.astype(pred.dtype), ignore_label=self._ignore_label,
            multi_output=self._sparse_label,
            use_ignore=True, normalization='valid' if self._size_average else 'null')
        loss = -F.pick(F.log(softmaxout), label, axis=1, keepdims=True)
        loss = F.where(label.expand_dims(axis=1) == self._ignore_label,
                       F.zeros_like(loss), loss)
        return F.mean(loss, axis=self._batch_axis, exclude=True)

class MixSoftmaxCrossEntropyLoss(SoftmaxCrossEntropyLoss):
    """SoftmaxCrossEntropyLoss2D with Auxiliary Loss

    Parameters
    ----------
    aux : bool, default True
        Whether to use auxiliary loss.
    aux_weight : float, default 0.2
        The weight for aux loss.
    ignore_label : int, default -1
        The label to ignore.
    """
    def __init__(self, aux=True, mixup=False, aux_weight=0.2, ignore_label=-1, **kwargs):
        super(MixSoftmaxCrossEntropyLoss, self).__init__(
            ignore_label=ignore_label, **kwargs)
        self.aux = aux
        self.mixup = mixup
        self.aux_weight = aux_weight

    def _aux_forward(self, F, pred1, pred2, label, **kwargs):
        """Compute loss including auxiliary output"""
        loss1 = super(MixSoftmaxCrossEntropyLoss, self). \
            hybrid_forward(F, pred1, label, **kwargs)
        loss2 = super(MixSoftmaxCrossEntropyLoss, self). \
            hybrid_forward(F, pred2, label, **kwargs)
        return loss1 + self.aux_weight * loss2

    def _aux_mixup_forward(self, F, pred1, pred2, label1, label2, lam):
        """Compute loss including auxiliary output"""
        loss1 = self._mixup_forwar(F, pred1, label1, label2, lam)
        loss2 = self._mixup_forwar(F, pred2, label1, label2, lam)
        return loss1 + self.aux_weight * loss2

    def _mixup_forward(self, F, pred, label1, label2, lam, sample_weight=None):
        if not self._from_logits:
            pred = F.log_softmax(pred, self._axis)
        if self._sparse_label:
            loss1 = -F.pick(pred, label1, axis=self._axis, keepdims=True)
            loss2 = -F.pick(pred, label2, axis=self._axis, keepdims=True)
            loss = lam * loss1 + (1 - lam) * loss2
        else:
            label1 = _reshape_like(F, label1, pred)
            label2 = _reshape_like(F, label2, pred)
            loss1 = -F.sum(pred*label1, axis=self._axis, keepdims=True)
            loss2 = -F.sum(pred*label2, axis=self._axis, keepdims=True)
            loss = lam * loss1 + (1 - lam) * loss2
        loss = _apply_weighting(F, loss, self._weight, sample_weight)
        return F.mean(loss, axis=self._batch_axis, exclude=True)

    def hybrid_forward(self, F, *inputs, **kwargs):
        """Compute loss"""
        if self.aux:
            if self.mixup:
                return self._aux_mixup_forward(F, *inputs, **kwargs)
            else:
                return self._aux_forward(F, *inputs, **kwargs)
        else:
            if self.mixup:
                return self._mixup_forward(F, *inputs, **kwargs)
            else:
                return super(MixSoftmaxCrossEntropyLoss, self). \
                    hybrid_forward(F, *inputs, **kwargs)

class SoftmaxCrossEntropyOHEMLoss(Loss):
    r"""SoftmaxCrossEntropyLoss with ignore labels

    Parameters
    ----------
    axis : int, default -1
        The axis to sum over when computing softmax and entropy.
    sparse_label : bool, default True
        Whether label is an integer array instead of probability distribution.
    from_logits : bool, default False
        Whether input is a log probability (usually from log_softmax) instead
        of unnormalized numbers.
    weight : float or None
        Global scalar weight for loss.
    batch_axis : int, default 0
        The axis that represents mini-batch.
    ignore_label : int, default -1
        The label to ignore.
    size_average : bool, default False
        Whether to re-scale loss with regard to ignored labels.
    """
    def __init__(self, sparse_label=True, batch_axis=0, ignore_label=-1,
                 size_average=True, **kwargs):
        super(SoftmaxCrossEntropyOHEMLoss, self).__init__(None, batch_axis, **kwargs)
        self._sparse_label = sparse_label
        self._ignore_label = ignore_label
        self._size_average = size_average

    def hybrid_forward(self, F, pred, label):
        """Compute loss"""
        softmaxout = F.contrib.SoftmaxOHEMOutput(
            pred, label.astype(pred.dtype), ignore_label=self._ignore_label,
            multi_output=self._sparse_label,
            use_ignore=True, normalization='valid' if self._size_average else 'null',
            thresh=0.6, min_keep=256)
        loss = -F.pick(F.log(softmaxout), label, axis=1, keepdims=True)
        loss = F.where(label.expand_dims(axis=1) == self._ignore_label,
                       F.zeros_like(loss), loss)
        return F.mean(loss, axis=self._batch_axis, exclude=True)

class MixSoftmaxCrossEntropyOHEMLoss(SoftmaxCrossEntropyOHEMLoss):
    """SoftmaxCrossEntropyLoss2D with Auxiliary Loss

    Parameters
    ----------
    aux : bool, default True
        Whether to use auxiliary loss.
    aux_weight : float, default 0.2
        The weight for aux loss.
    ignore_label : int, default -1
        The label to ignore.
    """
    def __init__(self, aux=True, aux_weight=0.2, ignore_label=-1, **kwargs):
        super(MixSoftmaxCrossEntropyOHEMLoss, self).__init__(
            ignore_label=ignore_label, **kwargs)
        self.aux = aux
        self.aux_weight = aux_weight

    def _aux_forward(self, F, pred1, pred2, label, **kwargs):
        """Compute loss including auxiliary output"""
        loss1 = super(MixSoftmaxCrossEntropyOHEMLoss, self). \
            hybrid_forward(F, pred1, label, **kwargs)
        loss2 = super(MixSoftmaxCrossEntropyOHEMLoss, self). \
            hybrid_forward(F, pred2, label, **kwargs)
        return loss1 + self.aux_weight * loss2

    def hybrid_forward(self, F, *inputs, **kwargs):
        """Compute loss"""
        if self.aux:
            return self._aux_forward(F, *inputs, **kwargs)
        else:
            return super(MixSoftmaxCrossEntropyOHEMLoss, self). \
                hybrid_forward(F, *inputs, **kwargs)
