"""Train YOLOv3 with random shapes."""
import argparse
import os
import logging
import time
import warnings
import numpy as np
import mxnet as mx
from mxnet import nd
from mxnet import gluon
from mxnet import autograd
import gluoncv as gcv
from gluoncv import data as gdata
from gluoncv import utils as gutils
from gluoncv.model_zoo import get_model
from gluoncv.data.batchify import Tuple, Stack, Pad
from gluoncv.data.transforms.presets.yolo import YOLO3DefaultTrainTransform
from gluoncv.data.transforms.presets.yolo import YOLO3DefaultValTransform
from gluoncv.data.dataloader import RandomTransformDataLoader
from gluoncv.utils.metrics.voc_detection import VOC07MApMetric
from gluoncv.utils.metrics.coco_detection import COCODetectionMetric
from gluoncv.utils.metrics.voc_polygon_detection import VOC07PolygonMApMetric
from gluoncv.utils import LRScheduler

def parse_args():
    parser = argparse.ArgumentParser(description='Train YOLO networks with random input shape.')
    parser.add_argument('--network', type=str, default='tiny_darknet',
                        help="Base network name which serves as feature extraction base.")
    parser.add_argument('--data-shape', type=int, default=416,
                        help="Input data shape for evaluation, use 320, 416, 608... " +
                             "Training is with random shapes from (320 to 608).")
    parser.add_argument('--batch-size', type=int, default=40,
                        help='Training mini-batch size')
    parser.add_argument('--dataset', type=str, default='voc',
                        help='Training dataset. Now support voc.')
    parser.add_argument('--num-workers', '-j', dest='num_workers', type=int,
                        default=8, help='Number of data workers, you can use larger '
                        'number to accelerate data loading, if you CPU and GPUs are powerful.')
    parser.add_argument('--gpus', type=str, default='1',
                        help='Training with GPUs, you can specify 1,3 for example.')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Training epochs.')
    parser.add_argument('--resume', type=str, default='',
                        help='Resume from previously saved parameters if not None. '
                        'For example, you can resume from ./yolo3_xxx_0123.params')
    parser.add_argument('--start-epoch', type=int, default=0,
                        help='Starting epoch for resuming, default is 0 for new training.'
                        'You can specify it to 100 for example to start from 100 epoch.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate, default is 0.001')
    parser.add_argument('--lr-mode', type=str, default='step',
                        help='learning rate scheduler mode. options are step, poly and cosine.')
    parser.add_argument('--lr-decay', type=float, default=0.1,
                        help='decay rate of learning rate. default is 0.1.')
    parser.add_argument('--lr-decay-period', type=int, default=0,
                        help='interval for periodic learning rate decays. default is 0 to disable.')
    parser.add_argument('--lr-decay-epoch', type=str, default= '160,180',
                        help='epochs at which learning rate decays. default is 260,280.')
    parser.add_argument('--warmup-lr', type=float, default=0.0,
                        help='starting warmup learning rate. default is 0.0.')
    parser.add_argument('--warmup-epochs', type=int, default=10,
                        help='number of warmup epochs.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum, default is 0.9')
    parser.add_argument('--wd', type=float, default=0.0005,
                        help='Weight decay, default is 5e-4')
    parser.add_argument('--log-interval', type=int, default=100,
                        help='Logging mini-batch interval. Default is 100.')
    parser.add_argument('--save-prefix', type=str, default='./tiny_result/',
                        help='Saving parameter prefix')
    parser.add_argument('--save-interval', type=int, default=10,
                        help='Saving parameters epoch interval, best model will always be saved.')
    parser.add_argument('--val-interval', type=int, default=10,
                        help='Epoch interval for validation, increase the number will reduce the '
                             'training time if validation is slow.')
    parser.add_argument('--seed', type=int, default=233,
                        help='Random seed to be fixed.')
    parser.add_argument('--num-samples', type=int, default=-1,
                        help='Training images. Use -1 to automatically get the number.')
    parser.add_argument('--syncbn', action='store_true',
                        help='Use synchronize BN across devices.')
    parser.add_argument('--no-random-shape', action='store_false',
                        help='Use fixed size(data-shape) throughout the training, which will be faster '
                        'and require less memory. However, final model will be slightly worse.')
    parser.add_argument('--no-wd', action='store_true',
                        help='whether to remove weight decay on bias, and beta/gamma for batchnorm layers.')
    parser.add_argument('--mixup', action='store_true',
                        help='whether to enable mixup.')
    parser.add_argument('--no-mixup-epochs', type=int, default=20,
                        help='Disable mixup training if enabled in the last N epochs.')
    parser.add_argument('--label-smooth', action='store_true', help='Use label smoothing.')
    parser.add_argument('--num_bases', type=int, default=50, help='the number of bases')
    parser.add_argument('--only_bbox', type=bool, default=False,
                        help="Only train boox")
    parser.add_argument('--val_2012', type=bool, default=False,
                        help="val in pascal voc 2012, or will val in sbd")
    args = parser.parse_args()
    return args

def get_dataset(dataset, args):
    if dataset.lower() == 'voc':
        train_dataset = gdata.VOCDetection(
            splits=[('sbdche', 'train'+'_'+'8'+'_bboxwh')])
        if args.val_2012 == True:
            val_dataset = gdata.VOC_Val_Detection(
                splits=[('sbdche', 'val_2012_bboxwh')])
        else:
            val_dataset = gdata.VOC_Val_Detection(
                splits=[('sbdche', 'val'+'_'+'8'+'_bboxwh')])
        val_metric = VOC07MApMetric(iou_thresh=0.5, class_names=val_dataset.classes)
        val_polygon_metric = VOC07PolygonMApMetric(iou_thresh=0.5, class_names=val_dataset.classes)
    elif dataset.lower() == 'coco_pretrain':
        train_dataset = gdata.coco_pretrain_Detection(
            splits=[('_coco_20', 'train'+'_'+'8'+'_bboxwh')])
        if args.val_2012 == True:
            val_dataset = gdata.VOC_Val_Detection(
                splits=[('sbdche', 'val_2012_bboxwh')])
        else:
            val_dataset = gdata.VOC_Val_Detection(
                splits=[('sbdche', 'val'+'_'+'8'+'_bboxwh')])
        val_metric = VOC07MApMetric(iou_thresh=0.5, class_names=val_dataset.classes)
        val_polygon_metric = VOC07PolygonMApMetric(iou_thresh=0.5, class_names=val_dataset.classes)
    else:
        raise NotImplementedError('Dataset: {} not implemented.'.format(dataset))
    if args.num_samples < 0:
        args.num_samples = len(train_dataset)
    if args.mixup:
        from gluoncv.data import MixupDetection
        train_dataset = MixupDetection(train_dataset)
    return train_dataset, val_dataset, val_metric, val_polygon_metric

def get_dataloader(net, train_dataset, val_dataset, data_shape, batch_size, num_workers, args):
    """Get dataloader."""
    width, height = data_shape, data_shape
    batchify_fn = Tuple(*([Stack() for _ in range(7)] + [Pad(axis=0, pad_val=-1) for _ in range(1)]))  # stack image, all targets generated
    if args.no_random_shape:
        # True
        train_loader = gluon.data.DataLoader(
            train_dataset.transform(YOLO3DefaultTrainTransform(width, height, net, mixup=args.mixup, num_bases = args.num_bases)),
            batch_size, True, batchify_fn=batchify_fn, last_batch='rollover', num_workers=num_workers)
    else:
        transform_fns = [YOLO3DefaultTrainTransform(x * 32, x * 32, net, mixup=args.mixup, num_bases = args.num_bases) for x in range(10, 20)]
        train_loader = RandomTransformDataLoader(
            transform_fns, train_dataset, batch_size=batch_size, interval=10, last_batch='rollover',
            shuffle=True, batchify_fn=batchify_fn, num_workers=num_workers)
    val_batchify_fn = Tuple(Stack(), Pad(pad_val=-1))
    val_loader = gluon.data.DataLoader(
        val_dataset.transform(YOLO3DefaultValTransform(width, height, args.num_bases)),
        batch_size, False, batchify_fn=val_batchify_fn, last_batch='keep', num_workers=num_workers)
    return train_loader, val_loader

def save_params(net, best_map, current_map, epoch, save_interval, prefix):
    current_map = float(current_map)
    if current_map > best_map[0]:
        best_map[0] = current_map
        net.save_parameters('{:s}_best.params'.format(prefix, epoch, current_map))
        with open(prefix+'_best_map.log', 'a') as f:
            f.write('{:04d}:\t{:.4f}\n'.format(epoch, current_map))
    if save_interval and epoch % save_interval == 0:
        net.save_parameters('{:s}_{:04d}_{:.4f}.params'.format(prefix, epoch, current_map))

def validate(net, val_data, ctx, eval_metric,polygon_metric, args):
    """Test on validation dataset."""
    eval_metric.reset()
    # set nms threshold and topk constraint
    net.set_nms(nms_thresh=0.45, nms_topk=400)
    mx.nd.waitall()
    net.hybridize()
    for batch in val_data:
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
        det_bboxes = []
        det_ids = []
        det_scores = []
        # det_coef_centers = []
        det_coefs = []
        # det_r_all = []
        gt_bboxes = []
        gt_points_xs = []
        gt_points_ys = []
        gt_ids = []
        gt_difficults = []
        gt_widths = []
        gt_heights = []
        for x, y in zip(data, label):
            # get prediction results
            ids, scores, bboxes, coef = net(x)
            det_ids.append(ids)
            det_scores.append(scores)
            # det_coef_centers.append(absolute_coef_centers)
            det_coefs.append(coef)
            # clip to image size
            det_bboxes.append(bboxes.clip(0, batch[0].shape[2]))
            # split ground truths
            gt_ids.append(y.slice_axis(axis=-1, begin=4 + 720, end=5 + 720))
            gt_bboxes.append(y.slice_axis(axis=-1, begin=0, end=4))
            gt_points_xs.append(y.slice_axis(axis=-1, begin=4, end=4 + 360))
            gt_points_ys.append(y.slice_axis(axis=-1, begin=4 + 360, end=4 + 720))
            gt_difficults.append(y.slice_axis(axis=-1, begin=5 + 720, end=6 + 720) if y.shape[-1] > 5 else None)
            gt_widths.append(y.slice_axis(axis=-1, begin=6 + 720, end=7 + 720))
            gt_heights.append(y.slice_axis(axis=-1, begin=7 + 720, end=8 + 720))
        # update metric
        eval_metric.update(det_bboxes, det_ids, det_scores, gt_bboxes, gt_ids, gt_difficults)
        polygon_metric.update(det_bboxes, det_coefs, det_ids, det_scores, gt_bboxes, gt_points_xs,
                              gt_points_ys, gt_ids, gt_widths, gt_heights, gt_difficults)
    return eval_metric.get(), polygon_metric.get()

def train(net, train_data, val_data, eval_metric, polygon_metric, ctx, args):
    """Training pipeline"""
    net.collect_params().reset_ctx(ctx)
    if args.no_wd:
        for k, v in net.collect_params('.*beta|.*gamma|.*bias').items():
            v.wd_mult = 0.0

    if args.label_smooth:
        net._target_generator._label_smooth = True

    if args.lr_decay_period > 0:
        lr_decay_epoch = list(range(args.lr_decay_period, args.epochs, args.lr_decay_period))
    else:
        lr_decay_epoch = [int(i) for i in args.lr_decay_epoch.split(',')]
    lr_scheduler = LRScheduler(mode=args.lr_mode,
                               baselr=args.lr,
                               niters=args.num_samples // args.batch_size,
                               nepochs=args.epochs,
                               step=lr_decay_epoch,
                               step_factor=args.lr_decay, power=2,
                               warmup_epochs=args.warmup_epochs)

    trainer = gluon.Trainer(
        net.collect_params(), 'sgd',
        {'wd': args.wd, 'momentum': args.momentum, 'lr_scheduler': lr_scheduler},
        kvstore='local')
    # targets
    sigmoid_ce = gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=False)
    l1_loss = gluon.loss.L1Loss()

    # metrics
    obj_metrics = mx.metric.Loss('ObjLoss')
    center_metrics = mx.metric.Loss('BoxCenterLoss')
    scale_metrics = mx.metric.Loss('BoxScaleLoss')
    # coef_center_metrics = mx.metric.Loss('CoefCenterLoss')
    coef_metrics = mx.metric.Loss('CoefLoss')
    # w_metrics = mx.metric.Loss('wLoss')
    cls_metrics = mx.metric.Loss('ClassLoss')
    # set up logger
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_file_path = args.save_prefix + '_train.log'
    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    fh = logging.FileHandler(log_file_path)
    logger.addHandler(fh)
    logger.info(args)
    logger.info('Start training from [Epoch {}]'.format(args.start_epoch))
    best_map = [0]
    for epoch in range(args.start_epoch, args.epochs):
        if args.mixup:
            # TODO(threshold): more elegant way to control mixup during runtime
            try:
                train_data._dataset.set_mixup(np.random.beta, 1.5, 1.5)
            except AttributeError:
                train_data._dataset._data.set_mixup(np.random.beta, 1.5, 1.5)
            if epoch >= args.epochs - args.no_mixup_epochs:
                try:
                    train_data._dataset.set_mixup(None)
                except AttributeError:
                    train_data._dataset._data.set_mixup(None)
        tic = time.time()
        btic = time.time()
        mx.nd.waitall()
        net.hybridize()
        for i, batch in enumerate(train_data):
            batch_size = batch[0].shape[0]
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
            fixed_targets = [gluon.utils.split_and_load(batch[it], ctx_list=ctx, batch_axis=0) for it in range(1, 7)]
            gt_boxes = gluon.utils.split_and_load(batch[7], ctx_list=ctx, batch_axis=0)
            sum_losses = []
            obj_losses = []
            center_losses = []
            scale_losses = []
            # coef_center_losses = []
            coef_losses = []
            cls_losses = []
            with autograd.record():
                for ix, x in enumerate(data):
                    obj_loss, center_loss, scale_loss, coef_loss, cls_loss = net(x, gt_boxes[ix], *[ft[ix] for ft in fixed_targets])
                    if(args.only_bbox):
                        sum_losses.append(obj_loss + center_loss + scale_loss +  cls_loss)
                    else:
                        sum_losses.append(obj_loss + center_loss + scale_loss + 0.5 * coef_loss  + cls_loss)
                        # coef_center_losses.append(coef_center_loss)
                        coef_losses.append(coef_loss)
                    obj_losses.append(obj_loss)
                    center_losses.append(center_loss)
                    scale_losses.append(scale_loss)
                    cls_losses.append(cls_loss)
                autograd.backward(sum_losses)
            lr_scheduler.update(i, epoch)
            trainer.step(batch_size)
            if(args.only_bbox == False):
                # coef_center_metrics.update(0, coef_center_losses)
                coef_metrics.update(0,coef_losses)
            obj_metrics.update(0, obj_losses)
            center_metrics.update(0, center_losses)
            scale_metrics.update(0, scale_losses)
            cls_metrics.update(0, cls_losses)
            if args.log_interval and not (i + 1) % args.log_interval:
                name1, loss1 = obj_metrics.get()
                name2, loss2 = center_metrics.get()
                name3, loss3 = scale_metrics.get()
                if(args.only_bbox == False):
                    # name4, loss4 = coef_center_metrics.get()
                    name5, loss5 = coef_metrics.get()
                name6, loss6 = cls_metrics.get()
                if(args.only_bbox):
                    logger.info('[Epoch {}][Batch {}], LR: {:.2E}, Speed: {:.3f} samples/sec, {}={:.3f}, {}={:.3f}, {}={:.3f}, {}={:.3f}'.format(
                    epoch, i, trainer.learning_rate, batch_size/(time.time()-btic), name1, loss1, name2, loss2, name3, loss3, name6, loss6))
                else:
                    logger.info('[Epoch {}][Batch {}], LR: {:.2E}, Speed: {:.3f} samples/sec, {}={:.3f}, {}={:.3f}, {}={:.3f}, {}={:.3f}, {}={:.3f}'.format(
                    epoch, i, trainer.learning_rate, batch_size/(time.time()-btic), name1, loss1, name2, loss2, name3, loss3, name5, loss5, name6, loss6))
            btic = time.time()

        name1, loss1 = obj_metrics.get()
        name2, loss2 = center_metrics.get()
        name3, loss3 = scale_metrics.get()
        if(args.only_bbox==False):
            # name4, loss4 = coef_center_metrics.get()
            name5, loss5 = coef_metrics.get()
        name6, loss6 = cls_metrics.get()
        if(args.only_bbox):
            logger.info('[Epoch {}] Training cost: {:.3f}, {}={:.3f}, {}={:.3f}, {}={:.3f}, {}={:.3f}'.format(
            epoch, (time.time()-tic), name1, loss1, name2, loss2, name3, loss3, name6, loss6))
        else:
            logger.info('[Epoch {}] Training cost: {:.3f}, {}={:.3f}, {}={:.3f}, {}={:.3f}, {}={:.3f}, {}={:.3f}'.format(
            epoch, (time.time()-tic), name1, loss1, name2, loss2, name3, loss3, name5, loss5, name6, loss6))
        if False and not (epoch) % args.val_interval:
            # consider reduce the frequency of validation to save time
            map_bbox, map_polygon = validate(net, val_data, ctx, eval_metric, polygon_metric,args)
            map_name, mean_ap = map_bbox
            polygonmap_name, polygonmean_ap = map_polygon
            val_msg = '\n'.join(['{}={}'.format(k, v) for k, v in zip(map_name, mean_ap)])
            logger.info('[Epoch {}] Validation: \n{}'.format(epoch, val_msg))
            polygonval_msg = '\n'.join(['{}={}'.format(k, v) for k, v in zip(polygonmap_name, polygonmean_ap)])
            logger.info('[Epoch {}] PolygonValidation: \n{}'.format(epoch, polygonval_msg))
            current_map = float(polygonmean_ap[-1])
        else:
            current_map = 0.
        save_params(net, best_map, current_map, epoch, args.save_interval, args.save_prefix)

if __name__ == '__main__':
    args = parse_args()
    # fix seed for mxnet, numpy and python builtin random generator.
    gutils.random.seed(args.seed)

    # training contexts
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',') if i.strip()]
    ctx = ctx if ctx else [mx.cpu()]

    # network
    net_name = '_'.join(('yolo3', args.network, args.dataset))
    args.save_prefix += net_name
    print(f"net_name = {net_name}")
    # use sync bn if specified
    if args.syncbn and len(ctx) > 1:
        net = get_model(net_name, pretrained_base=True, norm_layer=gluon.contrib.nn.SyncBatchNorm,
                        norm_kwargs={'num_devices': len(ctx)})
        async_net = get_model(net_name, pretrained_base=False)  # used by cpu worker
    else:
        net = get_model(net_name, pretrained_base=True)
        async_net = net
    if args.resume.strip():
        net.load_parameters(args.resume.strip())
        async_net.load_parameters(args.resume.strip())
    else:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            net.initialize()
            async_net.initialize()
    print("model loaded")
    # training data
    train_dataset, val_dataset, eval_metric, polygon_metric = get_dataset(args.dataset, args)
    # train_data, val_data = get_dataloader(
    #     async_net, train_dataset, val_dataset, args.data_shape, args.batch_size, args.num_workers, args)
    print("dataset done")
    train_data, val_data = get_dataloader(
        async_net, train_dataset, val_dataset, args.data_shape, args.batch_size, args.num_workers, args)
    print("dataloader done")
    # training
    train(net, train_data, val_data, eval_metric, polygon_metric, ctx, args)
