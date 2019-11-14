"""YOLO Demo script."""
import os
import argparse
import mxnet as mx
import gluoncv as gcv
import numpy as np
import time
from gluoncv.data.transforms import presets
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
plt.ioff()
from tqdm import tqdm
import cv2 as cv
import shutil
from matplotlib.ticker import NullLocator
from gluoncv.data.transforms.presets.yolo import YOLO3UsdSegCocoValTransform
from gluoncv.data.mscoco.instance import COCOInstance
from gluoncv.data import batchify
from gluoncv.data.batchify import Tuple, Stack, Pad
from mxnet import gluon
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description='Test with YOLO networks.')
    parser.add_argument('--network', type=str, default='yolo3_darknet53_coco',
                        help="Base network name yolo3_darknet53_voc\yolo3_tiny_darknet_voc")
    parser.add_argument('--images', type=str, default= 'vis_img',
                        help='Test images, use comma to split multiple.')
    parser.add_argument('--save_dir', type=str, default='vis_demo',
                        help='')
    parser.add_argument('--gpus', type=str, default='0',
                        help='Training with GPUs, you can specify 1,3 for example.')
    parser.add_argument('--pretrained', type=str, default='result_coco_pretrain_var_tanh_lbsm_small_lr_yolo3_darknet53_coco_0102_0.0000.params',
                        help='Load weights from previously saved parameters.')
    parser.add_argument('--thresh', type=float, default=0.45,
                        help='Threshold of object score when visualize the bboxes.')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # context list
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',') if i.strip()]
    ctx = [mx.cpu()] if not ctx else ctx
    if not os.path.exists(args.images):
        os.mkdir(args.images)  

    image_nameList = os.listdir(args.images)
    image_list = []
    for i in image_nameList:
        image_list.append(os.path.join(args.images,i))

    if args.pretrained.lower() in ['true', '1', 'yes', 't']:
        net = gcv.model_zoo.get_model(args.network, pretrained=True)
    else:
        net = gcv.model_zoo.get_model(args.network, pretrained=False, pretrained_base=False)
        net.load_parameters(args.pretrained)
    net.set_nms(0.45, 200)

    net.collect_params().reset_ctx(ctx = ctx)
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    # image_list_batch = []
    # image_list = open('/disk1/home/tutian/ese_seg/sbd/ImageSets/Segmentation/val_2012_bboxwh.txt').readlines()
    # for img in image_list:
    #     image_list_batch.append('/disk1/home/tutian/ese_seg/sbd/img/'+img[:-1]+'.jpg')
    image_list_batch = image_list 
    total_time_net = 0
    total_time_post = 0
    # print(image_list_batch)
    for image in tqdm(image_list_batch):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1)
        img_str = image.split('/')[-1]
        x, img = presets.yolo.load_test(image, short=416)
        img_w = img.shape[0]
        img_h = img.shape[1]
        x = x.as_in_context(ctx[0])
        a = time.time()
        ids, scores, bboxes, coef = [xx[0].asnumpy() for xx in net(x)]
        b = time.time()  # Pure network speed
        ax = gcv.utils.viz.plot_r_polygon(img, bboxes, coef, img_w, img_h, scores, ids , 
                                          thresh=args.thresh,class_names=net.classes, ax=ax,
                                          num_bases=50, method='var')
        c = time.time()
        total_time_net += b - a
        total_time_post += c - b
        plt.axis('off')
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        plt.subplots_adjust(top = 0.995, bottom = 0.005, right = 0.995, left = 0.005, hspace = 0, wspace = 0)
        plt.savefig(os.path.join(args.save_dir,img_str))
        plt.close()
    print("network speed ",1.0*len(image_list_batch) / total_time_net, "fps")
    print("post process speed ",1.0*len(image_list_batch) / total_time_post, "fps")


def split_and_load(batch, ctx_list):
    """Split data to 1 batch each device."""
    num_ctx = len(ctx_list)
    new_batch = []
    for i, data in enumerate(batch):
        new_data = [x.as_in_context(ctx) for x, ctx in zip(data, ctx_list)]
        new_batch.append(new_data)
    return new_batch


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

# The bases - as global variables to save time
bases = np.load('/home/tutian/dataset/coco_to_voc/coco_all_50_1.npy')

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

def generate_bbox_mask(coefs, bboxes, im_height, im_width):
    # TO the original size
    bboxes[:, 0] *= (im_width / 416.0)
    bboxes[:, 2] *= (im_width / 416.0)
    bboxes[:, 1] *= (im_height / 416.0)
    bboxes[:, 3] *= (im_height / 416.0)
    masks = np.zeros((coefs.shape[0], im_height, im_width), dtype=np.uint8)
    bboxs = np.zeros((coefs.shape[0], 4), dtype=np.int32)
    for i, coef in enumerate(coefs):
        xmin, ymin, xmax, ymax = [int(x) for x in bboxes[i]]
        bboxw = xmax - xmin
        bboxh = ymax - ymin 

        mask_single = np.dot(coef, bases).reshape(64, 64)
        theta = (mask_single.max() + mask_single.min()) / 2
        resized = (cv.resize(mask_single, (bboxw , bboxh)) > theta)

        if (ymin<0):
            resized = resized[-ymin:,:]
            ymin = 0
        if (xmin<0):
            resized = resized[:, -xmin:]
            xmin = 0
        if (xmax > im_width):
            resized = resized[:, : -(xmax - im_width)]
            xmax = im_width
        if (ymax > im_height):
            resized = resized[:-(ymax - im_height), :]
            ymax = im_height
        masks[i, ymin:ymax, xmin:xmax] = resized
        bboxs[i] = [xmin, ymin, bboxw, bboxh]
    return bboxs, masks


def speed_test():
    args = parse_args()
    # context list
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',') if i.strip()]
    ctx = [mx.cpu()] if not ctx else ctx

    # Get net
    net = gcv.model_zoo.get_model(args.network, pretrained=False, pretrained_base=False)
    net.load_parameters(args.pretrained)
    net.set_nms(0.45, 200)
    net.collect_params().reset_ctx(ctx = ctx)

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    # Dataset
    val_dataset = COCOInstance(root='/home/tutian/coco_val2017/', skip_empty=False)
    val_bfn = batchify.Tuple(*[batchify.Append() for _ in range(2)])
    # val_bfn = Tuple(Stack(), Pad(pad_val=-1))
    val_loader = gluon.data.DataLoader(
        val_dataset.transform(YOLO3UsdSegCocoValTransform(416, 416, 50, 'coco')),
        1, False, batchify_fn=val_bfn, last_batch='keep', num_workers=1)

    # Some preparation
    total_time_net = 0
    total_time_post = 0
    total_time_cpu = 0
    total_time_predot = 0
    total_time_dot = 0
    total_time_genmask = 0
    total_time_calculate = 0
    colors = {i: plt.get_cmap('hsv')(i / len(CLASSES)) for i in range(len(CLASSES))}
    img_ids = sorted(val_dataset.coco.getImgIds())

    mx.nd.waitall()
    net.hybridize()
    save_images = False

    # print(image_list_batch)
    with tqdm(total=5000) as pbar:
        for ibt, batch in enumerate(val_loader):
            batch = split_and_load(batch, ctx_list=ctx)
            for x, im_info in zip(*batch):
                # get prediction results
                t1 = time.time()
                ids, scores, bboxes, coefs = net(x)

                t_c0 = time.time()
                mx.nd.waitall()
                t_c1 = time.time()
                t2 = time.time()

                # Post process
                t_cpu0 = time.time()
                bboxes = bboxes.asnumpy()[0]
                ids = ids.asnumpy()[0]
                scores = scores.asnumpy()[0]
                coefs = coefs.asnumpy()[0]
                im_info = im_info.asnumpy()[0]
                t_cpu1 = time.time()
                total_time_cpu += (t_cpu1 - t_cpu0)

                t_cpu0 = time.time()
                im_height, im_width = [int(i) for i in im_info]
                valid = np.where(((ids >= 0) & (scores >= 0.45)))[0]
                ids = ids[valid]
                scores = scores[valid]
                bboxes = bboxes[valid]
                coefs = coefs[valid]
                coefs = coefs * sqrt_var + x_mean
                t_cpu1 = time.time()
                total_time_predot += (t_cpu1 - t_cpu0)

                t_cpu0 = time.time()
                masks = np.dot(coefs, bases)
                t_cpu1 = time.time()
                total_time_dot += (t_cpu1 - t_cpu0)

                t_cpu0 = time.time()
                bboxes, masks = generate_bbox_mask(coefs, bboxes, im_height, im_width)
                t_cpu1 = time.time()
                total_time_genmask += (t_cpu1 - t_cpu0)

                t3 = time.time()
                if ibt >= 500:
                    total_time_net += (t2 - t1)
                    total_time_calculate += (t_c1 - t_c0)
                total_time_post += (t3 - t2)

                # Save the masks
                if save_images:
                    fig = plt.figure(figsize=(10, 10))
                    ax = fig.add_subplot(1, 1, 1)
                    ax.imshow(Image.open('/disk1/data/coco/val2017/' + str(img_ids[ibt]).zfill(12)+'.jpg'))
                    for bbox, mask, idt in zip(bboxes, masks, ids):
                        idt = int(idt)
                        # bbox
                        rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], fill=False,
                                    edgecolor=colors[idt], linewidth=2)
                        ax.add_patch(rect)
                        # mask
                        mask_channel = np.zeros((im_height, im_width, 4))
                        for i in range(4):
                            mask_channel[:,:,i] = mask * colors[idt][i]
                        ax.imshow(mask_channel, alpha=0.5)

                    plt.axis('off')
                    plt.gca().xaxis.set_major_locator(NullLocator())
                    plt.gca().yaxis.set_major_locator(NullLocator())
                    plt.subplots_adjust(top = 0.995, bottom = 0.005, right = 0.995, left = 0.005, hspace = 0, wspace = 0)
                    plt.savefig(os.path.join(args.save_dir, str(img_ids[ibt]).zfill(12)+'.jpg'))
                    plt.close()

            pbar.update(1)

    print("network speed      ", 4500 / total_time_net, "fps")
    print("Sync speed         ", 4500 / total_time_calculate, "fps")
    print("post process speed ", 5000 / total_time_post, "fps")
    print("GPU to CPU speed   ", 5000 / total_time_cpu, "fps")
    print("Pre-dot speed      ", 5000 / total_time_predot, "fps")
    print("np.dot speed       ", 5000 / total_time_dot, "fps")
    print("Gen-Mask speed     ", 5000 / total_time_genmask, "fps")
    print("total speed        ", 5000 / (total_time_net + total_time_post), "fps")

speed_test()
