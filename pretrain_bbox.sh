python sbd_train_che_8_1.py --syncbn --network darknet53 --dataset coco_pretrain --only_bbox True  \
--batch-size 64 --gpus 2,3 --num-workers 16                                                       \
--warmup-epochs 4 --lr 0.003 --epochs 21  \
--save-prefix ./darknet53_pretrain_
# --resume darknet53_result_each_uniformyolo3_darknet53_voc_0150_0.0000.params --start-epoch 150
