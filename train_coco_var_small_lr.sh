python sbd_train_che_8_1.py --syncbn --network darknet53 --dataset coco \
--batch-size 32 --gpus 4,5,6,7  --num-workers 16 \
--warmup-epochs 1 \
--lr 0.0005 --epochs 140 --lr-decay 0.1  --lr-decay-epoch 100,120 \
--save-prefix ./result_coco_pretrain_var_tanh_lbsm_small_lr_ \
--resume result_coco_pretrain_var_tanh_lbsm_yolo3_darknet53_coco_0092_0.0000.params --start-epoch 100 \
--lr-mode cosine --val_2012 True --label-smooth \
--save-interval 1
