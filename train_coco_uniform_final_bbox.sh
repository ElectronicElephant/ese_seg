python sbd_train_che_8_1.py --syncbn --network darknet53 --dataset coco \
--batch-size 32 --gpus 4,5,6,7  --num-workers 16 \
--warmup-epochs 1 \
--lr 0.0005 --epochs 201 --lr-decay 0.1  --lr-decay-epoch 160,180 \
--save-prefix ./result_coco_pretrain_uniform_cos_lbsm_ \
--resume pretrain_80_epoch_30.params --start-epoch 31 \
--lr-mode cosine --val_2012 True --label-smooth \
--save-interval 1 --only_bbox True
