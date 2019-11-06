python sbd_train_che_8_1.py --syncbn --network darknet53 --dataset coco \
--batch-size 32 --gpus 0,1,2,3  --num-workers 16 \
--warmup-epochs 0 --lr 0.0005 --epochs 201 --lr-decay 0.1  --lr-decay-epoch 160,180 \
--save-prefix ./coco_pretrain_80_ --save-interval 1 \
--resume pretrain_80_epoch_30.params --start-epoch 31 \
--lr-mode cosine --val_2012 True \
--only_bbox True \
--label-smooth \
