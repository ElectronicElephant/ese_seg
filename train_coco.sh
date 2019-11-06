python sbd_train_che_8_1.py --syncbn --network darknet53 --dataset coco \
--batch-size 32 --gpus 4,5,6,7  --num-workers 32 \
--warmup-epochs 4 --lr 0.001 --epochs 201 --lr-decay 0.1  --lr-decay-epoch 160,180 \
--save-prefix ./darknet53_result_coco \
--lr-mode cosine --val_2012 True --label-smooth \
