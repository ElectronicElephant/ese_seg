python train_tinyyolo.py --syncbn --network tiny_darknet --dataset voc --val_2012 True \
--batch-size 8 --gpus 0,1  --num-workers 8 \
--warmup-epochs 1 \
--lr 0.0005 --epochs 201 --lr-decay 0.1  --lr-decay-epoch 160,180 \
--save-prefix ./tinyyolo/tiny_yolo_ \
--label-smooth \
--save-interval 1 \
--log-interval 1
