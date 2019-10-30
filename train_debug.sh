python sbd_train_che_8_1.py --syncbn --network darknet53 --dataset voc \
--batch-size 2 --gpus 3 --num-workers 4 \
--warmup-epochs 4 --lr 0.0015 --epochs 201 --lr-decay 0.1  --lr-decay-epoch 160,180 \
--save-prefix ./debug_ \
--resume darknet53_result_pretrain_bases20_var_tanh_smooth_test2_yolo3_darknet53_voc_0070_0.2502.params --start-epoch 70 \
--lr-mode cosine --val_2012 True --label-smooth \
