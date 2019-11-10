python coco_eval.py --network darknet53  --dataset coco \
--resume result_coco_pretrain_var_tanh_lbsm_small_lr_yolo3_darknet53_coco_0101_0.0000.params \
--save-prefix ./eval_coco \
--gpus 0 --batch-size 1 --num-workers 1 --start-epoch 101
