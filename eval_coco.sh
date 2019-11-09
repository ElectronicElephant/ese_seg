python coco_eval.py --network darknet53  --dataset coco \
--resume result_coco_pretrain_var_tanh_lbsm_yolo3_darknet53_coco_0064_0.0000.params \
--save-prefix ./eval_coco \
--gpus 2 --batch-size 1 --num-workers 1
