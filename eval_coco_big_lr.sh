for epoch in {0155..0159..1}
do
    scp -P 21600 tutian@202.121.182.216:/home/tutian/ese_seg/result_coco_pretrain_var_tanh_lbsm_yolo3_darknet53_coco_${epoch}_0.0000.params result_coco_pretrain_var_tanh_lbsm_yolo3_darknet53_coco_${epoch}_0.0000.params
    python coco_eval.py --network darknet53  --dataset coco \
    --resume result_coco_pretrain_var_tanh_lbsm_yolo3_darknet53_coco_${epoch}_0.0000.params \
    --save-prefix ./eval_coco_big_lr_ \
    --gpus 1 --batch-size 1 --num-workers 1 --start-epoch ${epoch}
done
