for epoch in {36..92..1}
do
    python coco_eval.py --network darknet53  --dataset coco \
    --resume result_coco_pretrain_var_tanh_lbsm_yolo3_darknet53_coco_00${epoch}_0.0000.params \
    --save-prefix ./eval_coco_ \
    --gpus 0 --batch-size 1 --num-workers 1 --start-epoch ${epoch}
done
