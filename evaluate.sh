# python evaluate.py \
#     --config configs/evaluation/r3d3/r3d3_evaluation_ddad_tiny_nuscenes.yaml \
#     --r3d3_weights data/models/r3d3/r3d3_finetuned.ckpt \
#     --r3d3_image_size 384 640 \
#     --r3d3_init_motion_only \
#     --r3d3_n_edges_max=184 \
#     --prediction_data_path /data/huyb/cvpr-2024/r3d3/logs/ddad_tiny_nuscenes/eval_predictions \
    

# python evaluate.py \
#     --config configs/evaluation/r3d3/r3d3_evaluation_nuscenes.yaml \
#     --r3d3_weights data/models/r3d3/r3d3_finetuned.ckpt \
#     --r3d3_image_size 448 768 \
#     --r3d3_init_motion_only \
#     --r3d3_dt_inter=0 \
#     --r3d3_n_edges_max=184 \
#     --prediction_data_path /data/huyb/cvpr-2024/r3d3/logs/nuscenes/eval_predictions \


python evaluate.py \
    --config configs/evaluation/r3d3/r3d3_evaluation_ddad_tiny.yaml \
    --r3d3_weights data/models/r3d3/r3d3_finetuned.ckpt \
    --r3d3_image_size 384 640 \
    --r3d3_init_motion_only \
    --r3d3_n_edges_max=184 \
    --prediction_data_path /data/huyb/cvpr-2024/r3d3/logs/ddad_tiny/eval_predictions \