import os
import numpy as np
import cv2

depth_dir = '/data/huyb/cvpr-2024/r3d3/logs/nuscenes/eval_predictions/Town02'
vis_dir = '/data/huyb/cvpr-2024/r3d3/logs/nuscenes/depth_visualizations/Town02'

if not os.path.exists(vis_dir):
    os.makedirs(vis_dir)

for root, dirs, files in os.walk(depth_dir):
    for file in files:
        if file.endswith('.npz'):
            depth_path = os.path.join(root, file)
            vis_path = depth_path.replace(depth_dir, vis_dir).replace('.npz', '.png')
            if not os.path.exists(os.path.dirname(vis_path)):
                os.makedirs(os.path.dirname(vis_path))
            # import pdb; pdb.set_trace()
            depth = np.load(depth_path)['depth']
            depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            cv2.imwrite(vis_path, depth)

# depth_dir = '/data/huyb/cvpr-2024/r3d3/logs/ddad_tiny_nuscenes/eval_predictions'
# vis_dir = '/data/huyb/cvpr-2024/r3d3/logs/ddad_tiny_nuscenes/depth_visualizations'

# if not os.path.exists(vis_dir):
#     os.makedirs(vis_dir)

# for root, dirs, files in os.walk(depth_dir):
#     for file in files:
#         if file.endswith('.npz'):
#             depth_path = os.path.join(root, file)
#             vis_path = depth_path.replace(depth_dir, vis_dir).replace('.npz', '.png')
#             if not os.path.exists(os.path.dirname(vis_path)):
#                 os.makedirs(os.path.dirname(vis_path))
#             # import pdb; pdb.set_trace()
#             depth = np.load(depth_path)['depth']
#             depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
#             cv2.imwrite(vis_path, depth)
