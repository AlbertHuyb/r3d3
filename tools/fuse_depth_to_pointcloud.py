import os
import pickle
import cv2
import numpy as np
import open3d as o3d
from tqdm import tqdm

def depth_to_points(depth, intrinsic_mat, pose, h=1080, w=1920):
    grids = np.meshgrid(np.arange(w), np.arange(h))
    grids = np.stack(grids, axis=-1)
    grids = grids.reshape(-1, 2)
    grids = np.concatenate([grids, np.ones((grids.shape[0], 1))], axis=-1)
    grids = grids.astype(np.float32)
    grids = np.matmul(np.linalg.inv(intrinsic_mat), grids.T).T

    camera_points = grids[depth.reshape(-1, 1)[:,0]<500, :] * depth.reshape(-1, 1)[depth.reshape(-1, 1)[:,0]<500, :]
    camera_points = np.concatenate([camera_points, np.ones((camera_points.shape[0], 1))], axis=-1)
    
    world_points = np.matmul(pose, camera_points.T).T
    
    return world_points[:, :3]
    
    
    
def depth_fusion(poses_all, depths_all, masks_all, intrinsic_mat, h=1080, w=1920):
    all_points_o3d = o3d.geometry.PointCloud()
    idx = 0
    for pose, depth, mask in tqdm(zip(poses_all, depths_all, masks_all)):
        pose_np = pose
        depth_np = depth
        points = depth_to_points(depth_np, intrinsic_mat, pose_np, h=1080, w=1920)
        points = points[mask.reshape(-1, 1)[:,0], :]
        
        # import pdb; pdb.set_trace()
        
        # append the points to the all_points
        points_o3d = o3d.geometry.PointCloud()
        points_o3d.points = o3d.utility.Vector3dVector(points)
        points_o3d = points_o3d.voxel_down_sample(voxel_size=0.025)
        all_points_o3d += points_o3d
        
        # downsample the points with open3d
        # import pdb; pdb.set_trace()
        if idx%5==0:
            all_points_o3d = all_points_o3d.voxel_down_sample(voxel_size=0.25)
        
        idx += 1
        # all_points.append(points)
    
    # all_points_o3d = o3d.geometry.PointCloud()
    # all_points_o3d.points = o3d.utility.Vector3dVector(np.concatenate(all_points, axis=0))
    # import pdb; pdb.set_trace()
    all_points_o3d = all_points_o3d.voxel_down_sample(voxel_size=0.025)
    
    return all_points_o3d


data_root = '/data/huyb/cvpr-2024/data/ss3dm/DATA'

r3d3_exp_root = '/data/huyb/cvpr-2024/r3d3/logs/nuscenes/eval_predictions'

for town_name in os.listdir(data_root):
# for town_name in ['Town02']:
    if os.path.isdir(os.path.join(data_root, town_name)):
        town_dir = os.path.join(data_root, town_name)
        for seq_name in os.listdir(town_dir):
        # for seq_name in ['Town02_260']:
            print("Fusing point cloud for town: {}, seq: {}".format(town_name, seq_name))
            if os.path.isdir(os.path.join(town_dir, seq_name)):
                seq_dir = os.path.join(town_dir, seq_name)
                
                if os.path.exists(os.path.join(r3d3_exp_root, town_name, seq_name, 'point_cloud_from_depth_fusion.ply')):
                    print("Point cloud already exists, skip ...")
                    continue
                
                meta_path = os.path.join(seq_dir, 'scenario.pt')
                with open(meta_path, 'rb') as f:
                    meta_data = pickle.load(f)
                
                r3d3_exp_dir = os.path.join(r3d3_exp_root, town_name, seq_name)
                
                pose_all = []
                depth_selected = []
                mask_selected = []
                pose_selected = []
                
                print("Loading the poses and depth images ...")  
                for observer in meta_data['observers'].keys():
                    if meta_data['observers'][observer]['class_name'] == 'Camera':
                        
                        for i in range(meta_data['metas']['n_frames']):
                            pose_all.append(meta_data['observers'][observer]['data']['c2w'][i].astype(np.float32))
                            
                            intrinsic_mat = meta_data['observers'][observer]['data']['intr'][i].astype(np.float32)
                            
                            # import pdb; pdb.set_trace()
                            if len(pose_all) % 13 == 0:
                            # if len(pose_all) < 2:
                                image = np.load(os.path.join(r3d3_exp_dir, 'images', observer, '%.8d.jpg_depth(0)_pred.npz' % (i)), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)['depth']
                                depth_selected.append(image)
                                pose_selected.append(meta_data['observers'][observer]['data']['c2w'][i].astype(np.float32))
                                
                                ## need to add mask to the depth image, in order to remove the sky points.
                                
                                this_mask_image = np.load(os.path.join(data_root, town_name, seq_name, 'masks', observer, '%.8d.npz' % (i)))['arr_0']
                                valid_mask = this_mask_image!=11
                                mask_selected.append(valid_mask)
                                
                                # for semantic_label in range(25):
                                #     mask = this_mask_image==semantic_label
                                #     print(town_name, seq_name, observer, i)
                                #     cv2.imwrite(os.path.join(r3d3_exp_dir, '%s-%.8d-%d.jpg_depth(0)_pred_sky_mask.png' % (observer, i, semantic_label)), mask.astype(np.uint8)*255)

                                # import pdb; pdb.set_trace()
                                
                                
                point_cloud_from_depth_fusion = depth_fusion(pose_selected, depth_selected, mask_selected, intrinsic_mat)
                
                o3d.io.write_point_cloud(os.path.join(r3d3_exp_dir, 'point_cloud_from_depth_fusion.ply'), point_cloud_from_depth_fusion)
                
                