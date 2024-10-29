import open3d as o3d
import os
import numpy as np
import pickle


log_root = '/data/huyb/cvpr-2024/r3d3/logs/nuscenes/eval_predictions'
data_root = '/data/huyb/cvpr-2024/data/ss3dm/DATA'

for town_name in ['Town01', 'Town02', 'Town03', 'Town04', 'Town05', 'Town06', 'Town07', 'Town10']:
# for town_name in ['Town02']:
    for seq_name in os.listdir(os.path.join(log_root, town_name)):
    # for seq_name in ['Town02_260']:
        print("Processing {} {} ...".format(town_name, seq_name))
        if os.path.isdir(os.path.join(log_root, town_name, seq_name)):
            if os.path.exists(os.path.join(log_root, town_name, seq_name, 'mesh.ply')):
                print("Mesh already exists, skip ...")
                continue
            
            pcd_path = os.path.join(log_root, town_name, seq_name, 'point_cloud_from_depth_fusion.ply')
            pcd = o3d.io.read_point_cloud(pcd_path)
            ## check the normal property
            pcd.estimate_normals()
            
            # seq_dir = os.path.join(data_root, town_name, seq_name)
            # meta_path = os.path.join(seq_dir, 'scenario.pt')
            # ## load the meta data with pickle
            # with open(meta_path, 'rb') as f:
            #     meta_data = pickle.load(f)
            # pose_all = []
            
            # for observer in meta_data['observers'].keys():
            #     if meta_data['observers'][observer]['class_name'] == 'Camera':
            #         for i in range(meta_data['metas']['n_frames']):
            #             pose_all.append(meta_data['observers'][observer]['data']['c2w'][i].astype(np.float32))
            
            # print("Orienting the normals towards the camera locations ...")
            # for pose in pose_all:
            #     cam_loc = pose[:3, 3]
            #     pcd.orient_normals_towards_camera_location(cam_loc)
            
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
            
            print('remove low density vertices')
            vertices_to_remove = densities < np.quantile(densities, 0.01)
            mesh.remove_vertices_by_mask(vertices_to_remove)
            
            o3d.io.write_triangle_mesh(os.path.join(log_root, town_name, seq_name, 'mesh.ply'), mesh)