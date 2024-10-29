import os
import csv
import torch
from tqdm import tqdm
from lietorch import SE3

from r3d3.r3d3 import R3D3
from r3d3.modules.completion import DepthCompletion

from vidar.utils.write import write_npz
from vidar.utils.config import read_config, get_folder_name, load_class, Config, cfg_has
from vidar.utils.networks import load_checkpoint
from vidar.utils.config import recursive_assignment

from evaluation_utils.dataloader_wrapper import setup_dataloaders, SceneIterator, SampleIterator
from vidar.utils.setup import setup_metrics
from r3d3.utils import pose_matrix_to_quaternion

import pickle
import cv2
import numpy as np

def load_completion_network(cfg: Config) -> DepthCompletion:
    """ Loads completion network with vidar framework
    Args:
        cfg: Completion network config
    Returns:
        Completion network with loaded checkpoint if path is provided
    """
    folder, name = get_folder_name(cfg.file, 'networks', root=cfg_has(cfg, 'root', 'vidar/arch'))
    network = load_class(name, folder)(cfg)
    recursive_assignment(network, cfg, 'networks', verbose=True)
    if cfg_has(cfg, 'checkpoint'):
        network = load_checkpoint(
            network,
            cfg.checkpoint,
            strict=False,
            verbose=True,
            prefix='completion'
        )
    return network.networks.cuda().eval()


class Evaluator:
    """ R3D3 evaluation module
    """
    def __init__(self, args):
        """
        Args:
            args: Arguments from argparser containing
                config: Path to vidar-config file (yaml) containing configurations for dataset, metrics (optional) and
                    completion network (optional)
                R3D3-args: As described by R3D3
                training_data_path: Path to directory where R3D3 training samples should be stored. If None, training
                    samples are not stored. Default - None
                prediction_data_path: Path to directory where R3D3 predictions should be stored. If None, predictions
                    are not stored. Default - None
        """
        self.args = args
        self.cfg = read_config(self.args.config)
        self.dataloaders = setup_dataloaders(self.cfg.datasets, n_workers=args.n_workers)
        self.completion_network = None
        if cfg_has(self.cfg, 'networks') and cfg_has(self.cfg.networks, 'completion'):
            self.completion_network = load_completion_network(
                self.cfg.networks.completion
            )
        self.metrics = {}
        if cfg_has(self.cfg, 'evaluation'):
            self.metrics = setup_metrics(self.cfg.evaluation)
        self.depth_results = []
        self.trajectory_results = []
        self.confidence_stats = []

        self.training_data_path = args.training_data_path
        self.prediction_data_path = args.prediction_data_path

    def eval_scene(self, scene: str, n_cams: int, sample_iterator: SampleIterator) -> None:
        """ Evaluates a given scene by 1. Initializing R3D3, 2. Running R3D3 for each frame, 3. Terminate R3D3
        Args:
            scene: Current scene to be processed
            n_cams: Number of cameras
            sample_iterator: Iterator yielding samples from each timestep in chronological order
        """
        scene_depth_results = []
        pred_poses, gt_poses = [], []
        depth_res_idx = 0
        pred_pose_list = []
        pose_keys = ['x', 'y', 'z', 'r', 'i', 'j', 'k']

        r3d3 = R3D3(
            completion_net=self.completion_network,
            n_cams=n_cams,
            **{key.replace("r3d3_", ""): val for key, val in vars(self.args).items() if key.startswith("r3d3_")}
        )

        for timestamp, sample in enumerate(tqdm(sample_iterator, desc='Sample', position=0, leave=True)):
            # image = sample['rgb'][0][0,0].permute(1,2,0).numpy()
            # cv2.imwrite('/data/huyb/cvpr-2024/r3d3/logs/rgb.png', (image*255).astype(np.uint8))
            # cv2.imwrite('/data/huyb/cvpr-2024/r3d3/logs/rgb_reverse.png', (image*255).astype(np.uint8)[:,:,::-1])
            # # save a sample to pickle file
            # if not os.path.exists('/data/huyb/cvpr-2024/r3d3/logs/sample.pkl'):
            #     import pickle
            #     with open('/data/huyb/cvpr-2024/r3d3/logs/sample.pkl', 'wb') as f:
            #         pickle.dump(sample, f)
            #     # save a sample to txt file for visualization
            # with open('/data/huyb/cvpr-2024/r3d3/logs/sample.txt', 'w') as f:
            #     f.write(str(sample))
                
            pose = SE3(pose_matrix_to_quaternion(sample['pose'][0][0]).cuda())
            pose = pose.inv()
            pose_rel = (pose * pose[0:1].inv())

            intrinsics = sample['intrinsics'][0][0, :, [0, 1, 0, 1], [0, 1, 2, 2]]
            is_keyframe = 'depth' in sample and sample['depth'][0].max() > 0.

            output = r3d3.track(
                tstamp=float(timestamp),
                image=(sample['rgb'][0][0] * 255).type(torch.uint8).cuda(),
                intrinsics=intrinsics.cuda(),
                mask=(sample['mask'][0][0, :, 0] > 0).cuda() if 'mask' in sample else None,
                pose_rel=pose_rel.data
            )

            output = {key: data.cpu() if torch.is_tensor(data) else data for key, data in output.items()}
            pred_pose = None
            if output['pose'] is not None:
                pred_pose = (pose_rel.cpu() * SE3(output['pose'][None])).inv()
                pred_pose_list.append(
                    {'filename': sample['filename'][0][0][0], **dict(zip(pose_keys, pred_pose[0].data.numpy()))}
                )
                pred_poses.append(pred_pose.matrix())
                gt_poses.append(sample['pose'][0][0])
            if output['disp_up'] is not None and 'depth' in self.metrics and is_keyframe:
                results = {
                    'ds_idx': sample['idx'][0],
                    'sc_idx': torch.tensor(depth_res_idx, dtype=sample['idx'][0].dtype, device=sample['idx'][0].device),
                    'scene': scene
                }
                results.update({key: metric[0] for key, metric in self.metrics['depth'].evaluate(
                    batch=sample,
                    output={'depth': {0: [1 / output['disp_up'].unsqueeze(0).unsqueeze(2)]}}
                )[0].items()})
                scene_depth_results.append(results)
                depth_res_idx += 1

            if self.training_data_path is not None and pred_pose is not None:
                for cam, filename in enumerate(sample['filename'][0]):
                    write_npz(
                        os.path.join(
                            self.training_data_path,
                            filename[0].replace('rgb', 'r3d3').replace('CAM_', 'R3D3_') + '.npz'
                        ),
                        {
                            'intrinsics': intrinsics[cam].numpy(),
                            'pose': pred_pose[cam].data.numpy(),
                            'disp': output['disp'][cam].numpy()[None],
                            'disp_up': output['disp_up'][cam].numpy()[None],
                            'conf': output['conf'][cam].numpy()[None],
                        }
                    )
            if self.prediction_data_path is not None and output['disp_up'] is not None and is_keyframe:
                for cam, filename in enumerate(sample['filename'][0]):
                    write_npz(
                        os.path.join(
                            self.prediction_data_path,
                            filename[0] + '_depth(0)_pred.npz'
                        ),
                        {
                            'depth': (1.0 / output['disp_up'][cam].numpy()),
                            'intrinsics': intrinsics[cam].numpy(),
                            'd_info': 'r3d3_depth',
                            't': float(timestamp)
                        }
                    )

        # Terminate
        del r3d3
        torch.cuda.empty_cache()

        if self.prediction_data_path is not None and len(pred_pose_list) > 0:
            pose_dir = os.path.join(self.prediction_data_path, 'poses')
            if not os.path.exists(pose_dir):
                os.makedirs(pose_dir)
            with open(os.path.join(pose_dir, f'{scene}_poses.csv'), 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=pred_pose_list[0].keys())
                writer.writeheader()
                writer.writerows(pred_pose_list)
        self.depth_results.extend([{'idx': res['ds_idx'], **res} for res in scene_depth_results])
        if 'depth' in self.metrics:
            reduced_data = self.metrics['depth'].reduce_metrics(
                [[{'idx': res['sc_idx'], **res} for res in scene_depth_results]],
                [scene_depth_results], strict=False
            )
            self.metrics['depth'].print(reduced_data, [f'scene-{scene}'])
        if 'trajectory' in self.metrics and len(gt_poses) >= 2:
            results = {'scene': scene}
            results.update(self.metrics['trajectory'].evaluate(
                batch={'trajectory': {0: torch.stack(gt_poses)}},
                output={'trajectory': {0: [torch.stack(pred_poses)]}}
            )[0])
            self.trajectory_results.append({'idx': torch.tensor(len(self.trajectory_results)), **results})
            reduced_data = self.metrics['trajectory'].reduce_metrics(
                [[{'idx': torch.tensor(0), **results}]],
                [[results]], strict=False
            )
            self.metrics['trajectory'].print(reduced_data, [f'scene-{scene}'])

    
    def eval_ss3dm_scene(self, town_name, seq_name, n_cams = 6):
        output_dir = os.path.join(self.args.prediction_data_path, town_name, seq_name)
        if os.path.exists(output_dir):
            print("Scene {} already exists. Skipping.".format(seq_name))
            return
        
        ss3dm_root = '/data/huyb/cvpr-2024/data/ss3dm/DATA'
        seq_dir = os.path.join(ss3dm_root, town_name, seq_name)

        # import pdb; pdb.set_trace()
        target_size = [int(384/1080*1920/32)*32,384] # [W, H]
        orig_size = [1920, 1080]
        self.args.r3d3_image_size = [target_size[1], target_size[0]] # [H, W]
        r3d3 = R3D3(
            completion_net=self.completion_network,
            n_cams=n_cams,
            **{key.replace("r3d3_", ""): val for key, val in vars(self.args).items() if key.startswith("r3d3_")}
        )
        
        ## prepare data sample for each timestep
        meta_path = os.path.join(seq_dir, 'scenario.pt')
        with open(meta_path, 'rb') as f:
            meta_data = pickle.load(f)
        
        timestep_list = meta_data['observers']['ego_car']['data']['timestamp']
        
        for sample_idx, timestamp in enumerate(timestep_list):
            # prepare data for this timestep.
            sample = {}
            sample['idx'] = torch.tensor([sample_idx])
            sample['frame_idx'] = {}
            sample['frame_idx'][0] = [torch.tensor([sample_idx])] * n_cams
            sample['cam'] = []
            
            for name in meta_data['observers'].keys():
                if 'camera' in name:
                    sample['cam'].append([name])
            
            sample['filename'] = {}
            sample['filename'][0] = []
            for cam in sample['cam']:
                sample['filename'][0].append([os.path.join(town_name, seq_name, 'images', cam[0], '%08d.jpg'%sample_idx)])
            
            sample['rgb'] = {}
            # torch.Size([1, n_cams, 3, H, W])
            # max = 1.0, min = 0.0
            # dtype = float32
            all_rgbs = []
            for cam in sample['cam']:
                rgb_path = os.path.join(seq_dir, 'images', cam[0], '%08d.jpg'%sample_idx)
                # We need to reverse the image channels.
                rgb = cv2.imread(rgb_path)
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                rgb = cv2.resize(rgb, (target_size[0], target_size[1]))
                rgb = rgb.astype(np.float32) / 255.0
                rgb = torch.tensor(rgb).permute(2, 0, 1).unsqueeze(0)
                all_rgbs.append(rgb)
            all_rgbs = torch.cat(all_rgbs, dim=0)
            sample['rgb'][0] = all_rgbs[None,...]    

            sample['intrinsics'] = {}
            # torch.Size([1, n_cams, 3, 3])
            intrinsics = []
            orig_intrinsics = []
            for cam in sample['cam']:
                orig_intrinsic = meta_data['observers'][cam[0]]['data']['intr'][sample_idx]
                new_intrinsic = orig_intrinsic.copy()
                # import pdb; pdb.set_trace()
                new_intrinsic[0,0] = orig_intrinsic[0,0] / orig_size[0] * target_size[0]
                new_intrinsic[1,1] = orig_intrinsic[1,1] / orig_size[1] * target_size[1]
                new_intrinsic[0,2] = orig_intrinsic[0,2] / orig_size[0] * target_size[0]
                new_intrinsic[1,2] = orig_intrinsic[1,2] / orig_size[1] * target_size[1]
                
                intrinsics.append(new_intrinsic)
                orig_intrinsics.append(orig_intrinsic)
                
            sample['intrinsics'][0] = torch.tensor(intrinsics).unsqueeze(0)
            sample['raw_intrinsics'] = {}
            sample['raw_intrinsics'][0] = torch.tensor(orig_intrinsics).unsqueeze(0)
            
            
            sample['pose'] = {}
            # torch.Size([1, n_cams, 4, 4])
            poses = []
            for cam in sample['cam']:
                poses.append(meta_data['observers'][cam[0]]['data']['c2w'][sample_idx])
            
            sample['pose'][0] = torch.tensor(poses).unsqueeze(0)
            
            sample['depth'] = {}
            # torch.Size([1, n_cams, 1, H, W])
            # image = cv2.imread(os.path.join(args.depth_gt_dir, observer, '%.8d.png' % (i)), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            # image = image.astype(np.float32) / 65535 * 1000.0
            depth_images = []
            for cam in sample['cam']:
                depth_path = os.path.join(seq_dir, 'depth_gts', cam[0], '%08d.png'%sample_idx)
                depth = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
                depth = depth.astype(np.float32) / 65535 * 1000.0
                depth = torch.tensor(depth).unsqueeze(0).unsqueeze(0)
                depth_images.append(depth)
            
            sample['depth'][0] = torch.cat(depth_images, dim=1)
            
            sample['mask'] = {}
            # torch.Size([1, n_cams, 3, H, W])
            # all zeros
            # import pdb; pdb.set_trace()
            sample['mask'][0] = torch.zeros((1, n_cams, 3, all_rgbs.shape[2], all_rgbs.shape[3]))
            
            
            sample['scene'] = seq_name
            
            
            
            ## The sample data is well-prepared.
            pose = SE3(pose_matrix_to_quaternion(sample['pose'][0][0]).cuda())
            pose = pose.inv()
            pose_rel = (pose * pose[0:1].inv())

            intrinsics = sample['intrinsics'][0][0, :, [0, 1, 0, 1], [0, 1, 2, 2]]
            orig_intrinsics = sample['raw_intrinsics'][0][0, :, [0, 1, 0, 1], [0, 1, 2, 2]]
            is_keyframe = 'depth' in sample and sample['depth'][0].max() > 0.

            output = r3d3.track(
                tstamp=float(timestamp),
                image=(sample['rgb'][0][0] * 255).type(torch.uint8).cuda(),
                intrinsics=intrinsics.cuda(),
                mask=(sample['mask'][0][0, :, 0] > 0).cuda() if 'mask' in sample else None,
                pose_rel=pose_rel.data
            )

            # import pdb; pdb.set_trace()
            
            output = {key: data.cpu() if torch.is_tensor(data) else data for key, data in output.items()}

            if self.prediction_data_path is not None and output['disp_up'] is not None and is_keyframe:
                for cam, filename in enumerate(sample['filename'][0]):
                    depth_data = (1.0 / output['disp_up'][cam].numpy())
                    orig_size_depth_data = cv2.resize(depth_data, (orig_size[0], orig_size[1]))
                    # import pdb; pdb.set_trace()
                    write_npz(
                        os.path.join(
                            self.prediction_data_path,
                            filename[0] + '_depth(0)_pred.npz'
                        ),
                        {
                            'depth': orig_size_depth_data,
                            'intrinsics': orig_intrinsics[cam].numpy(),
                            'd_info': 'r3d3_depth',
                            't': float(timestamp)
                        }
                    )

        # import pdb; pdb.set_trace()
        # Terminate
        del r3d3
        torch.cuda.empty_cache()
            
    
    def eval_datasets_ss3dm(self) -> None:
        data_root = '/data/huyb/cvpr-2024/data/ss3dm/DATA'

        for town_name in os.listdir(data_root):
            if os.path.isdir(os.path.join(data_root, town_name)):
                town_dir = os.path.join(data_root, town_name)
                for seq_name in os.listdir(town_dir):
                    if os.path.isdir(os.path.join(town_dir, seq_name)):
                        print("Evaluating scene: {}".format(seq_name))
                        self.eval_ss3dm_scene(town_name, seq_name)
                        # exit(0)
        
    def eval_datasets(self) -> None:
        """ Evaluates datasets consisting of multiple scenes
        """
        for dataloader in tqdm(self.dataloaders, desc='Datasets', position=2, leave=True):
            n_cams = len(dataloader.dataset.cameras)
            # import pdb; pdb.set_trace()
            pbar = tqdm(SceneIterator(dataloader), desc='Scenes', position=1, leave=True)
            # import pdb; pdb.set_trace()
            for scene, sample_iterator in pbar:
                pbar.set_postfix_str("Processing Scene - {}".format(scene))
                pbar.refresh()
                self.eval_scene(scene, n_cams, sample_iterator)

            if 'depth' in self.metrics and len(self.depth_results) > 0:
                reduced_data = self.metrics['depth'].reduce_metrics(
                    [self.depth_results],
                    [dataloader.dataset], strict=False
                )
                self.metrics['depth'].print(reduced_data, ['Overall'])
            if 'trajectory' in self.metrics and len(self.trajectory_results) > 0:
                reduced_data = self.metrics['trajectory'].reduce_metrics(
                    [self.trajectory_results],
                    [self.trajectory_results], strict=False
                )
                self.metrics['trajectory'].print(reduced_data, ['Overall'])
            # Use to evaluate confidence statistics => Can find scenes where metric was not recovered / failed
            # if len(self.confidence_stats) > 0:
            #     confidence_stats_summary = {}
            #     for element in self.confidence_stats:
            #         scene = element['scene']
            #         if scene not in confidence_stats_summary:
            #             stats_keys = [key for key in element.keys() if key is not scene]
            #             confidence_stats_summary[scene] = {k: [] for k in stats_keys if k not in ['scene', 'idx']}
            #         for key in confidence_stats_summary[scene]:
            #             confidence_stats_summary[scene][key].append(element[key])
            #     confidence_stats_summary = {
            #         scene: {key: sum(val) / len(val) for key, val in stats.items()}
            #         for scene, stats in confidence_stats_summary.items()
            #     }
            #     import csv
            #     with open('confidence_stats.csv', 'w', newline='') as output_file:
            #         dict_writer = csv.DictWriter(output_file, list(self.confidence_stats[0].keys()))
            #         dict_writer.writeheader()
            #         dict_writer.writerows(self.confidence_stats)
