import os
import glob
import itertools
import collections

import torch
import smplx
import trimesh
import numpy as np
import torch.nn.functional as F

from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes

from leap.tools.libmesh import check_mesh_contains

from coap import attach_coap

class SMPLDataset(torch.utils.data.Dataset):

    @torch.no_grad()
    def __init__(self, smpl_cfg, amass_data, **data_cfg):
        super().__init__()
        self.smpl_cfg = smpl_cfg
        self.amass_data = amass_data

        self.data_keys = list(amass_data.keys())
        self.smplx_body = attach_coap(smplx.create(**smpl_cfg), pretrained=False)
        self.faces = self.smplx_body.faces.copy()
        
        self.points_sigma = 0.01
        self.points_uniform_ratio = 0.5
        self.n_points = data_cfg.get('n_points', 256)

    @classmethod
    @torch.no_grad()
    def from_config(cls, smpl_cfg, data_cfg, split='train'):
        # load data
        gender = smpl_cfg['gender']
        smpl_body = smplx.create(**smpl_cfg)
        model_type = smpl_cfg['model_type']
        num_body_joints = smpl_body.NUM_BODY_JOINTS

        data_root = data_cfg['data_root'] 
        datasets = data_cfg[split]['datasets']
        select_every = data_cfg[split].get('select_every', 1)

        dataset = collections.defaultdict(list)
        for ds in datasets:
            subject_dirs = [s_dir for s_dir in sorted(glob.glob(os.path.join(data_root, ds, '*'))) if os.path.isdir(s_dir)]
            for subject_dir in subject_dirs:
                seq_paths = [sn for sn in glob.glob(os.path.join(subject_dir, '*.npz')) if not sn.endswith('shape.npz') and not sn.endswith('neutral_stagei.npz')]
                for seq_path in seq_paths:
                    seq_sample = np.load(seq_path, allow_pickle=True)
                    pose = seq_sample['poses'][::select_every]
                    betas = seq_sample['betas'].reshape((1, -1))[:, :smpl_body.num_betas]
                    n_frames = pose.shape[0]

                    dataset['betas'].append(betas.repeat(n_frames, axis=0))
                    dataset['global_orient'].append(pose[:, :3])
                    dataset['body_pose'].append(pose[:, 3:3+num_body_joints*3])

                    dataset['global_orient_init'].append(pose[:1, :3].repeat(n_frames, axis=0))
                    seq_name = os.path.join(os.path.basename(subject_dir), os.path.splitext(os.path.basename(seq_path))[0])
                    dataset['seq_names'].append([seq_name]*n_frames)
                    dataset['frame_ids'].append(list(map(lambda x: f'{x:06d}', list(range(seq_sample['poses'].shape[0]))[::select_every])))

                    if model_type == 'smplx' or model_type == 'smplh':
                        b_ind = 3+num_body_joints*3
                        dataset['left_hand_pose'].append(pose[:, b_ind:b_ind+45])
                        dataset['right_hand_pose'].append(pose[:, b_ind+45:b_ind+2*45])
                    elif model_type == 'smpl':  # flatten hands for smpl
                        dataset['body_pose'][-1][:, -6:] = 0

        data = {}
        for key, val in dataset.items():
            if isinstance(val[0], np.ndarray):
                data[key] = torch.from_numpy(np.concatenate(val, axis=0)).float()
            else:
                data[key] = list(itertools.chain.from_iterable(val))

        return SMPLDataset(smpl_cfg, data)

    def sample_points(self, smpl_output):
        bone_trans = self.smplx_body.coap.compute_bone_trans(smpl_output.full_pose, smpl_output.joints)
        bbox_min, bbox_max = self.smplx_body.coap.get_bbox_bounds(smpl_output.vertices, bone_trans)  # (B, K, 1, 3) [can space]
        n_parts = bbox_max.shape[1]

        #### Sample points inside local boxes
        n_points_uniform = int(self.n_points * self.points_uniform_ratio)
        n_points_surface = self.n_points - n_points_uniform

        bbox_size = (bbox_max - bbox_min).abs()*self.smplx_body.coap.bbox_padding - 1e-3  # (B,K,1,3)
        bbox_center = (bbox_min + bbox_max) * 0.5
        bb_min = (bbox_center - bbox_size*0.5)  # to account for padding
        
        uniform_points = bb_min + torch.rand((1, n_parts, n_points_uniform, 3)) * bbox_size  # [0,bs] (B,K,N,3)

        # project points to the posed space
        abs_transforms = torch.inverse(bone_trans)  # B,K,4,4
        uniform_points = (abs_transforms.reshape(1, n_parts, 1, 4, 4).repeat(1, 1, n_points_uniform, 1, 1) @ F.pad(uniform_points, [0, 1], "constant", 1.0).unsqueeze(-1))[..., :3, 0]

        #### Sample surface points
        meshes = Meshes(smpl_output.vertices.float().expand(n_parts, -1, -1), 
            self.smplx_body.coap.get_tight_face_tensor())
        surface_points = sample_points_from_meshes(meshes, num_samples=n_points_surface)
        surface_points += torch.from_numpy(np.random.normal(scale=self.points_sigma, size=surface_points.shape))
        surface_points = surface_points.reshape((1, n_parts, -1, 3))

        points = torch.cat((uniform_points, surface_points), dim=-2).float()  # B,K,n_points,3

        #### Check occupancy
        points = points.reshape(-1, 3).numpy()
        mesh = trimesh.Trimesh(smpl_output.vertices.squeeze().numpy(), self.faces, process=False)
        gt_occ = check_mesh_contains(mesh, points).astype(np.float32)
        return dict(points=points, gt_occ=gt_occ)


    @torch.no_grad()
    def __getitem__(self, idx):
        smpl_data = {key: self.amass_data[key][idx:idx+1] for key in self.data_keys}
        smpl_output = self.smplx_body(**smpl_data, return_verts=True, return_full_pose=True)  # smpl fwd pass
        smpl_data = {key: val.squeeze(0) if torch.is_tensor(val) else val[0] for key, val in smpl_data.items()}  # remove B dim
        smpl_data.update(self.sample_points(smpl_output))
        return smpl_data
    
    def __len__(self):
        return len(self.amass_data[self.data_keys[0]])
