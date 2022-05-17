import os
import sys
import time
import pickle
import pathlib
import argparse

import smplx
import torch
import trimesh
import pyrender
import numpy as np
import torch.nn.functional as F

from coap import attach_coap

torch.manual_seed(0)
np.random.seed(0)


def visualize(model, smpl_output, samples=None):
    if not VISUALIZE:
        return

    def vis_create_pc(pts, color=(0.0, 1.0, 0.0), radius=0.005):
        if torch.is_tensor(pts):
            pts = pts.cpu().numpy()

        tfs = np.tile(np.eye(4), (pts.shape[0], 1, 1))
        tfs[:, :3, 3] = pts
        sm_in = trimesh.creation.uv_sphere(radius=radius)
        sm_in.visual.vertex_colors = color

        return pyrender.Mesh.from_trimesh(sm_in, poses=tfs)

    VIEWER.render_lock.acquire()
    # clear scene
    while len(VIEWER.scene.mesh_nodes) > 0:
        VIEWER.scene.mesh_nodes.pop()

    posed_mesh = trimesh.Trimesh(smpl_output.vertices[0].detach().cpu().numpy(), model.faces)
    posed_mesh.visual.vertex_colors = (0.3, 0.3, 0.3, 0.8)

    # update scene
    VIEWER.scene.add(pyrender.Mesh.from_trimesh(posed_mesh))
    if samples is not None:
        VIEWER.scene.add(vis_create_pc(samples))

    VIEWER.render_lock.release()

def load_smpl_data(pkl_path):
    def to_tensor(x, device):
        if torch.is_tensor(x):
            return x.to(device=device)
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).to(device=device)
        return x

    with open(pkl_path, 'rb') as f:
        param = pickle.load(f)
    torch_param = {key: to_tensor(val, args.device) for key, val in param.items()}
    
    # for visualization -- disable global rotation and translation as they do not influence self-penetration
    for zero_key in ['global_orient', 'transl', 'left_hand_pose', 'right_hand_pose']:
        if zero_key in torch_param:
            torch_param[zero_key][:] = 0.0

    if args.model_type == 'smpl':
        smpl_body_pose = torch.zeros((1, 69), dtype=torch.float, device=args.device)
        smpl_body_pose[:, :63] = torch_param['body_pose']
        torch_param['body_pose'] = smpl_body_pose

    return torch_param


def main():
    # create a SMPL body and attach COAP
    model = smplx.create(model_path=args.bm_dir_path, model_type=args.model_type, gender=args.gender, num_pca_comps=12)
    assert model.joint_mapper is None, 'COAP requires valid SMPL joints as input'
    model = attach_coap(model, device=args.device)

    data = load_smpl_data(args.sample_body)
    init_pose = data['body_pose'].detach().clone()

    data['body_pose'].requires_grad = True
    opt = torch.optim.SGD([data['body_pose']], lr=args.lr)

    for step in range(args.max_iters):
        # smpl forward pass
        smpl_output = model(**data, return_verts=True, return_full_pose=True)
        # NOTE: make sure that smpl_output contains the valid SMPL variables (pose parameters, joints, and vertices). 

        # compute self collision loss
        selfpen_loss, _samples = model.coap.self_collision_loss(smpl_output, ret_samples=True)

        # pose prior
        pose_prior_loss = args.pose_prior_weight*F.mse_loss(init_pose, data['body_pose'])

        # visualization and opt step 
        selfpen_loss = selfpen_loss*args.selfpen_weight
        loss = selfpen_loss + pose_prior_loss
        if VISUALIZE:
            visualize(model, smpl_output, _samples[0])
            print('iter ', step, ':\t', selfpen_loss.item(), ':\t', pose_prior_loss.item(), '\tWaiting 0.5s')
            time.sleep(1.4)
        
        if selfpen_loss < 0.5:
            print('Converged')
            break
        loss.backward(retain_graph=True)
        opt.step()
    
    visualize(model, smpl_output)
    print('exiting in 10 seconds')
    time.sleep(10)
    VIEWER.close_external()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('An example script to resolve self-intersections.')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default='cuda', help='Device (cuda or cpu).')

    # SMPL specification
    parser.add_argument('--bm_dir_path', type=str, required=True, help='Directory with SMPL bodies.')
    parser.add_argument('--model_type', type=str, choices=['smpl', 'smplx'], default='smplx', help='SMPL-based body type.')
    parser.add_argument('--gender', type=str, choices=['male', 'female', 'neutral'], default='neutral', help='SMPL gender.')

    # data samples
    parser.add_argument('--sample_body', type=str, default='./samples/scene_collision/selfpen_examples/001.pkl', help='SMPL parameters.')

    parser.add_argument('--max_iters', default=200, type=int, help='The maximum number of optimization steps.')
    parser.add_argument('--lr', default=0.00001, type=float, help='Learning rate.')
    parser.add_argument('--pose_prior_weight', default=1e3, type=float, help='Weight for the pose prior term (discourages large deviations from the initial pose).')
    parser.add_argument('--selfpen_weight', default=0.1, type=float, help='Weight for the self-penetration term.')
    
    args = parser.parse_args()
    VISUALIZE = True
    if VISUALIZE:
        _scene = pyrender.Scene(ambient_light=[.1, 0.1, 0.1], bg_color=[1.0, 1.0, 1.0])
        VIEWER = pyrender.Viewer(_scene, use_raymond_lighting=True, run_in_thread=True)
    main()
