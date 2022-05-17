import copy
import trimesh
import numpy as np

kinect2scene = np.array([[0.718967938589424, -0.0854097393587232, 0.6897755284896544, -0.6849263160394952], [-0.013824958036209914, -0.9939820361576085, -0.10866730111347925, -0.4792354698455745], [0.6949057301647507, 0.06859218773785698, -0.7158220015936552, 2.887512806158444], [0.0, 0.0, 0.0, 1.0]])
rot, trans = kinect2scene[:3, :3], kinect2scene[:3, -1]

scene_mesh = trimesh.load_mesh(f'/home/marko/projects/COAP-private/tutorials/samples/scene_collision/raw_kinect_scan/_scan.obj')
# scene_mesh.apply_transform(np.linalg.inv(kinect2scene))
svert = scene_mesh.vertices - trans[None]  # N,3
scene_mesh.vertices = (rot.T @ svert.T).T
scene_mesh.vertex_normals = (rot.T @ scene_mesh.vertex_normals.T).T
scene_mesh.face_normals = (rot.T @ scene_mesh.face_normals.T).T
# scene_mesh = copy.deepcopy(scene_mesh)
scene_mesh.export(f'/home/marko/projects/COAP-private/tutorials/samples/scene_collision/raw_kinect_scan/tmp/scan.obj')