import torch
import numpy as np

from pytorch3d import renderer
from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh import Textures


class Renderer():
    """ Adapted from SNARF """

    @torch.no_grad()
    def __init__(self, image_size=512):
        super().__init__()

        R = torch.from_numpy(np.array([[-1., 0., 0.],
                                       [0., 1., 0.],
                                       [0., 0., -1.]])).float().unsqueeze(0)

        t = torch.from_numpy(np.array([[0., 0.0, 5.]])).float()

        cameras = renderer.FoVOrthographicCameras(R=R, T=t)
        lights = renderer.PointLights(location=[[0.0, 0.0, 3.0]],
                                           ambient_color=((1, 1, 1),), diffuse_color=((0, 0, 0),),
                                           specular_color=((0, 0, 0),))
        raster_settings = renderer.RasterizationSettings(image_size=image_size, faces_per_pixel=100, blur_radius=0)
        rasterizer = renderer.MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
        shader = renderer.HardPhongShader(cameras=cameras, lights=lights)
        shader = renderer.SoftPhongShader(cameras=cameras, lights=lights)
        self.renderer = renderer.MeshRenderer(rasterizer=rasterizer, shader=shader)

    @torch.no_grad()
    def render_mesh(self, verts, faces, colors=None, mode='npat'):
        """
        mode: normal, phong, texture
        """
        mesh = Meshes(verts, faces)

        normals = torch.stack(mesh.verts_normals_list())
        front_light = torch.tensor([0, 0, 1]).float().to(verts.device)
        shades = (normals * front_light.view(1, 1, 3)).sum(-1).clamp(min=0).unsqueeze(-1).expand(-1, -1, 3)
        results = []

        self.renderer.to(verts.device)
        # normal
        if 'n' in mode:
            normals_vis = normals * 0.5 + 0.5
            mesh_normal = Meshes(verts, faces, textures=Textures(verts_rgb=normals_vis))
            image_normal = self.renderer(mesh_normal)
            results.append(image_normal)

        # shading
        if 'p' in mode:
            mesh_shading = Meshes(verts, faces, textures=Textures(verts_rgb=shades))
            image_phong = self.renderer(mesh_shading)
            results.append(image_phong)

        # albedo
        if 'a' in mode:
            assert (colors is not None)
            mesh_albido = Meshes(verts, faces, textures=Textures(verts_rgb=colors))
            image_color = self.renderer(mesh_albido)
            results.append(image_color)

        # albedo*shading
        if 't' in mode:
            assert (colors is not None)
            mesh_teture = Meshes(verts, faces, textures=Textures(verts_rgb=colors * shades))
            image_color = self.renderer(mesh_teture)
            results.append(image_color)

        return torch.cat(results, axis=1)
