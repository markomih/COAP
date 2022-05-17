# COAP: Compositional Articulated Occupancy of People

[**Paper**](https://arxiv.org/abs/2204.06184) | [**Video**](https://www.youtube.com/watch?v=qU0q5h6IldU) | [**Project Page**](https://neuralbodies.github.io/COAP)

<div style="text-align: center">
    <a href="https://neuralbodies.github.io/COAP"><img src="https://neuralbodies.github.io/COAP/images/teaser.png" alt="teaser figure"/></a>
</div>

This is the official implementation of the CVPR 2022 paper [**COAP: Learning Compositional Occupancy of People**](https://neuralbodies.github.io/COAP).

## Description
This repository provides the official implementation of an implicit human body model (COAP) which implements efficient loss terms for resolving self-intersection and collisions with 3D geometries. 

## Dependencies
First manually install the [Pytorch3D](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md) package and then install other dependencies via: 

```pip install git+https://github.com/markomih/COAP.git```

### Optional Dependencies
Install the [pyrender](https://pyrender.readthedocs.io/en/latest/install/index.html) package to use the visualization/tutorial scripts and follow the additional instructions specified [here](./training_code/) if you wish to retrain COAP.

# Tutorials
COAP extends the interface of the SMPL-X package (follow its [instructions](https://github.com/vchoutas/smplx) for the usage) via two volumetric loss terms: 1) a loss for resolving self-intersections and 2) a loss for resolving collisions with 3D geometries flexibly represented as point clouds. 
In the following, we provide a minimal interface to access the COAP's functionalities:

```python
import smplx
from coap import attach_coap

# create a SMPL body and extend the SMPL body via COAP (we support: smpl, smplh, and smplx model types)
model = smplx.create(**smpl_parameters)
attach_coap(model)

smpl_output = model(**smpl_data)  # smpl forward pass
# NOTE: make sure that smpl_output contains the valid SMPL variables (pose parameters, joints, and vertices). 
assert model.joint_mapper is None, 'COAP requires valid SMPL joints as input'

# access two loss functions
model.coap.selfpen_loss(smpl_output)  # self-intersections
model.coap.collision_loss(smpl_output, scan_point_cloud)  # collisions with other geometris
```
Additionally we provide two [tutorials](./tutorials) on how to use these terms to resolve self-intersections and collisions with environment.

# Pretrained Models
A respective pretrained model will be automatically fetched and loaded.
All the pretrained models are available on the `dev` branch inside the `./TRAINED_MODELS` directory. 

# Citation
```bibtex
@inproceedings{Mihajlovic:CVPR:2022,
   title = {{COAP}: Compositional Articulated Occupancy of People},
   author = {Mihajlovic, Marko and Saito, Shunsuke and Bansal, Aayush and Zollhoefer, Michael and Tang, Siyu},
   booktitle = {Proceedings IEEE Conf. on Computer Vision and Pattern Recognition (CVPR)},
   month = jun,
   year = {2022}
}
```
# Contact
For questions, please contact Marko Mihajlovic (_markomih@ethz.ch_) or raise an issue on GitHub.
