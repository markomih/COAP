## Dependencies
#### 1) [LEAP](https://github.com/neuralbodies/leap)
The LEAP package is used to create the ground truth occupancy and for visualization. 
```bash
pip install cython
mkdir external; cd external 
git clone https://github.com/neuralbodies/leap.git
cd leap
python setup.py build_ext --inplace
pip install -e .
```

#### 2) Pytorch Lightining
```
pip install pytorch-lighitning
```

## Data
COAP is trained on AMASS and supports SMPL, SMPL-H, and SMPL+X body models. 

Follow (instructions)[https://github.com/vchoutas/smplx#model-loading] to download these models and store them under a `${MODELS}` directory. 
Then, download the [AMASS](https://amass.is.tue.mpg.de/) dataset and store it under directories `${AMASS_ROOT_smpl}` and `${AMASS_ROOT_smplx}` for the SMPL and SMPL-X compatible COAP versions respectively.

## Training COAP
We provide several configuration scripts under `./configs` to retrain COAP. Use to following command to retrain COAP:
```bash
python train.py ./training_code/configs/body_models/smplx_neutral.yml --out_dir ./TRAINED_MODELS/body_models/smplx_neutral --model_path `${MODELS}` --data_root `${AMASS_ROOT_smplx}` --accelerator 'gpu' --max_epochs 30 --devices 1
```

To see other configuration parameters, check:
```bash
python train.py -h
```
