import torch
import smplx
from .coap import COAPBodyModel

def attach_coap(parametric_body: smplx.SMPL, pretrained=True, device=None):
    coap_body = COAPBodyModel(parametric_body)
    setattr(parametric_body, 'coap', coap_body)
    if pretrained:
        model_type = coap_body.model_type
        gender = parametric_body.gender
        checkpoint = f'https://github.com/markomih/COAP/blob/dev/models/coap_{model_type}_{gender}.ckpt?raw=true'
        state_dict = torch.hub.load_state_dict_from_url(checkpoint, progress=True)
        coap_body.load_state_dict(state_dict['state_dict'])
    if device is not None:
        parametric_body = parametric_body.to(device=device)

    # overwrite smpl functions
    def reset_params(self, **params_dict) -> None:
        with torch.no_grad():
            for param_name, param in self.named_parameters():
                if 'coap' in param_name:  # disable reset of coap parameters
                    continue
                if param_name in params_dict:
                    param[:] = torch.tensor(params_dict[param_name])
                else:
                    param.fill_(0)
    setattr(parametric_body, 'reset_params', lambda **x: reset_params(parametric_body, **x))
    
    return parametric_body

__all__ = [
    attach_coap,
]