from .autoencoder import CXRCrossEncoder
import torch

def get_model(z_dict, pred_dict, swap_list, latent_code_order, input_shape, ckpt_path = None, **kwargs):
    
    model = CXRCrossEncoder(
        z_dict = z_dict, 
        pred_dict = pred_dict,
        swap_list = swap_list,
        latent_code_order = latent_code_order,
        input_shape = input_shape
    )


    if ckpt_path is not None:
        print (f'Loading trained weight from {ckpt_path}..')
        ckpt = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(ckpt['weight'])

        # encoder fc
        for k, v in model.encoder_fc.items():
            k2 = 'encoder_' + k 
            model.encoder_fc[k].load_state_dict( ckpt[k2])

        # prediction networks
        for k, v in model.prediction_net.items():
            k2 = 'prediction_' + k 
            model.prediction_net[k].load_state_dict( ckpt[k2])

    model.cuda()

    return model