import torch
import torch.nn as nn
import numpy as np 


try:
    from . import modules
except:
    import modules


def _get_eleme_num(model, x):
    y = model(x)
    _, C, H, W = y.shape
    return (C,H,W)


class CXRCrossEncoder(nn.Module):
    def __init__(self, 
        z_dict,
        swap_list,
        pred_dict,
        input_shape=(2, 3, 448, 448), 
    ):
        super(CXRCrossEncoder, self).__init__()

        _, _, org_height, org_width = input_shape
        bottle_neck_height = org_height // 32
        bottle_neck_width  = org_width  // 32

        print ('bottle_neck size: ', bottle_neck_height, bottle_neck_width)

        assert org_height >= 224 and org_width >= 224

        # encoder 
        self.encoder = modules.resnet18(pretrained = True)
        bottleneck_shape = _get_eleme_num(self.encoder, torch.randn(input_shape))

        # calculate z-dim
        self.z_dim = 0
        for k, v in z_dict.items():
            self.z_dim += v 
        self.z_dict =z_dict
        self.swap_list = swap_list
        self.pred_dict = pred_dict
              
        # encoder fc    
        self.encoder_fc = {}
        for k, v in z_dict.items():
            self.encoder_fc[k] = nn.Linear(bottleneck_shape[0], v)
            self.encoder_fc[k].cuda()
        
        # prediction networks
        self.prediction_net = {}
        for k, v in pred_dict.items():
            self.prediction_net[k] = nn.Sequential(
                nn.Linear(z_dict[k], 256),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(256, v)
            )     
            self.prediction_net[k].cuda()   


        # decoder 
        self.decoder = modules.ResDeconv(
            block=modules.BasicBlock,
            global_avg_pool = True,
            z_all = self.z_dim,
            bottleneck_shape = (2048, bottle_neck_height, bottle_neck_width)
        )

   
    def forward(self, x): # x: x-ray
        features = self.encoder(x)
        features = features.mean(-1).mean(-1) # (2048 X H x W) -> (2048)

        # feature to latent code
        latent_code = {}
        _latent_code_gather = []
        for k, v in self.z_dict.items():
            latent_code[k] = self.encoder_fc[k](features)
            _latent_code_gather.append(latent_code[k])


        # make predictions
        pred = {}
        for k, v in self.pred_dict.items():
            pred[k] = self.prediction_net[k](latent_code[k])

        # feature swap

        for swap_feat_name in self.swap_list:
            






        # reconstruction
        all_latent_code = torch.cat(_latent_code_gather, dim = 1)
        x_hat = self.decoder(all_latent_code)

        
        output_dict = {
            'latent_code_dict' : latent_code,
            'pred_dict' : pred,
            'x_hat' : x_hat
        }


        return output_dict

if __name__ == '__main__':
    z_dict = {
        'common' : 20,
        'age' : 8,
        'sex' : 6,
        'cac' : 12
    }
    swap_list = ['common', 'sex', 'cac']
    pred_dict = {
        'age' : 1,
        'sex' : 2,
        'cac' : 1,
    }
    
    model2 = CXRCrossEncoder(z_dict, swap_list, pred_dict,input_shape=(2,3,448, 448)).cuda()

    batch_size = 4
    image = torch.rand(batch_size, 3, 448, 448).cuda()


    output_dict2 = model2(image) # w/ bottleneck-linear



    for k, v in output_dict2.items():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                print(k, k2, v2.shape)
        else:
            print(k, v.shape)
