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
        latent_code_order, # ['...']
        input_shape=(2, 3, 448, 448), 
    ):
        super(CXRCrossEncoder, self).__init__()

        _, _, org_height, org_width = input_shape
        bottle_neck_height = org_height // 32
        bottle_neck_width  = org_width  // 32

        print ('bottle_neck size: ', bottle_neck_height, bottle_neck_width)

        assert org_height >= 224 and org_width >= 224

        # encoder 
        self.encoder = modules.resnet18(pretrained = False)
        # modify input chnnel (first conv)
        self.encoder.conv1 = nn.Conv2d(6,64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias = False)
        bottleneck_shape = _get_eleme_num(self.encoder, torch.randn(input_shape))

        # calculate z-dim
        self.z_dim = 0
        for k, v in z_dict.items():
            self.z_dim += v 
        self.z_dict =z_dict
        self.swap_list = swap_list
        self.pred_dict = pred_dict
        self.latent_code_order = latent_code_order
        assert len(self.latent_code_order) == len(self.z_dict.keys())
              
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

    def _concat_latent_code(self, latent_dict):
        ret = []
        for code_name in self.latent_code_order:
            ret.append(latent_dict[code_name])

        ret = torch.cat(ret, dim=1)    # 128 x 20, 128 x 8, 128 x 4 = 128 x 32
        return ret

    def _swap_latent_code(self, latent_code, z_size):
        ''' docstring
        args: 
            latent_code (torch.Tensor) : before swap
            z_size (int) : dimension of the input latent code
        return:
            latent_code (torch.Tensor) : after swap

        e.g.
            (before) torch.Tensor([1,2,3,4, 5,6,7,8])
            (after)  torch.Tensor([5,6,7,8, 1,2,3,4])

        '''
        swap_order = [x for x in range(z_size//2, z_size)] # (8) 4 5 6 7
        swap_order += [x for x in range(z_size//2)]
        latent_code = latent_code[:, swap_order]
        return latent_code

   
    def forward(self, x): # x: x-ray
        features = self.encoder(x)
        features = features.mean(-1).mean(-1) # (2048 X H x W) -> (2048)

        # feature to latent code
        latent_code = {}
        for k, v in self.z_dict.items():
            latent_code[k] = self.encoder_fc[k](features)

        # make predictions
        pred = {}
        for k, v in self.pred_dict.items():
            pred[k] = self.prediction_net[k](latent_code[k])

        # feature swap
        for swap_feat_name in self.swap_list:
            latent_code[swap_feat_name] = self._swap_latent_code(
                latent_code[swap_feat_name], self.z_dict[swap_feat_name]
            )
            
        # reconstruction
        z_all = self._concat_latent_code(latent_code)
        x_hat = self.decoder(z_all)

        
        output_dict = {
            'latent_code_dict' : latent_code,    # after swap
            'pred_dict' : pred,
            'x_hat' : x_hat                      # reconstructed image
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
    
    latent_code_order = ['sex', 'age', 'cac', 'common']
    model2 = CXRCrossEncoder(z_dict, swap_list, pred_dict, latent_code_order, input_shape=(2,6,448, 448)).cuda()

    batch_size = 4
    image = torch.rand(batch_size, 6, 448, 448).cuda()


    output_dict2 = model2(image) # w/ bottleneck-linear



    for k, v in output_dict2.items():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                print(k, k2, v2.shape)
        else:
            print(k, v.shape)
