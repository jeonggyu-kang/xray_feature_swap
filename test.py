import os 
import copy 
import argparse 

from utils import parse_args_from_config 
from models import get_model 
from logger import get_logger 
from runner import tester 
from dataset import get_dataloader
# from visualizer import Umap

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_root', type=str, default=None, help='Path to log directory')
    _args = parser.parse_args()

    if not os.path.exists(_args.log_root):
        print('Can not find directory {}'.format(_args.log_root))
        exit()


    _args.config = os.path.join(_args.log_root, 'config.py')
    modulevar = parse_args_from_config(_args.config)

    args = modulevar.get_hyperparameters(config=_args.config)


    ckpt_path = os.path.join(_args.log_root, 'best.pt')

    model = get_model(
        z_dict = args.get('z_dict'),
        pred_dict = args.get('pred_dict'),
        swap_list = args.get('swap_list'),
        latent_code_order = args.get('latent_code_order'),
        input_shape = args.get('input_shape'),   
        ckpt_path=ckpt_path
    )
          

    mode = 'test'
    
    test_loader = get_dataloader(
        dataset = args['dataset'],
        data_dir = args[mode]['img_dir'],
        ann_path = args[mode]['ann_file'],
        mode = mode,
        batch_size = args['batch_size'],
        num_workers = args['workers_per_gpu'],
        pipeline = args[mode]['pipeline'],
        csv = False,
        pickle_path = args['pickle_path']
    )

    save_path = os.path.join(_args.log_root, 'eval')
    writer = get_logger(save_path)


    tester(
        model = model,
        test_loader = test_loader,
        writer = writer,
        confusion_matrix = True,
        csv = False,
        hard_sample = True,   # sex_hard_sample
        meta = dict(
            age_ratio_thres = 10,
            age_diff_thres = 0.2
        )

    )



if __name__ == '__main__':
    main()