# Hyperparameter control
import os, shutil

def get_hyperparameters(config = None):
    if config is None: # Train
        _save()
    else: # test
        _load(config)

    ret = {}
    ret.update( model_dict )         # updata: function for updating dictionary
    ret.update( data_dict )

    return ret

_IMAGE_WIDTH = 448
_IMAGE_HEIGHT = 448

# model-related params
model_dict = dict(  
    z_dict = dict(
        common = 24,
        age    = 12,
        sex    = 6,
        cac    = 18
    ), 
    pred_dict = dict(
        age = 1,
        sex = 2,
        cac = 5,
    ),
    swap_list = ['common', 'sex'],
    latent_code_order = ['common', 'age', 'sex', 'cac'],
    max_epoch = 200,
    learning_rate = 1e-4,
    # mile_stone = None,
    mile_stone = [160, 180],
    decay_rate = 0.1,
    input_shape = (1, 6, _IMAGE_WIDTH, _IMAGE_HEIGHT),   # width, height
    extra = ['debug']    
)

train_pipeline = [
    dict(
        type = 'Resize',
        width = _IMAGE_WIDTH,
        height = _IMAGE_HEIGHT
    ),


    dict(
        type = 'Contrastive',
        p = 0.5,
        w = 1.3
    ),

    dict(
        type = 'Sharpness',
        p = 0.5
    ),

    dict(
        type= 'ToTensor'
    ),
]

test_pipeline = [
    dict(
        type = 'Resize',
        width = _IMAGE_WIDTH,
        height = _IMAGE_HEIGHT
    ),
    dict(
        type= 'ToTensor'
    ),
]


# dataset-related params
data_dict = dict(
    dataset = 'VariableAgeDataset',
    save_root = './work_dir',
    batch_size = 16,
    workers_per_gpu = 1,

    pickle_path = '/mnt/xray_feature_swap/data/title.pkl',
    train = dict(
        img_dir = '/mnt/project_classifiers/data',
        ann_file = '/mnt/xray_feature_swap/data/train_dataset.parquet',
        pipeline = train_pipeline
    ),
    test = dict(
        img_dir = '/mnt/project_classifiers/data',
        ann_file = '/mnt/xray_feature_swap/data/test_dataset.parquet',
        pipeline = test_pipeline
    ),
)

def _save():
    model_version = []
    for k in ['name', 'imagenet_pretrained', 'extra']:
        if k in model_dict:
            if isinstance(model_dict[k], list):
                model_version += model_dict[k]
            else:
                model_version.append(str(model_dict[k]))
    
    os.makedirs(data_dict['save_root'], exist_ok = True)
    VERSION = '.'.join(model_version)
    VERSION = str('{:04d}'.format(len(os.listdir(data_dict['save_root'])) + 1) + '_') + VERSION

    SAVE_ROOT_DIR = os.path.join(data_dict['save_root'], VERSION)
    os.makedirs(SAVE_ROOT_DIR, exist_ok=True)
    shutil.copy2(os.path.abspath(__file__), os.path.join(SAVE_ROOT_DIR, __name__ + '.py'))
    data_dict['save_root'] = SAVE_ROOT_DIR

def _load(config):
    data_dict['save_root'] = os.path.join(os.path.dirname(config), 'eval')
    data_dict['max_epoch'] = None