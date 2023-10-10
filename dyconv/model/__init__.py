import torch
from collections import OrderedDict
from .mobilenet import MobileNet
from .resnet import (  # noqa: F401
    resnet18, resnet26, resnet34, resnet50, resnet10,
    resnet101, resnet152, resnet_custom
)
from .resnet_dyconv import DyResNet10, DyResNet18, DyResNet50
from .resnet_dyconv_pad import DyResNet10_PAD, DyResNet18_PAD, DyResNet50_PAD
from .mobilenetv2_dyconv import DyMobileNetV2, Dymobilenetv2
from .mobilenetv2_dyconv_pad import DyMobileNetV2_PAD, Dymobilenetv2_PAD
from .resnet_dcd import resnet10_dcd, resnet18_dcd, resnet50_dcd
from .resnet_dcd_pad import resnet10_dcd_pad, resnet18_dcd_pad, resnet50_dcd_pad
from .mobilenetv2_dcd import mobilenetv2_dcd
from .mobilenetv2_dcd_pad import mobilenetv2_dcd_cmd
from .resnet_odconv import od_resnet10, od_resnet18, od_resnet50
from .resnet_odconv_pad import od_resnet10_pad, od_resnet18_pad, od_resnet50_pad
from .mobilenetv2_odconv import od_mobilenetv2_050, od_mobilenetv2_075, od_mobilenetv2_100
from .mobilenetv2_odconv_pad import od_mobilenetv2_050, od_mobilenetv2_075, od_mobilenetv2_100

def load_model_pytorch(model, load_model, replace_dict={}):

    checkpoint = torch.load(load_model, map_location='cpu')

    if 'state_dict' in checkpoint.keys():
        load_from = checkpoint['state_dict']
    elif 'model' in checkpoint.keys():
        load_from = checkpoint['model']
    else:
        load_from = checkpoint

    # remove "module." in case the model is saved with Dist Mode
    if 'module.' in list(load_from.keys())[0]:
        load_from = OrderedDict([(k.replace("module.", ""), v) for k, v in load_from.items()])
    for keys in replace_dict.keys():
        load_from = OrderedDict([(k.replace(keys, replace_dict[keys]), v) for k, v in load_from.items()])

    model.load_state_dict(load_from, strict=False)


def model_entry(config, pretrained=False):

    if config['type'] not in globals():
        if config['type'].startswith('spring_'):
            try:
                from spring.models import SPRING_MODELS_REGISTRY
            except ImportError:
                print('Please install Spring2 first!')
            model_name = config['type'][len('spring_'):]
            config['type'] = model_name
            return SPRING_MODELS_REGISTRY.build(config)

    model = globals()[config['type']](**config['kwargs'])
    return model
