from torch.utils.data import DataLoader
from torchvision import transforms

from .datasets import ImageNetDataset
from .transforms import build_transformer, TwoCropsTransform, GaussianBlur
from .auto_augmentation import ImageNetPolicy
from .sampler import build_sampler
# from .metrics import build_evaluator
from .pipelines import ImageNetTrainPipeV2, ImageNetValPipeV2
from .nvidia_dali_dataloader import DaliDataloader


def build_common_augmentation(aug_type):
    """
    common augmentation settings for training/testing ImageNet
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if aug_type == 'STANDARD':
        augmentation = [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.ToTensor(),
            normalize,
        ]
    elif aug_type == 'AUTOAUG':
        augmentation = [
            transforms.RandomResizedCrop(224),
            ImageNetPolicy(),
            transforms.ToTensor(),
            normalize,
        ]
    elif aug_type == 'MOCOV1':
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
    elif aug_type == 'MOCOV2' or aug_type == 'SIMCLR':
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
    elif aug_type == 'LINEAR':
        augmentation = [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    elif aug_type == 'ONECROP':
        augmentation = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    else:
        raise RuntimeError("undefined augmentation type for ImageNet!")

    if aug_type in ['MOCOV1', 'MOCOV2', 'SIMCLR']:
        return TwoCropsTransform(transforms.Compose(augmentation))
    else:
        return transforms.Compose(augmentation)


def build_imagenet_train_dataloader(cfg_dataset, data_type='train'):
    """
    build training dataloader for ImageNet
    """
    cfg_train = cfg_dataset['train']
    # build dataset
    if cfg_dataset['use_dali']:
        # NVIDIA dali preprocessing
        assert cfg_train['transforms']['type'] == 'STANDARD', 'only support standard augmentation'
        dataset = ImageNetDataset(
            root_dir=cfg_train['root_dir'],
            meta_file=cfg_train['meta_file'],
            read_from=cfg_dataset['read_from'],
            server_cfg=cfg_train.get("server", {}),
        )
    else:
        image_reader = cfg_dataset[data_type].get('image_reader', {})
        # PyTorch data preprocessing
        if isinstance(cfg_train['transforms'], list):
            transformer = build_transformer(cfgs=cfg_train['transforms'],
                                            image_reader=image_reader)
        else:
            transformer = build_common_augmentation(cfg_train['transforms']['type'])
        dataset = ImageNetDataset(
            root_dir=cfg_train['root_dir'],
            meta_file=cfg_train['meta_file'],
            transform=transformer,
            read_from=cfg_dataset['read_from'],
            image_reader_type=image_reader.get('type', 'pil'),
            server_cfg=cfg_train.get("server", {}),
        )
    # build sampler
    cfg_train['sampler']['kwargs'] = {}
    cfg_dataset['dataset'] = dataset
    sampler = build_sampler(cfg_train['sampler'], cfg_dataset)
    if cfg_dataset['last_iter'] >= cfg_dataset['max_iter']:
        return {'loader': None}
    # build dataloader
    if cfg_dataset['use_dali']:
        # NVIDIA dali pipeline
        pipeline = ImageNetTrainPipeV2(
            data_root=cfg_train['root_dir'],
            data_list=cfg_train['meta_file'],
            sampler=sampler,
            crop=cfg_dataset['input_size'],
            colorjitter=[0.2, 0.2, 0.2, 0.1]
        )
        loader = DaliDataloader(
            pipeline=pipeline,
            batch_size=cfg_dataset['batch_size'],
            epoch_size=len(sampler),
            num_threads=cfg_dataset['num_workers'],
            last_iter=cfg_dataset['last_iter']
        )
    else:
        # PyTorch dataloader
        loader = DataLoader(
            dataset=dataset,
            batch_size=cfg_dataset['batch_size'],
            shuffle=False,
            num_workers=cfg_dataset['num_workers'],
            pin_memory=True,
            sampler=sampler
        )
    return {'type': 'train', 'loader': loader}


def build_imagenet_test_dataloader(cfg_dataset, data_type='test'):
    """
    build testing/validation dataloader for ImageNet
    """
    cfg_test = cfg_dataset['test']
    # build evaluator
    evaluator = None
    if cfg_test.get('evaluator', None):
        evaluator = build_evaluator(cfg_test['evaluator'])
    if cfg_dataset['use_dali']:
        # NVIDIA dali preprocessing
        assert cfg_test['transforms']['type'] == 'ONECROP', 'only support onecrop augmentation'
        dataset = ImageNetDataset(
            root_dir=cfg_test['root_dir'],
            meta_file=cfg_test['meta_file'],
            read_from=cfg_dataset['read_from'],
            evaluator=evaluator,
        )
    else:
        image_reader = cfg_dataset[data_type].get('image_reader', {})
        # PyTorch data preprocessing
        if isinstance(cfg_test['transforms'], list):
            transformer = build_transformer(cfgs=cfg_test['transforms'],
                                            image_reader=image_reader)
        else:
            transformer = build_common_augmentation(cfg_test['transforms']['type'])
        dataset = ImageNetDataset(
            root_dir=cfg_test['root_dir'],
            meta_file=cfg_test['meta_file'],
            transform=transformer,
            read_from=cfg_dataset['read_from'],
            evaluator=evaluator,
            image_reader_type=image_reader.get('type', 'pil'),
        )
    # build sampler
    assert cfg_test['sampler'].get('type', 'distributed') == 'distributed'
    cfg_test['sampler']['kwargs'] = {'dataset': dataset, 'round_up': False}
    cfg_dataset['dataset'] = dataset
    sampler = build_sampler(cfg_test['sampler'], cfg_dataset)
    # build dataloader
    if cfg_dataset['use_dali']:
        # NVIDIA dali pipeline
        pipeline = ImageNetValPipeV2(
            data_root=cfg_test['root_dir'],
            data_list=cfg_test['meta_file'],
            sampler=sampler,
            crop=cfg_dataset['input_size'],
            size=cfg_dataset['test_resize'],
        )
        loader = DaliDataloader(
            pipeline=pipeline,
            batch_size=cfg_dataset['batch_size'],
            epoch_size=len(sampler),
            num_threads=cfg_dataset['num_workers'],
            dataset=dataset,
        )
    else:
        # PyTorch dataloader
        loader = DataLoader(
            dataset=dataset,
            batch_size=cfg_dataset['batch_size'],
            shuffle=False,
            num_workers=cfg_dataset['num_workers'],
            pin_memory=cfg_dataset['pin_memory'],
            sampler=sampler
        )
    return {'type': 'test', 'loader': loader}


def build_imagenet_search_dataloader(cfg_dataset, data_type='arch'):
    """
    build ImageNet dataloader for neural network search (NAS)
    """
    cfg_search = cfg_dataset[data_type]
    # build dataset
    if cfg_dataset['use_dali']:
        # NVIDIA dali preprocessing
        assert cfg_search['transforms']['type'] == 'ONECROP', 'only support onecrop augmentation'
        dataset = ImageNetDataset(
            root_dir=cfg_search['root_dir'],
            meta_file=cfg_search['meta_file'],
            read_from=cfg_dataset['read_from'],
        )
    else:
        image_reader = cfg_dataset[data_type].get('image_reader', {})
        # PyTorch data preprocessing
        if isinstance(cfg_search['transforms'], list):
            transformer = build_transformer(cfgs=cfg_search['transforms'],
                                            image_reader=image_reader)
        else:
            transformer = build_common_augmentation(cfg_search['transforms']['type'])
        dataset = ImageNetDataset(
            root_dir=cfg_search['root_dir'],
            meta_file=cfg_search['meta_file'],
            transform=transformer,
            read_from=cfg_dataset['read_from'],
            image_reader_type=image_reader.get('type', 'pil'),
        )
    # build sampler
    assert cfg_search['sampler'].get('type', 'distributed_iteration') == 'distributed_iteration'
    cfg_search['sampler']['kwargs'] = {}
    cfg_dataset['dataset'] = dataset
    sampler = build_sampler(cfg_search['sampler'], cfg_dataset)
    if cfg_dataset['last_iter'] >= cfg_dataset['max_iter']:
        return {'loader': None}
    # build dataloder
    if cfg_dataset['use_dali']:
        # NVIDIA dali pipeline
        pipeline = ImageNetValPipeV2(
            data_root=cfg_search['root_dir'],
            data_list=cfg_search['meta_file'],
            sampler=sampler,
            crop=cfg_dataset['input_size'],
            size=cfg_dataset['test_resize'],
        )
        loader = DaliDataloader(
            pipeline=pipeline,
            batch_size=cfg_dataset['batch_size'],
            epoch_size=len(sampler),
            num_threads=cfg_dataset['num_workers'],
        )
    else:
        # PyTorch dataloader
        loader = DataLoader(
            dataset=dataset,
            batch_size=cfg_dataset['batch_size'],
            shuffle=False,
            num_workers=cfg_dataset['num_workers'],
            pin_memory=cfg_dataset['pin_memory'],
            sampler=sampler
        )
    return {'type': data_type, 'loader': loader}
