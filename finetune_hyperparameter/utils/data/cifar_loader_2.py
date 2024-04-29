import os
import random
import logging
import numpy as np
from functools import partial
from typing import Callable
import torch.utils.data
from torchvision.datasets import ImageFolder
from .transform_timm.transforms_factory import transforms_imagenet_train, transforms_imagenet_eval,create_transform
import torchvision
from .transform_timm.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import DataLoader
def create_loader_cifar(args):
    train_transform = create_transform(
        input_size = 32,
        is_training=True,
        use_prefetcher=False,
        no_aug=False,
        scale=[1.0,1.0],
        ratio=[1.0,1.0],
        hflip=0.5,
        vflip=0.,
        color_jitter=0,
        auto_augment=args.aa,
        interpolation='bicubic',
        mean=[0.4914,0.4822,0.4465],
        std=[0.2470,0.2435,0.2616],
        re_prob=args.re_prob,
        re_mode='const',
        re_count=1,
        re_num_splits=0,
        crop_pct=1.0,
        tf_preprocessing=False,
        separate=False)
    # print("train_transform:",train_transform)
    trainset = torchvision.datasets.CIFAR100(root=os.path.join(args.data, 'cifar100'), train=True,
                                            download=True, transform=train_transform)
    if args.distributed: sampler_train = torch.utils.data.distributed.DistributedSampler(trainset)

    # valid_transform=create_transform(
    #     input_size= 32,
    #     is_training=False,
    #     use_prefetcher=False,
    #     no_aug=False,
    #     scale=None,
    #     ratio=None,
    #     hflip=0.5,
    #     vflip=0.,
    #     color_jitter=0,
    #     auto_augment=None,
    #     interpolation='bicubic',
    #     mean=(0.5071, 0.4867, 0.4408),
    #     std=(0.2675, 0.2565, 0.2761),
    #     re_prob=0.,
    #     re_mode='const',
    #     re_count=1,
    #     re_num_splits=0,
    #     crop_pct=1.0,
    #     tf_preprocessing=False,
    #     separate=False)
    # valset = torchvision.datasets.CIFAR100(root=os.path.join(args.data_dir, 'cifar10'), train=False,
    #                                     download=True, transform=valid_transform)
    # if args.distributed: sampler_val = torch.utils.data.distributed.DistributedSampler(valset)
    loader_class = torch.utils.data.DataLoader
    loader_args_train = dict(
        batch_size=args.batch_size,
        shuffle= True,
        num_workers=20,
        # sampler=sampler_train,
        collate_fn=None,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=partial(_worker_init, worker_seeding='all'),
        persistent_workers=True
    )
    # loader_args_val = dict(
    #     batch_size=args.val_batch_size,
    #     shuffle= sampler_val is None and False,
    #     num_workers=16,
    #     sampler=sampler_val,
    #     collate_fn=None,
    #     pin_memory=True,
    #     drop_last=False,
    #     worker_init_fn=partial(_worker_init, worker_seeding='all'),
    #     persistent_workers=True
    # )
    train_loader = loader_class(trainset, **loader_args_train)
    # val_loader = loader_class(trainset, **loader_args_val)
    test_dataset = torchvision.datasets.CIFAR100(root=os.path.join(args.data, 'cifar100'), train=False,
                                                    transform=torchvision.transforms.Compose([
                                                        torchvision.transforms.ToTensor(),
                                                        torchvision.transforms.Normalize((0.5071, 0.4867, 0.4408),
                                                                                        (0.2675, 0.2565, 0.2761))
                                                    ]), download=True)

    val_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=20, pin_memory=True)
   
    return train_loader,val_loader

def _worker_init(worker_id, worker_seeding='all'):
    worker_info = torch.utils.data.get_worker_info()
    assert worker_info.id == worker_id
    if isinstance(worker_seeding, Callable):
        seed = worker_seeding(worker_info)
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed % (2 ** 32 - 1))
    elif worker_seeding in ('all', 'part'):
        # random / torch seed already called in dataloader iter class w/ worker_info.seed
        # to reproduce some old results (same seed + hparam combo), partial seeding is required (skip numpy re-seed)
        if worker_seeding == 'all':
            np.random.seed(worker_info.seed % (2 ** 32 - 1))
    else:
        seed = worker_seeding
        # _logger.info('seed: {}'.format(seed))
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed % (2 ** 32 - 1))