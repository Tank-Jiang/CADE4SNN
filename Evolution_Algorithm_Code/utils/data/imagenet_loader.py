import os
import random
import logging
import numpy as np
from functools import partial
from typing import Callable
import torch.utils.data
from torchvision.datasets import ImageFolder
from .transform_timm.transforms_factory import transforms_imagenet_train, transforms_imagenet_eval

_logger = logging.getLogger(__name__)

def create_dataset(data, name, img_size=224, auto_augment=None, is_training=False):
    datadir = os.path.join(data, name)
    if not os.path.isdir(datadir):
        _logger.error('Validation folder does not exist at: {}'.format(datadir))
        exit(1)

    if is_training:
        preprocess = transforms_imagenet_train(img_size=img_size, 
                                               auto_augment=auto_augment)
    else:
        preprocess = transforms_imagenet_eval(img_size=img_size, crop_pct=0.875)
    dataset = ImageFolder(datadir, transform=preprocess)
    return dataset

def create_loader(data, name,
        batch_size,
        num_workers,
        is_training=False,
        img_size=224,
        auto_augment=None,
        distributed=True,
        persistent_workers=False,
        worker_seeding='all',
):
    dataset = create_dataset(data, name, img_size=img_size, auto_augment=auto_augment, is_training=False)
    sampler = None
    if distributed: sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    loader_class = torch.utils.data.DataLoader
    loader_args = dict(
        batch_size=batch_size,
        shuffle= sampler is None and is_training,
        num_workers=num_workers,
        sampler=sampler,
        collate_fn=None,
        pin_memory=True,
        drop_last=is_training,
        worker_init_fn=partial(_worker_init, worker_seeding=worker_seeding),
        persistent_workers=persistent_workers
    )
    loader = loader_class(dataset, **loader_args)
    return loader

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
