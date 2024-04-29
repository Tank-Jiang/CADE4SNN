from data.transform_timm.transforms_factory import transforms_imagenet_train
from datasets.imagenet import ImageNet2p
import argparse
import time
import os
from utils.tools.common import maybe_dictionarize_batch
import shutil

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-location",
        type=str,
        default='/data/guodong/',
        help="The root directory for the datasets.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
    )
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    preprocess = transforms_imagenet_train(
        img_size=224,
        mean=(0.48145466, 0.4578275, 0.40821073),
       std=(0.26862954, 0.26130258, 0.27577711)
    )

    dataset = ImageNet2p(preprocess, location=args.data_location, batch_size=1, num_workers=16)

    loader = dataset.test_loader
    print(type(dataset).__name__)
    if type(dataset).__name__ == 'ImageNet2p':
        loader = dataset.train_loader
        # assert to make sure the imagenet held-out minival logic is consistent across machines.
        # tested on a few machines but if this fails for you please submit an issue and we will resolve.
        assert dataset.train_dataset.__getitem__(dataset.sampler.indices[1000])['image_paths'].endswith('n01675722_4108.JPEG')
        print('the dataset ImageNet2p is ok')
    end = time.time()
    # base_path = '/data/guodong/imagenet/train/'
    targ_path = '/data/guodong/imagenet/train2p/'

    for i, batch in enumerate(loader):
        batch = maybe_dictionarize_batch(batch)
        inputs, labels = batch['images'].cuda(), batch['labels'].cuda()
        data_time = time.time() - end
        y = labels
        # if 'image_paths' in batch:
        image_paths = batch['image_paths']
        source_image = image_paths[0]
        print('i:', i, 'image_paths:', os.path.basename(source_image), 'labels:', labels.item())
        class_pth = os.path.basename(source_image).split('_')[0]
        target_floder = os.path.join(targ_path, class_pth)
        if not os.path.exists(target_floder): os.makedirs(target_floder)
        shutil.move(source_image, target_floder)

