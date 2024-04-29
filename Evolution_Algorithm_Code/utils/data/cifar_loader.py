import os
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
from PIL import Image
def create_loader_cifar(args):
    class Cutout(object):
        """Randomly mask out one or more patches from an image.

        Args:
            n_holes (int): Number of patches to cut out of each image.
            length (int): The length (in pixels) of each square patch.
        """
        def __init__(self, n_holes, length):
            self.n_holes = n_holes
            self.length = length

        def __call__(self, img):
            """
            Args:
                img (Tensor): Tensor image of size (C, H, W).
            Returns:
                Tensor: Image with n_holes of dimension length x length cut out of it.
            """
            h = img.size(1)
            w = img.size(2)

            mask = np.ones((h, w), np.float32)

            for n in range(self.n_holes):
                y = np.random.randint(h)
                x = np.random.randint(w)

                y1 = np.clip(y - self.length // 2, 0, h)
                y2 = np.clip(y + self.length // 2, 0, h)
                x1 = np.clip(x - self.length // 2, 0, w)
                x2 = np.clip(x + self.length // 2, 0, w)

                mask[y1: y2, x1: x2] = 0.

            mask = torch.from_numpy(mask)
            mask = mask.expand_as(img)
            img = img * mask

            return img


    trans1 = transforms.Compose([transforms.RandomCrop(32, padding=4, fill=128),
                                    transforms.RandomHorizontalFlip(), transforms.ToTensor(),#Cutout(n_holes=1, length=16),
                                    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
    trans2 = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),Cutout(n_holes=1, length=16),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])


    train_dataset = torchvision.datasets.CIFAR100(root=os.path.join(args.data, 'cifar100'), train=True,
                                                    transform=trans1, download=True)
    test_dataset = torchvision.datasets.CIFAR100(root=os.path.join(args.data, 'cifar100'), train=False,
                                                    transform=torchvision.transforms.Compose([
                                                        torchvision.transforms.ToTensor(),
                                                        torchvision.transforms.Normalize((0.5071, 0.4867, 0.4408),
                                                                                        (0.2675, 0.2565, 0.2761))
                                                    ]), download=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=20, pin_memory=True)
    val_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=20, pin_memory=True)
    return train_loader,val_loader,train_loader
def split_cifar100(args):
    trans1 = transforms.Compose([transforms.RandomCrop(32, padding=4, fill=128),
                                    transforms.RandomHorizontalFlip(), transforms.ToTensor(),#Cutout(n_holes=1, length=16),
                                    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
    
    train_dataset = torchvision.datasets.CIFAR100(root=os.path.join(args.data, 'cifar100'), train=True,
                                                    transform=trans1, download=True)
    test_dataset = torchvision.datasets.CIFAR100(root=os.path.join(args.data, 'cifar100'), train=False,
                                                    transform=torchvision.transforms.Compose([
                                                        torchvision.transforms.ToTensor(),
                                                        torchvision.transforms.Normalize((0.5071, 0.4867, 0.4408),
                                                                                        (0.2675, 0.2565, 0.2761))
                                                    ]), download=True)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=20, pin_memory=True)
    # val_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=20, pin_memory=True)
    x_list = [[] for i in range(args.split_num)]
    y_list = [[] for i in range(args.split_num)]

    idx_of_dataset=[len(train_loader)/args.split_num*i for i in range(0,args.split_num+1)]

    dataset_list=[[] for i in range(args.split_num)]
    for batch_idx, (input, target) in enumerate(train_loader):
        for i in range(1,args.split_num+1):
            if idx_of_dataset[i-1] <= batch_idx < idx_of_dataset[i]:
                x_list[i-1].append(input.squeeze())
                y_list[i-1].append(target)
    for i in range(args.split_num):
        images_tensor = torch.stack(x_list[i])
        labels_tensor = torch.Tensor(y_list[i])
        print(images_tensor.shape)
        print(labels_tensor.shape)
        dataset = TensorDataset(images_tensor,labels_tensor)
        dataset_list[i].append(DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=20, pin_memory=True)) 
    val_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=20, pin_memory=True)
    # for i in range(args.split_num):
    #     print(len(x_list[0]))
    #     print(len(y_list[0]))
    #     print(x_list[0][0].shape)
    #     print(y_list[0][0].shape)
    for batch in dataset_list[0]:
        # input, target=batch
        print(batch)
        return

    return dataset_list,val_loader
        # print(y_list[0][i].shape)
#  
# def split_cifar100(args):
#     trans1 = transforms.Compose([transforms.RandomCrop(32, padding=4, fill=128),
#                                     transforms.RandomHorizontalFlip(), transforms.ToTensor(),#Cutout(n_holes=1, length=16),
#                                     transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
    
#     train_dataset = torchvision.datasets.CIFAR100(root=os.path.join(args.data, 'cifar100'), train=True,
#                                                     transform=trans1, download=True)
#     test_dataset = torchvision.datasets.CIFAR100(root=os.path.join(args.data, 'cifar100'), train=False,
#                                                     transform=torchvision.transforms.Compose([
#                                                         torchvision.transforms.ToTensor(),
#                                                         torchvision.transforms.Normalize((0.5071, 0.4867, 0.4408),
#                                                                                         (0.2675, 0.2565, 0.2761))
#                                                     ]), download=True)
#     x = train_dataset.data
#     y = np.array(train_dataset.targets)
#     x_transformed = []
#     for i in x:
#         x_transformed.append(trans1(Image.fromarray(i.astype('uint8'))))

#     # 将转换后的数据转换为Tensor
#     x_transformed = torch.stack(x_transformed)
#     x_splits, x_splits1, y_splits, y_splits1 = train_test_split(x_transformed, y, test_size=0.2, random_state=42)
#     x_splits, x_splits2, y_splits, y_splits2 = train_test_split(x_splits, y_splits, test_size=0.25, random_state=42)#!!!!!
#     x_splits, x_splits3, y_splits, y_splits3 = train_test_split(x_splits, y_splits, test_size=1/3, random_state=42)
#     x_splits, x_splits4, y_splits, y_splits4 = train_test_split(x_splits, y_splits, test_size=0.5, random_state=42)
#     x_splits, y_splits,x_splits1, y_splits1,x_splits2, y_splits2,x_splits3, \
#         y_splits3,x_splits4, y_splits4 = torch.tensor(x_splits).permute(0, 3, 1, 2) ,torch.tensor(y_splits),torch.tensor(x_splits1).permute(0, 3, 1, 2),torch.tensor(y_splits1),\
#         torch.tensor(x_splits2).permute(0, 3, 1, 2),torch.tensor(y_splits2),torch.tensor(x_splits3).permute(0, 3, 1, 2),torch.tensor(y_splits3),torch.tensor(x_splits4).permute(0, 3, 1, 2),torch.tensor(y_splits4)
 
#     # x_splits, y_splits,x_splits1, y_splits1,x_splits2, y_splits2,x_splits3, \
#     #     y_splits3,x_splits4, y_splits4 = torch.tensor(x_splits),torch.tensor(y_splits),torch.tensor(x_splits1),torch.tensor(y_splits1),\
#     #     torch.tensor(x_splits2),torch.tensor(y_splits2),torch.tensor(x_splits3),torch.tensor(y_splits3),torch.tensor(x_splits4),torch.tensor(y_splits4)
#     dataset_1=TensorDataset(x_splits, y_splits)
#     dataset_2=TensorDataset(x_splits1, y_splits1)
#     dataset_3=TensorDataset(x_splits2, y_splits2)
#     dataset_4=TensorDataset(x_splits3, y_splits3)
#     dataset_5=TensorDataset(x_splits4, y_splits4)
#     dataset_all = TensorDataset(torch.tensor(x).permute(0, 3, 1, 2),torch.tensor(y))
#     train_loader_1 = DataLoader(dataset_1, batch_size=args.batch_size, shuffle=True, num_workers=20, pin_memory=True)
#     train_loader_2 = DataLoader(dataset_2, batch_size=args.batch_size, shuffle=True, num_workers=20, pin_memory=True)
#     train_loader_3 = DataLoader(dataset_3, batch_size=args.batch_size, shuffle=True, num_workers=20, pin_memory=True)
#     train_loader_4 = DataLoader(dataset_4, batch_size=args.batch_size, shuffle=True, num_workers=20, pin_memory=True)
#     train_loader_5 = DataLoader(dataset_5, batch_size=args.batch_size, shuffle=True, num_workers=20, pin_memory=True)
#     train_loader_all = DataLoader(dataset_all, batch_size=args.batch_size, shuffle=True, num_workers=20, pin_memory=True)
    
#     val_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=20, pin_memory=True)

#     return [train_loader_1,train_loader_2,train_loader_3,train_loader_4,train_loader_5],train_loader_all,val_loader
#     # return train_loader_1,val_loader