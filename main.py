import os
import argparse

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets

import cifar10.cifar_loader as cifar_loader
import resnet
from advGAN import AdvGAN_Attack
from cifar10.cifar_resnets import resnet32
from models import MNIST_target_net
import models
use_cuda = True
image_nc = 3
batch_size = 128
BOX_MIN = 0
BOX_MAX = 1

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'fashion-mnist', 'cifar10', 'cifar100', 'svhn', 'stl10', 'lsun-bed'],
                    help='The name of dataset')
parser.add_argument('--epoch', type=int, default=60, help='The number of epochs to run')
parser.add_argument('--batch_size', type=int, default=128, help='The size of batch')
parser.add_argument('--checkpoint',type=str,default='')
parser.add_argument('--target_model',type=str,default="")
args = parser.parse_args()

if __name__ == '__main__':
    # Define what device we are using
    print("CUDA Available: ", torch.cuda.is_available())
    device = torch.device("cuda" if (
        use_cuda and torch.cuda.is_available()) else "cpu")

    # net = resnet.ResNet18()
    # net = net.cuda()
    # net = torch.nn.DataParallel(net)
    # checkpoint = torch.load("H:/adversarial_attacks/pytorch-cifar/checkpoint/DataPackpt.pth")
    # net.load_state_dict(checkpoint['net'])
    # target_model = net

    # resnet32
    if args.target_model == 'resnet32':
        target_model = cifar_loader.load_pretrained_cifar_resnet(flavor=32)
    elif args.target_model == 'resnet20':
        target_model = cifar_loader.load_pretrained_cifar_resnet(flavor=20)
    elif args.target_model == 'wideresnet':
        target_model = cifar_loader.load_pretrained_cifar_wide_resnet()
    elif args.target_model=="mnist_2":
        target_model = models.LeNet5()
        target_model.load_state_dict(torch.load('./trained_lenet5.pkl'))
    # target_model = target_model.cuda()
    # target_model.eval()

    # resnet32_advtrain
    # target_model = resnet32()
    # target_model.load_state_dict(torch.load('./advtrain.resnet32.000100.path.tar'))

    target_model = target_model.cuda()
    target_model.eval()

    model_num_labels = 10
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
        normalize,
    ])
    # MNIST train dataset and dataloader declaration
    # mnist_dataset = torchvision.datasets.MNIST('./dataset', train=True, transform=transforms.ToTensor(), download=True)
    # transform = transforms.Compose([transforms.ToTensor()])
    # dataloader = DataLoader(
    #     datasets.MNIST('./dataset/MNIST',
    #                     train=True, download=True, transform=transform),  
    #     batch_size=batch_size, shuffle=True)

    # cifar10
    dataloader = DataLoader(
            datasets.CIFAR10('../cifar-10-batches-py', train=True, download=True, transform=transform),
            batch_size=batch_size, shuffle=True)
    advGAN = AdvGAN_Attack(device,
                        target_model,
                        model_num_labels,
                        image_nc,
                        BOX_MIN,
                        BOX_MAX,
                        args)

    advGAN.train(dataloader, args.epoch)
