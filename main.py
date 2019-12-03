import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
import resnet
from advGAN import AdvGAN_Attack
from models import MNIST_target_net
import os
use_cuda=True
image_nc=1
epochs = 60
batch_size = 128
BOX_MIN = 0
BOX_MAX = 1

# Define what device we are using
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

net = resnet.ResNet18()
net = net.cuda()
net = torch.nn.DataParallel(net)
checkpoint = torch.load("H:/adversarial_attacks/pytorch-cifar/checkpoint/DataPackpt.pth")
net.load_state_dict(checkpoint['net'])
targeted_model = net
# pretrained_model = "./MNIST_target_model.pth"
# targeted_model = MNIST_target_net().to(device)
# targeted_model.load_state_dict(torch.load(pretrained_model))
targeted_model.eval()
model_num_labels = 10

# MNIST train dataset and dataloader declaration
# mnist_dataset = torchvision.datasets.MNIST('./dataset', train=True, transform=transforms.ToTensor(), download=True)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
# dataloader = DataLoader(
#             datasets.CIFAR10('H:/cifar-10-batches-py', train=True, download=True, transform=transform),
#             batch_size=batch_size, shuffle=True)
roubust_data_path = 'H:/adversarial_attacks/no-robness-adv/release_datasets/d_robust_CIFAR'
train_data = torch.cat(torch.load(os.path.join(roubust_data_path,f"CIFAR_ims")))
train_labels = torch.cat(torch.load(os.path.join(roubust_data_path,f"CIFAR_lab")))
train_set = torch.utils.data.TensorDataset(train_data, train_labels)
dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)

advGAN = AdvGAN_Attack(device,
                          targeted_model,
                          model_num_labels,
                          image_nc,
                          BOX_MIN,
                          BOX_MAX)

advGAN.train(dataloader, epochs)
