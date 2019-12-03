import matplotlib.pyplot as plt
import torch
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import transforms as transforms

import models
import resnet
from models import MNIST_target_net

use_cuda=True
image_nc=3
batch_size = 64

gen_input_nc = image_nc

# Define what device we are using
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

# load the pretrained model
net = resnet.ResNet101()
net = net.cuda()
checkpoint = torch.load("H:/adversarial_attacks/pytorch-cifar/checkpoint/resnet101ckpt.pth")
net.load_state_dict(checkpoint['net'])
target_model = net
target_model.eval()

# load the generator of adversarial examples
pretrained_generator_path = './models/netG_epoch_60.pth'
pretrained_G = models.Generator(gen_input_nc, image_nc).to(device)
pretrained_G.load_state_dict(torch.load(pretrained_generator_path))
pretrained_G.eval()

# test adversarial examples in MNIST training dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
cifar10_dataset = torchvision.datasets.CIFAR10('H:/adversarial_attacks/cifar-10-batches-py', train=True, transform=transform, download=True)
train_dataloader = DataLoader(cifar10_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
num_correct = 0
for i, data in enumerate(train_dataloader, 0):
    test_img, test_label = data
    test_img, test_label = test_img.to(device), test_label.to(device)
    perturbation = pretrained_G(test_img)
    perturbation = torch.clamp(perturbation, -0.3, 0.3)
    adv_img = perturbation + test_img
    adv_img = torch.clamp(adv_img, 0, 1)
    pred_lab = torch.argmax(target_model(adv_img),1)
    num_correct += torch.sum(pred_lab==test_label,0)

print('MNIST training dataset:')
print('num_correct: ', num_correct.item())
print('accuracy of adv imgs in training set: %f\n'%(num_correct.item()/len(cifar10_dataset)),len(cifar10_dataset))

# test adversarial examples in MNIST testing dataset
cifar10_dataset_test = torchvision.datasets.CIFAR10('H:/adversarial_attacks/cifar-10-batches-py', train=False, transform=transform, download=True)
test_dataloader = DataLoader(cifar10_dataset_test, batch_size=batch_size, shuffle=False, num_workers=0)
num_correct = 0
for i, data in enumerate(test_dataloader, 0):
    test_img, test_label = data
    test_img, test_label = test_img.to(device), test_label.to(device)
    perturbation = pretrained_G(test_img)
    perturbation = torch.clamp(perturbation, -0.3, 0.3)
    adv_img = perturbation + test_img
    adv_img = torch.clamp(adv_img, 0, 1)
    pred_lab = torch.argmax(target_model(adv_img),1)
    num_correct += torch.sum(pred_lab==test_label,0)

print('num_correct: ', num_correct.item())
print('accuracy of adv imgs in testing set: %f\n'%(num_correct.item()/len(cifar10_dataset_test)),len(cifar10_dataset_test))
