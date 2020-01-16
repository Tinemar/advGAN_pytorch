import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import transforms as transforms

import cifar10.cifar_loader as cifar_loader
import models
import resnet
import Utils
from cifar10.cifar_resnets import resnet32
from models import MNIST_target_net

use_cuda=True
image_nc=3
batch_size = 64

gen_input_nc = image_nc

# Define what device we are using
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

def visualize_results(G, batch_size,adv_images,i):
    # G.eval()
    tot_num_samples = 64
    image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))
    # print(image_frame_dim)
    # # sample_z_ = torch.rand((64, 2352))
    # sample_z_ = torch.rand((batch_size, 62))
    # sample_z_ = sample_z_.cuda()
    # samples = G(sample_z_)

    samples = adv_images.cpu().data.numpy().transpose(0, 2, 3, 1)

    samples = (samples + 1) / 2
    Utils.save_images(samples[:(image_frame_dim * image_frame_dim)+i*64, :, :, :], [image_frame_dim, image_frame_dim],
                      './results/cifar10/test'+str(i)+'.png')
# load the pretrained model
# net = resnet.ResNet101()
# net = net.cuda()
# checkpoint = torch.load("H:/adversarial_attacks/pytorch-cifar/checkpoint/resnet101ckpt.pth")
# net.load_state_dict(checkpoint['net'])
# target_model = net
# target_model.eval()

#resnet32
# target_model = cifar_loader.load_pretrained_cifar_resnet(flavor=20)
target_model = cifar_loader.load_pretrained_cifar_resnet(flavor=32)
# target_model = cifar_loader.load_pretrained_cifar_wide_resnet()
target_model = target_model.cuda()
target_model.eval()

#resnet32_advtrain
# target_model = resnet32()
# target_model.load_state_dict(torch.load('./target_models/PGDadv_trained_resnet32_197.pkl'))
# target_model.load_state_dict(torch.load('./target_models/PGDadv_trained_wideresnet_69.pkl'))
# target_model.load_state_dict(torch.load('./target_models/PGDadv_trained_resnet20_149.pkl'))
# target_model.load_state_dict(torch.load('./target_models\FGSMtrain.resnet20.000100.path.tar'))
target_model.load_state_dict(torch.load('./target_models\FGSMtrain.resnet32.000100.path.tar'))
# target_model = target_model.cuda()
# target_model.eval()


# load the generator of adversarial examples
pretrained_generator_path = './models/resnet32netG_best.pth'
# pretrained_generator_path = './models/wideresnetnetG_best.pth'
# pretrained_generator_path = './models/resnet20netG_best.pth'
pretrained_G = models.Generator(gen_input_nc, image_nc).to(device)
pretrained_G.load_state_dict(torch.load(pretrained_generator_path))
pretrained_G.eval()

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, 4),
    transforms.ToTensor(),
    normalize,
])
# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
# test adversarial examples in MNIST training dataset

# cifar10_dataset = torchvision.datasets.CIFAR10('H:/adversarial_attacks/cifar-10-batches-py', train=True, transform=transform, download=True)
# train_dataloader = DataLoader(cifar10_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
# num_correct = 0
# for i, data in enumerate(train_dataloader, 0):
#     test_img, test_label = data
#     test_img, test_label = test_img.to(device), test_label.to(device)
#     perturbation = pretrained_G(test_img)
#     perturbation = torch.clamp(perturbation, -0.3, 0.3)
#     adv_img = perturbation + test_img
#     adv_img = torch.clamp(adv_img, 0, 1)
#     pred_lab = torch.argmax(target_model(adv_img),1)
#     target_label = torch.LongTensor([0]*len(test_label))
#     target_label = target_label.cuda()
#     num_correct += torch.sum(pred_lab==target_label,0)

# print('CIFAR10 training dataset:')
# print('num_correct: ', num_correct.item())
# print('accuracy of adv imgs in training set: %f\n'%(num_correct.item()/len(cifar10_dataset)),len(cifar10_dataset))

# test adversarial examples in MNIST testing dataset
cifar10_dataset_test = torchvision.datasets.CIFAR10('H:/adversarial_attacks/cifar-10-batches-py', train=False, transform=transform, download=True)
test_dataloader = DataLoader(cifar10_dataset_test, batch_size=batch_size, shuffle=False, num_workers=0)
num_correct = 0
num_correct_ori = 0
for i, data in enumerate(test_dataloader, 0):
    test_img, test_label = data
    test_img, test_label = test_img.to(device), test_label.to(device)
    perturbation = pretrained_G(test_img)
    perturbation = torch.clamp(perturbation, -0.3, 0.3)
    adv_img = perturbation + test_img
    adv_img = torch.clamp(adv_img, 0, 1)
    pred_lab = torch.argmax(target_model(adv_img),1)
    pred_lab_ori = torch.argmax(target_model(test_img),1)
    target_label = torch.LongTensor([0]*len(test_label))
    target_label = target_label.cuda()
    visualize_results(pretrained_G, batch_size,adv_img,i)
    num_correct += torch.sum(pred_lab==test_label,0)
    num_correct_ori += torch.sum(pred_lab_ori==test_label,0)

print('num_correct: ', num_correct.item())
print('num_correct_ori: ', num_correct_ori.item())
acc_adv = (num_correct.item()/len(cifar10_dataset_test))
acc_ori = (num_correct_ori.item()/len(cifar10_dataset_test))
print('accuracy of adv imgs in testing set: %f\n'% acc_adv)
print('accuracy of ori imgs in testing set: %f\n'% acc_ori)
print(acc_ori-acc_adv)