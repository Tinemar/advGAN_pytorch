import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import transforms as transforms

import models
import Utils
from models import MNIST_target_net

use_cuda=True
image_nc=1
batch_size = 128

gen_input_nc = image_nc
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
                      './results/mnist/test'+str(i)+'.png')
# Define what device we are using
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

# load the pretrained model
# pretrained_model = "./MNIST_target_model.pth"
# target_model = MNIST_target_net().to(device)
# target_model.load_state_dict(torch.load(pretrained_model))
# target_model.eval()
target_model = models.LeNet5()
target_model.load_state_dict(torch.load('./FGSMadv_trained_lenet5.pkl'))
target_model.cuda()
target_model.eval()
# load the generator of adversarial examples
pretrained_generator_path = './models/mnist_2netG_epoch_60.pth'
pretrained_G = models.Generator(gen_input_nc, image_nc).to(device)
pretrained_G.load_state_dict(torch.load(pretrained_generator_path))
pretrained_G.eval()

# test adversarial examples in MNIST training dataset
mnist_dataset = torchvision.datasets.MNIST('./dataset', train=True, transform=transforms.ToTensor(), download=True)
train_dataloader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
num_correct = 0
for i, data in enumerate(train_dataloader, 0):
    test_img, test_label = data
    test_img, test_label = test_img.to(device), test_label.to(device)
    perturbation = pretrained_G(test_img)
    perturbation = torch.clamp(perturbation, -0.3, 0.3)
    adv_img = perturbation + test_img
    adv_img = torch.clamp(adv_img, 0, 1)
    pred_lab = torch.argmax(target_model(adv_img),1)
    target_label = torch.LongTensor([0]*len(test_label))
    target_label = target_label.cuda()
    num_correct += torch.sum(pred_lab==target_label,0)

print('MNIST training dataset:')
print('num_correct: ', num_correct.item())
print('accuracy of adv imgs in training set: %f\n'%(num_correct.item()/len(mnist_dataset)))

# test adversarial examples in MNIST testing dataset
mnist_dataset_test = torchvision.datasets.MNIST('./dataset', train=False, transform=transforms.ToTensor(), download=True)
test_dataloader = DataLoader(mnist_dataset_test, batch_size=batch_size, shuffle=False, num_workers=0)
num_correct = 0
for i, data in enumerate(test_dataloader, 0):
    test_img, test_label = data
    test_img, test_label = test_img.to(device), test_label.to(device)
    perturbation = pretrained_G(test_img)
    perturbation = torch.clamp(perturbation, -0.3, 0.3)
    adv_img = perturbation + test_img
    adv_img = torch.clamp(adv_img, 0, 1)
    pred_lab = torch.argmax(target_model(adv_img),1)
    print(pred_lab)
    target_label = torch.LongTensor([0]*len(test_label))
    target_label = target_label.cuda()
    num_correct += torch.sum(pred_lab==target_label,0)

print('num_correct: ', num_correct.item())
print('accuracy of adv imgs in testing set: %f\n'%(num_correct.item()/len(mnist_dataset_test)))
