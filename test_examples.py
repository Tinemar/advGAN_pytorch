import matplotlib.pyplot as plt
import torch
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import transforms as transforms
import cifar10.cifar_loader as cifar_loader
from cifar10.cifar_resnets import resnet32
import torch.nn.functional as F

import models
import resnet

use_cuda=True
image_nc=3
batch_size = 64

# Define what device we are using
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
target_model = cifar_loader.load_pretrained_cifar_resnet(flavor=32,return_normalizer=False)
# target_model = resnet32()
# target_model.load_state_dict(torch.load('./cifar10_resnet32.th')['state_dict'].items())
target_model = target_model.cuda()
target_model.eval()
transform = transforms.Compose([transforms.ToTensor()])
cifar10_dataset = torchvision.datasets.CIFAR10('../cifar-10-batches-py', train=False, transform=transform, download=True)
train_dataloader = DataLoader(cifar10_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
num_correct = 0
for i, data in enumerate(train_dataloader, 0):
    test_img, test_label = data
    test_img, test_label = test_img.to(device), test_label.to(device)
    probs_model = F.softmax(target_model(test_img), dim=1)
    onehot_labels = torch.eye(10, device='cuda')[test_label]
    print(onehot_labels.size())
    real = torch.sum(onehot_labels * probs_model, dim=1)
    print(real)
    pred_lab = torch.argmax(target_model(test_img),1)
    print(pred_lab,test_label)
    num_correct += torch.sum(pred_lab==test_label,0)
    exit()

print('num_correct: ', num_correct.item())
print('accuracy of adv imgs in training set: %f\n'%(num_correct.item()/len(cifar10_dataset)),len(cifar10_dataset))