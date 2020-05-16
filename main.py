import torch
import torchvision
from torchvision import transforms, datasets
import torchvision.transforms as standard_transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from PIL import Image 
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import sys

from models import ConvDeconv as cd   
from models import ConvDeconvWithSkipConnection as cds  
from models import SegNet as s
from models import SegNetWithSkipConnection as ss  

data_path = "/home/sur/SemSeg/cityscape/"

# Loading the dataset
train = datasets.Cityscapes(data_path, split = 'train', mode = 'fine', target_type = 'semantic',transform=transforms.Compose([transforms.Resize((256,512)),transforms.ToTensor()]),target_transform=transforms.Compose([transforms.Resize((256,512)),transforms.ToTensor()]))
test = datasets.Cityscapes(data_path, split = 'test', mode = 'fine', target_type = 'semantic' ,transform=transforms.Compose([transforms.Resize((256,512)),transforms.ToTensor()]),target_transform=transforms.Compose([transforms.Resize((256,512)),transforms.ToTensor()]))
val = datasets.Cityscapes(data_path, split = 'val', mode = 'fine', target_type = 'semantic' ,transform=transforms.Compose([transforms.Resize((256,512)),transforms.ToTensor()]),target_transform=transforms.Compose([transforms.Resize((256,512)),transforms.ToTensor()]))

trainset = torch.utils.data.DataLoader(train, batch_size=2, shuffle=True) #1488 images
testset = torch.utils.data.DataLoader(test, batch_size=2, shuffle=False) #763 images
valset = torch.utils.data.DataLoader(val, batch_size=2, shuffle=True) #250 images

# print(trainset.size())

output_visualization =0
learing_rate = 0.001
epoachs = 100

def main():
    # CHecking tha availability of the GPU and running
  if torch.cuda.is_available():
    device = torch.device("cuda:0") 
    print("Running on the GPU")
  else:
    device = torch.device("cpu")
    print("Running on the CPU")
    #  Creating the object of model class
  net = cd.ConvDeconv().to(device)
#   net = cds.ConvDeconvWithSkipConnection().to(device) 
#   net = s.SegNet().to(device) 
#   net = ss.SegNetWithSkipConnection().to(device) 

#   print(net)
  return net,device 


def load_pretrained_weights(net):
  vgg = torchvision.models.vgg16(pretrained=True,progress=True)
  pretrained_dict = vgg.state_dict()
  model_dict = net.state_dict()
  list1 = ['layer10_conv.weight',
    'layer10_conv.bias',
    'layer11_conv.weight',
    'layer11_conv.bias',
    'layer20_conv.weight',
    'layer20_conv.bias',
    'layer21_conv.weight',
    'layer21_conv.bias',
    'layer30_conv.weight',
    'layer30_conv.bias',
    'layer31_conv.weight',
    'layer31_conv.bias',
    'layer32_conv.weight',
    'layer32_conv.bias',
    'layer40_conv.weight',
    'layer40_conv.bias',
    'layer41_conv.weight',
    'layer41_conv.bias',
    'layer42_conv.weight',
    'layer42_conv.bias',
    'layer50_conv.weight',
    'layer50_conv.bias',
    'layer51_conv.weight',
    'layer51_conv.bias',
    'layer52_conv.weight',
    'layer52_conv.bias'
    ]
  list2 = ['features.0.weight',
    'features.0.bias',
    'features.2.weight',
    'features.2.bias',
    'features.5.weight',
    'features.5.bias',
    'features.7.weight',
    'features.7.bias',
    'features.10.weight',
    'features.10.bias',
    'features.12.weight',
    'features.12.bias',
    'features.14.weight',
    'features.14.bias',
    'features.17.weight',
    'features.17.bias',
    'features.19.weight',
    'features.19.bias',
    'features.21.weight',
    'features.21.bias',
    'features.24.weight',
    'features.24.bias',
    'features.26.weight',
    'features.26.bias',
    'features.28.weight',
    'features.28.bias'
    ]
  for l in range(len(list1)):
    pretrained_dict[list1[l]] = pretrained_dict.pop(list2[l])

  pretrained_dict = {k: v for k,v in pretrained_dict.items() if k in model_dict}
  model_dict.update(pretrained_dict)
  net.load_state_dict(model_dict)
  return net

def train(net, device):
# Defining the loss function as its segmentation task we will use pixel wise cross entropy loss
  loss_function = nn.CrossEntropyLoss(torch.ones(34)).cuda()
# Defining the optimizer and learing rate 
  optimizer = optim.Adam(net.parameters(), lr=learing_rate)
# As we are doing the batch training we nee dto pass through data multiple times this pass is called epoach
  for epoch in range(epoachs): 
    #   Loading the data as a feature vector and labels for all the images in trianing dataset  
    for data in trainset:  
        X, y = data
        # model is loaded in the GPU for running
        X, y = X.to(device), y.to(device)
        # before doing the back propogation we need to set gradient zero as pytorch accumulates the gadients on subsequent backward passes
        net.zero_grad()  
        # forward pass of data from the model
        # output size will be [batch size, #of classes, image height, image width]
        output = net(X)       
        # output size will be [batch size, #of classes, image height*image width] as we flatten the output
        output = output.view(output.size(0),output.size(1), -1)        
        output = torch.transpose(output,1,2).contiguous()
        output = output.view(-1,output.size(2))
        # output size will be [batch size*image height*image width, #of classes] 
        # normalising the labels so that there values is between 1 and #on classes              
        label = y*255
        label = label.long()
        # flattening the label vector [batch size*image height*image width, #of classes]
        label = label.view(-1)
        # Loss calculation and back propagation
        loss = loss_function(output, label)
        loss.backward() 
        optimizer.step()
    print("Epoch No: ",epoch)
    print("loss: ",loss) 

    # Storing the weights after every epoach
    torch.save(net.state_dict(),'/home/sur/SegNet/weights/wts_segnet.pth')

def decode_segmap(image,classes=33):
    label_color_dict = np.array([(0,0,0),(0,0,127),(0,0,255),(0,127,0),(0,127,127),(0,127,255),(0,255,0),(0,255,127),(0,255,255),
                                 (85,0,0),(85,0,127),(85,0,255),(85,127,0),(85,127,127),(85,127,255),(85,255,0),(85,255,127),(85,255,255),
                                 (170,0,0),(170,0,127),(170,0,255),(170,127,0),(170,127,127),(170,127,255),(170,255,0),(170,255,127),(170,255,255),
                                 (255,0,0),(255,0,127),(255,0,255),(255,127,0),(255,127,127),(255,127,255),(255,255,0)])
    r_label = np.zeros_like(image).astype(np.uint8)
    g_label = np.zeros_like(image).astype(np.uint8)
    b_label = np.zeros_like(image).astype(np.uint8) 

    # iterating over image and updating the r,g,b values
    for i in range (0,classes):
        index = image == i
        r_label[index] = label_color_dict[i, 0]
        g_label[index] = label_color_dict[i, 1]
        b_label[index] = label_color_dict[i, 2]

    rgb = np.stack([r_label, g_label, b_label], axis=2)
    return rgb

def test(net, device):
    correct_classification =0
    valset_size = 500

    with torch.no_grad():
        wrongclassifiedpixelcount = 0
        # iterating over all the data from the validation set
        for data in valset:
            X, y = data
            X, y = X.to(device), y.to(device)
            # passing the data from network to get the output
            output = net(X)
            # len(output) is nothing but batch size
            for idx in range(len(output)):
                # following is the softmax out put for every pixel size = imagesize*#class
                predicted_output = output[idx]
                # Here we are storing the indixes of max value in the class direction size = image size
                predicted_output_max_index = torch.argmax(predicted_output,0)
                # converting the array to numpy array
                predicted_max_idx = predicted_output_max_index.detach().cpu().numpy()

                # comparision between actual labels and predicted labels
                label = y[idx][0].detach().cpu().numpy()
                final_diff = predicted_max_idx - label*255
                wrongclassifiedpixelcount = wrongclassifiedpixelcount + np.count_nonzero(final_diff)

                # output visualization
                # funtion to assign the rgb value to pixel and generating the rgb image based on the predicted labels
                if(output_visualization):
                    predicted_rgb = decode_segmap(predicted_max_idx) 
                    fig = plt.figure(1)
                    plt.imshow(predicted_rgb)
                    plt.figure(2)
                    plt.imshow(transforms.ToPILImage()(data[0][idx]))
                    plt.show()
        accuracy = 1 - nonzerocount/(valset_size*256*512)
        print("Accuracy",accuracy)

if __name__ == '__main__':
  sec = time.time()
# Crating the network and assigning the GPUs
  net, device = main()
# Loading the weights of pretrainied VGG network for encoder, incase you have already trainied the model then no need to load the weights
  net = load_pretrained_weights(net)
# When you train your model it will save the weights so in case if you want to triain it more you can just take the saved weights and start training
# when it comes to giving the benchmarks to the model you can use the same weights for validation and testing purpose

  net.load_state_dict(torch.load('/home/sur/SegNet/weights/wts_segnet.pth'))

# Training the network  
#   train(net,device)
# Testing the network
  test(net, device)
  sec_last = time.time()
  print("Network Runtime",sec_last-sec)