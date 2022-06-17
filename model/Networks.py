import torch
from torch import nn
from torchvision import models



class Classifier_Module_VGG(nn.Module):

    def __init__(self, dims_in, dilation_series, padding_series, num_classes):
        super(Classifier_Module_VGG, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(nn.Conv2d(dims_in, num_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias = True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list)-1):
            out += self.conv2d_list[i+1](x)
            return out

class BaseNet(nn.Module):
    def __init__(self, num_classes, vgg16_caffe_path=None, pretrained=False):
        super(BaseNet, self).__init__()
        vgg = models.vgg16()
        if pretrained:
            vgg.load_state_dict(torch.load(vgg16_caffe_path))

        features, classifier = list(vgg.features.children()), list(vgg.classifier.children())

        features = nn.Sequential(*(features[i] for i in list(range(23))+list(range(24,30))))

        for i in [23,25,27]:
            features[i].dilation = (2,2)
            features[i].padding = (2,2)

        fc6 = nn.Conv2d(512, 1024, kernel_size=3, padding=4, dilation=4)
        fc7 = nn.Conv2d(1024, 1024, kernel_size=3, padding=4, dilation=4)

        self.features = nn.Sequential(*([features[i] for i in range(len(features))] + [ fc6, nn.ReLU(inplace=True), fc7, nn.ReLU(inplace=True)]))
       
        self.classifier1 = Classifier_Module_VGG(1024, [6,12,18,24],[6,12,18,24],num_classes)
        self.classifier2 = Classifier_Module_VGG(1024, [6,12,18,24],[6,12,18,24],num_classes)


    def forward(self, x):
        x = self.features(x)        
        p1 = self.classifier1(x)   
        p2 = self.classifier2(x)
        return p1,p2

    def optim_parameters(self, args):
        return self.parameters()

