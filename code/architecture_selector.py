import torch.nn as nn
from torchvision import transforms, datasets, models


def resnet34(outshape, source_domain):
    if source_domain == "ImageNet":
        resnet = models.resnet34(pretrained=True)
    elif not source_domain:
        resnet = models.resnet34(pretrained=False)
    else:
        raise NotImplementedError("Unknown source domain.")
    num_features = resnet.fc.in_features
    resnet.fc = nn.Linear(num_features, outshape)
    return resnet


def resnet50(outshape, source_domain):
    if source_domain == "ImageNet":
        resnet = models.resnet50(pretrained=True)
    elif not source_domain:
        resnet = models.resnet50(pretrained=False)
    else:
        raise NotImplementedError("Unknown source domain.")
    num_features = resnet.fc.in_features
    resnet.fc = nn.Linear(num_features, outshape)
    return resnet


def densenet121(outshape, source_domain):
    if source_domain == "ImageNet":
        densenet =  models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    elif not source_domain:
        densenet =  models.densenet121(weights=None)
    else:
        raise NotImplementedError("Unknown source domain.")
    densenet.classifier =nn.Linear(densenet.classifier.in_features, outshape)
    return densenet
 


class XrayModel(nn.Module):
    def __init__(self, backbone):
        super(XrayModel, self).__init__()
        self.resnet = backbone # output size = resnet18_shape

    def forward(self, x):
        x = self.resnet(x)
        return x


def model_selector(model_name, source_domain=None, outshape=13):

    # resnet family

    if model_name == "resnet34":
        return XrayModel(resnet34(outshape, source_domain))
    
    elif model_name == "resnet50":
        return XrayModel(resnet50(outshape, source_domain))

    elif model_name == "densenet121":
        return XrayModel(densenet121(outshape, source_domain))
    elif model_name == "vgg":
        pass
    elif model_name == "inception":
        pass
    elif model_name == "vit":
        pass
    else:
        raise NotImplementedError("Unknown model name")