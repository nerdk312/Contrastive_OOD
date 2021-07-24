import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from typing import Type, Any, Callable, Union, List, Optional
from Contrastive_uncertainty.general.models.resnet_models import BasicBlock, Bottleneck, ResNet

# Differs from other cases as it only has class forward branch, no separate branch for unsupervised learning
class CustomResNet(ResNet):
    def __init__(
        self,
        latent_size: int,
        num_channels:int,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:

        super(CustomResNet, self).__init__(block,
        layers,
        num_classes,
        zero_init_residual,
        groups,
        width_per_group,
        replace_stride_with_dilation,
        norm_layer)

        self.conv1 = nn.Conv2d(
            num_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.maxpool = nn.Identity()

        self.class_fc1 = nn.Linear(512 * block.expansion, latent_size)
        self.class_fc2 = nn.Linear(latent_size, num_classes)
        

    # Nawid - made new function to obtain the representation of the data
    def forward(self,x:Tensor)-> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        z = F.relu(self.class_fc1(x)) # Unnormalized currently though it will be normalised in the method    
        
        return z
    
    
   

def _custom_resnet(
    arch: str,
    latent_size:int,
    num_channels:int,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> ResNet:
    model = CustomResNet(latent_size,num_channels,block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def custom_resnet18(latent_size:int = 128, num_channels:int =3,pretrained: bool = False, progress: bool = True, **kwargs: Any) -> CustomResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _custom_resnet('resnet18',latent_size,num_channels, BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)

def custom_resnet34(latent_size:int = 128, num_channels:int =3,pretrained: bool = False, progress: bool = True, **kwargs: Any) -> CustomResNet:
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _custom_resnet('resnet34',latent_size,num_channels, BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)

def custom_resnet50(latent_size:int = 128, num_channels:int =3,pretrained: bool = False, progress: bool = True, **kwargs: Any) -> CustomResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _custom_resnet('resnet50',latent_size,num_channels, Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)