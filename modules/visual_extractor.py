import torch
import torch
import torch.nn as nn
import torchvision.models as models
from modules.Swing_Transformer import SwinTransformer as STBackbone
from modules.crate import CRATE


# CNN-based Methods
class VisualExtractor(nn.Module):
    def __init__(self, args):
        super(VisualExtractor, self).__init__()
        self.visual_extractor = args.visual_extractor
        self.pretrained = args.visual_extractor_pretrained
        model = getattr(models, self.visual_extractor)(pretrained=self.pretrained)
        modules = list(model.children())[:-2]
        self.model = nn.Sequential(*modules)
        self.avg_fnt = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)

    def forward(self, images):
        patch_feats = self.model(images)
        avg_feats = self.avg_fnt(patch_feats).squeeze().reshape(-1, patch_feats.size(1))
        batch_size, feat_size, _, _ = patch_feats.shape
        patch_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)
        #print(patch_feats.shape)
        return patch_feats, avg_feats

'''
class VisualExtractor(nn.Module):
    def __init__(self, args):
        super(VisualExtractor, self).__init__()
        self.CNN = STBackbone(
                    img_size=224, 
                    embed_dim=192, 
                    depths=[2, 2, 18, 2],
                    num_heads=[6, 12, 24, 48],
                    window_size=7,
                    num_classes=1000
                    )
        
        self.CNN.load_weights('/public/home/huarong/yixiulong/RM/Global_Enhanced_Transformer/data/Pretrained_Swin_Transformer-22K/large.pth')
        # Freeze parameters
        for _name, _weight in self.CNN.named_parameters():
            _weight.requires_grad = False

    def forward(self, images):
        patch_feats = self.CNN(images)  
        avg_feats = torch.mean(patch_feats,dim=1)
        return patch_feats, avg_feats
'''
'''
class VisualExtractor(nn.Module):
    def __init__(self, args):
        super(VisualExtractor, self).__init__()
        self.CNN = CRATE(image_size=224,
              patch_size = 16,
              num_classes = 1000,
              dim = 768,
              depth = 12,
              heads = 12,
              dim_head = 768 // 12)

    def forward(self, images):
        patch_feats = self.CNN(images)
        #print(patch_feats.shape)  
        avg_feats = torch.mean(patch_feats,dim=1)
        #print(avg_feats.shape)
        return patch_feats, avg_feats
'''