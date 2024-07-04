from functools import partial
from transformers import CLIPTextConfig, CLIPVisionConfig
from transformers import CLIPConfig, CLIPModel
from transformers import Blip2Model
from transformers import AutoTokenizer, CLIPTextModel
import torch
from torch import nn
import torch.nn.functional as F
import copy
from torch.distributions import Normal, Independent
from losses import *
import numpy as np

class BayesCap_MLP(nn.Module):
    '''
    Baseclass to create a simple MLP
    Inputs
        inp_dim: int, Input dimension
        out_dim: int, Output dimension
        hid_dim: int, hidden dimension
        num_layers: Number of hidden layers
        p_drop: dropout probability 
    '''
    def __init__(
        self, 
        inp_dim, 
        out_dim,
        hid_dim=512, 
        num_layers=1, 
        p_drop=0,
    ):
        super(BayesCap_MLP, self).__init__()
        mod = []
        for layer in range(num_layers):
            if layer==0:
                incoming = inp_dim
                outgoing = hid_dim
                mod.append(nn.Linear(incoming, outgoing))
                mod.append(nn.ReLU())
            elif layer==num_layers//2:
                incoming = hid_dim
                outgoing = hid_dim
                mod.append(nn.Linear(incoming, outgoing))
                mod.append(nn.ReLU())
                mod.append(nn.Dropout(p=p_drop))
            elif layer==num_layers-1:
                incoming = hid_dim
                outgoing = out_dim
                mod.append(nn.Linear(incoming, outgoing))
        self.mod = nn.Sequential(*mod)

        self.block_mu = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

        self.block_alpha = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            # nn.Linear(out_dim, out_dim),
            # nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
        )

        self.block_beta = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            # nn.Linear(out_dim, out_dim),
            # nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
        )
    
    def forward(self, x):
        x_intr = self.mod(x)
        # print('dbg', x_intr.shape, x.shape)
        x_intr = x_intr + x
        x_mu = self.block_mu(x_intr)
        x_1alpha = self.block_alpha(x_intr)
        x_beta = self.block_beta(x_intr)
        return x_mu, x_1alpha, x_beta

EPISILON=1e-9

class NCELoss(torch.nn.Module): ####有问题

    def __init__(self, temperature=0.1):
        super(NCELoss, self).__init__()
        self.temperature = temperature
        self.softmax = nn.Softmax(dim=1)

    def where(self, cond, x_1, x_2):
        cond = cond.type(torch.float32)
        return (cond * x_1) + ((1 - cond) * x_2)

    def forward(self, f1, f2, targets):
        ### cuda implementation
        f1 = F.normalize(f1, dim=1)
        f2 = F.normalize(f2, dim=1)

        ## set distances of the same label to zeros
        mask = targets.unsqueeze(1) - targets
        self_mask = (torch.zeros_like(mask) != mask).float()  ### where the negative samples are labeled as 1
        dist = (f1.unsqueeze(1) - f2).pow(2).sum(2)

        ## convert l2 distance to cos distance
        cos = 1 - 0.5 * dist

        ## convert cos distance to exponential space
        pred_softmax = self.softmax(cos / self.temperature) ### convert to multi-class prediction scores

        log_pos_softmax = - torch.log(pred_softmax + EPISILON) * (1 - self_mask.float())
        log_neg_softmax = - torch.log(1 - pred_softmax + EPISILON) * self_mask.float()
        log_softmax = log_pos_softmax.sum(1) / (1 - self_mask).sum(1).float() + log_neg_softmax.sum(1) / (1-self_mask).sum(1).float()
        loss = log_softmax

        return loss.mean()



class ALBEF(nn.Module):
    def __init__(self,                 
                 text_encoder = None,
                 tokenizer = None,
                 config = None,     
                 ):
        super().__init__()
        
        self.blip2 = Blip2Model.from_pretrained("/mnt/raid/yangcp/checkpoint/BLIP2", torch_dtype=torch.bfloat16)

        # self.image_encoder = copy.deepcopy(self.blip2.vision_model)
        # self.qformer_encoder = copy.deepcopy(self.blip2.qformer)
        # self.lm_encoder = copy.deepcopy(self.blip2.language_model)

        self.image_map = nn.Linear(1408, 64 ,dtype=torch.bfloat16)
        self.text_map = nn.Linear(768, 64,dtype=torch.bfloat16)

        self.cls_head = nn.Sequential(
                  nn.Linear(2176, 2176,dtype=torch.bfloat16),
                  nn.ReLU(),
                  nn.Linear(2176, 1,dtype=torch.bfloat16)#ve是三分类需要改成二分类
                )

    def forward(self, image, text, targets, alpha=0, train=True):
        image_embeds = self.blip2.get_image_features(image).pooler_output  #1408
        text_embeds = self.blip2.get_qformer_features(image).pooler_output #768

        # print(image_embeds.shape) #1024
        # print(output.shape) #768
        # image_features = self.image_map(image_embeds)
        # text_features = self.text_map(text_embeds)
        # image_features = F.normalize(image_features, p=2, dim=1)
        # text_features = F.normalize(text_features, p=2, dim=1)

        # features = torch.bmm(image_features.unsqueeze(2), text_features.unsqueeze(1)) # [batch_size, d, d]
        # features = features.reshape(features.shape[0], -1)  # [batch_size, d*d]  #32*32=1024
        #print(features.shape)
        
        # features = self.pre_output(features)
        logits = self.cls_head(torch.cat((image_embeds,text_embeds),dim=1))

        preds_proxy = torch.sigmoid(logits)
        preds = (preds_proxy >= 0.5).long()

        output = {}
        # print(loss_cl)
        # print(logits)
        #print(loss_cl)
        output['loss'] = torch.nn.BCEWithLogitsLoss()(logits, torch.unsqueeze(targets.float(), dim=1))
        #output['accuracy'] = self.acc(preds, targets)
        #output['auroc'] = self.auroc(preds_proxy, targets)
        output['preds'] = preds
        output['preds_proxy'] = preds_proxy

        return output 