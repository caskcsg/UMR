from functools import partial
from transformers import CLIPTextConfig, CLIPVisionConfig
from transformers import CLIPConfig, CLIPModel
from transformers import CLIPVisionModel
from transformers import AutoTokenizer, CLIPTextModel
import torch
from torch import nn
import torch.nn.functional as F
import copy
from torch.distributions import Normal, Independent
from losses import *
import numpy as np
from MLPProcess import MLPEncoder

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

class NCELoss(torch.nn.Module): 
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
        
        self.clip = CLIPModel.from_pretrained("/home/luwang/ycp/HM/ALBEF/clip_large")
        
        self.image_encoder = copy.deepcopy(self.clip.vision_model)
        self.text_encoder = copy.deepcopy(self.clip.text_model)



        self.image_map = nn.Linear(1024, 256)
        self.text_map = nn.Linear(768, 256)
        self.feature_map = nn.Linear(65536, 448)    

        self.img_BayesCap_proj = nn.Linear(1024, 768)
        self.img_BayesCap = BayesCap_MLP(inp_dim=768, out_dim=768, hid_dim=512, num_layers=3, p_drop=0.1)
        self.txt_BayesCap = BayesCap_MLP(inp_dim=768, out_dim=768, hid_dim=512, num_layers=3, p_drop=0.1)

        # self.image_mapmap = nn.Sequential(*[nn.Linear(512, 64), nn.Dropout(p=0.2)])
        # self.text_mapmap = nn.Sequential(*[nn.Linear(512, 64), nn.Dropout(p=0.2)])
        self.mlp_encoder = MLPEncoder(activate='gelu', d_in=[77, 2, 256], d_hiddens=[[50, 2, 64], [10,1,32]], d_outs=[[50, 2, 64], [10,1,32]], dropouts=[0.5,0.5,0.5], bias=False, ln_first=False, res_project=[True,True])
        
        self.Cri = TempCombLoss()
        self.nce_loss = NCELoss()

        #self.pre_output = nn.Sequential(*[nn.Linear(4224, 2112), nn.ReLU(), nn.Dropout(p=0.1)])
        #self.output = nn.Linear(2112, 1)
        self.cls_head = nn.Sequential(
                  nn.Linear(2304, 1252),
                  nn.ReLU(),
                  nn.Linear(1252, 1)#
                )
        
        for _, p in self.image_encoder.named_parameters():
            p.requires_grad_(False)

        for _, p in self.text_encoder.named_parameters():
            p.requires_grad_(False)

        del self.clip

            
    def forward(self, image, text, targets, alpha=0, train=True):
        
        image_embeds = self.image_encoder(image) 
        text_embeds = self.text_encoder(text.input_ids, attention_mask = text.attention_mask, return_dict = True)
        # print(image_embeds.shape) #1024
        # print(output.shape) #768
        image_embeds_mp = self.image_map(image_embeds.last_hidden_state)
        text_embeds_mp = self.text_map(text_embeds.last_hidden_state)


        tmp_img = self.img_BayesCap_proj(image_embeds.last_hidden_state[:,0,:])
        image_features = F.normalize(tmp_img, p=2, dim=1)
        text_features = F.normalize(text_embeds.last_hidden_state[:,0,:], p=2, dim=1)



        x = torch.stack([text_embeds_mp,image_embeds_mp[:,:77,:]], dim=2)
        x = self.mlp_encoder(x, mask=None)

        image_embeds_mp = F.normalize(image_embeds_mp[:,0,:], p=2, dim=1)
        text_embeds_mp = F.normalize(text_embeds_mp[:,0,:], p=2, dim=1)

        features_mf = torch.bmm(image_embeds_mp.unsqueeze(2), text_embeds_mp.unsqueeze(1)) # [batch_size, d, d]
        features_mf = features_mf.reshape(image_features.shape[0], -1)  # [batch_size, d*d]  #32*32=16384
        features_mf = self.feature_map(features_mf)
        features = x.reshape(image_features.shape[0], -1)  # [batch_size, d*d]  #32*32=16384
        #print(features.shape)
        


        features = torch.cat((features, features_mf),dim=1)



        # print(image_features.shape)
        img_mu, img_1alpha, img_beta = self.img_BayesCap(image_features)
        txt_mu, txt_1alpha, txt_beta = self.txt_BayesCap(text_features)

        loss_i, theta_i = self.Cri(img_mu, img_1alpha, img_beta, image_features, T1=1e0, T2=1e-7)  
        loss_t, theta_t = self.Cri(txt_mu, txt_1alpha, txt_beta, text_features, T1=1e0, T2=1e-7)
        #cross modal terms
        loss_i4t, _ = self.Cri(img_mu, img_1alpha, img_beta, text_features, T1=1e0, T2=1e-7)
        loss_t4i, _ = self.Cri(txt_mu, txt_1alpha, txt_beta, image_features, T1=1e0, T2=1e-7)
        
        loss_cl = loss_i + loss_t + 1e-6*(loss_i4t + loss_t4i)
        theta = torch.sigmoid((theta_i + theta_t) / 2)
        #############
        nce_loss = self.nce_loss(image_features, text_features, targets)
        # print(image_features.shape)
        # print(text_features.shape)
        # print(features.shape)
        # print(theta.shape)


        logits = self.cls_head(torch.cat(((1 - theta)*image_features, (1 - theta)*text_features, theta*features), dim=1))
            #print(nce_loss)
        preds_proxy = torch.sigmoid(logits)
        preds = (preds_proxy >= 0.5).long()
        output = {}
        # print(loss_cl)
        # print(logits)
        #print(loss_cl)
        output['loss'] = torch.nn.BCEWithLogitsLoss()(logits, torch.unsqueeze(targets.float(), dim=1)) + 0.01 * loss_cl + 0.01 * nce_loss
        #output['accuracy'] = self.acc(preds, targets)
        #output['auroc'] = self.auroc(preds_proxy, targets)
        output['preds'] = preds
        output['preds_proxy'] = preds_proxy

        return output 