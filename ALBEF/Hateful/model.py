from functools import partial
from transformers import CLIPTextConfig, CLIPVisionConfig
from transformers import CLIPConfig
from transformers import CLIPVisionModel
from functools import partial
from vit import VisionTransformer
from xbert import BertConfig, BertModel
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

        self.tokenizer = tokenizer 

        self.visual_encoder = VisionTransformer(
            img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12, 
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))    
        bert_config = BertConfig.from_json_file(config['bert_config'])
        self.text_encoder = BertModel.from_pretrained(text_encoder, config=bert_config, add_pooling_layer=False)          

        # self.visual_encoder_ml = copy.deepcopy(self.visual_encoder)
        # self.text_encoder_ml = copy.deepcopy(self.text_encoder)

        self.image_map = nn.Linear(768, 64)
        self.text_map = nn.Linear(768, 64)
        self.img_BayesCap = BayesCap_MLP(inp_dim=64, out_dim=64, hid_dim=32, num_layers=3, p_drop=0.1)
        self.txt_BayesCap = BayesCap_MLP(inp_dim=64, out_dim=64, hid_dim=32, num_layers=3, p_drop=0.1)

        # self.image_mapmap = nn.Sequential(*[nn.Linear(512, 64), nn.Dropout(p=0.2)])
        # self.text_mapmap = nn.Sequential(*[nn.Linear(512, 64), nn.Dropout(p=0.2)])
        
        self.Cri = TempCombLoss()

        self.softplus = nn.Softplus()
        self.target_con = 1.0
        self.kl_c = -1.0
        self.fisher_c = 1.0

        self.nce_loss = NCELoss()

        #self.pre_output = nn.Sequential(*[nn.Linear(4224, 2112), nn.ReLU(), nn.Dropout(p=0.1)])
        #self.output = nn.Linear(2112, 1)
        self.cls_head = nn.Sequential(
                  nn.Linear(768, 768),
                  nn.ReLU(),
                  nn.Linear(768, 1)#ve是三分类需要改成二分类
                )
        
        # for _, p in self.visual_encoder_ml.named_parameters():
        #     p.requires_grad_(False)

        # for _, p in self.text_encoder_ml.named_parameters():
        #     p.requires_grad_(False)

        #del self.clip



    def compute_mse(self, labels_1hot_, evi_alp_):
        evi_alp0_ = torch.sum(evi_alp_, dim=-1, keepdim=True)

        loss_mse_ = (labels_1hot_ - evi_alp_ / evi_alp0_).pow(2).sum(-1).mean()
        loss_var_ = (evi_alp_ * (evi_alp0_ - evi_alp_) / (evi_alp0_ * evi_alp0_ * (evi_alp0_ + 1))).sum(
            -1).mean()

        return loss_mse_, loss_var_

    def compute_fisher_mse(self, labels_1hot, evi_alp):
        evi_alp0 = torch.sum(evi_alp, dim=-1, keepdim=True)
        gamma1_alp = torch.polygamma(1, evi_alp)
        gamma1_alp0 = torch.polygamma(1, evi_alp0)

        gap = labels_1hot - evi_alp / evi_alp0

        loss_mse = (gap.pow(2) * gamma1_alp).sum(-1).mean()

        loss_var = (evi_alp * (evi_alp0 - evi_alp) * gamma1_alp / (evi_alp0 * evi_alp0 * (evi_alp0 + 1))).sum(-1).mean()
        # print(torch.log(gamma1_alp).sum(-1), 1- (gamma1_alp0 / gamma1_alp).sum(-1))

        loss_det_fisher = - (torch.log(gamma1_alp).sum(-1) + torch.log((gamma1_alp0 / gamma1_alp).sum(-1))).mean()

        return loss_mse, loss_var, loss_det_fisher

    def compute_kl_loss(self, alphas, labels, target_concentration, concentration=1.0, epsilon=1e-8):
        # TODO: Need to make sure this actually works right...
        # todo: so that concentration is either fixed, or on a per-example setup

        # Create array of target (desired) concentration parameters
        if target_concentration < 1.0:
            concentration = target_concentration

        target_alphas = torch.ones_like(alphas) * concentration
        target_alphas += torch.zeros_like(alphas).scatter_(0, labels.unsqueeze(-1), target_concentration - 1)

        alp0 = torch.sum(alphas, dim=-1, keepdim=True)
        target_alp0 = torch.sum(target_alphas, dim=-1, keepdim=True)

        alp0_term = torch.lgamma(alp0 + epsilon) - torch.lgamma(target_alp0 + epsilon)
        alp0_term = torch.where(torch.isfinite(alp0_term), alp0_term, torch.zeros_like(alp0_term))
        assert torch.all(torch.isfinite(alp0_term)).item()

        alphas_term = torch.sum(torch.lgamma(target_alphas + epsilon) - torch.lgamma(alphas + epsilon)
                                + (alphas - target_alphas) * (torch.digamma(alphas + epsilon) -
                                                              torch.digamma(alp0 + epsilon)), dim=-1, keepdim=True)
        alphas_term = torch.where(torch.isfinite(alphas_term), alphas_term, torch.zeros_like(alphas_term))
        assert torch.all(torch.isfinite(alphas_term)).item()

        loss = torch.squeeze(alp0_term + alphas_term).mean()

        return loss
            
    def forward(self, image, text, targets, alpha=0, train=True):
        
        image_embeds, image_state = self.visual_encoder(image) 
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)        
        
        text_embeds = self.text_encoder(text.input_ids, 
                                       attention_mask = text.attention_mask, 
                                       encoder_hidden_states = image_embeds,
                                       encoder_attention_mask = image_atts,        
                                       return_dict = True
                                      ) 
        image_embeds = image_embeds[:,0,:]
        text_embeds = text_embeds.last_hidden_state[:,0,:]
        # print(image_embeds.shape) #1024
        # print(output.shape) #768
        image_features = self.image_map(image_embeds)
        text_features = self.text_map(text_embeds)
        image_features = F.normalize(image_features, p=2, dim=1)
        text_features = F.normalize(text_features, p=2, dim=1)

        features = torch.bmm(image_features.unsqueeze(2), text_features.unsqueeze(1)) # [batch_size, d, d]
        features = features.reshape(features.shape[0], -1)  # [batch_size, d*d]  #32*32=1024
        #print(features.shape)
        
        # features = self.pre_output(features)
        # logits = self.cls_head(torch.cat((features, text_embeds), dim=1))
        logits = self.cls_head(text_embeds)##text_embeds必不可少

        preds_proxy = torch.sigmoid(logits)
        preds = (preds_proxy >= 0.5).long()

        loss_cl = 0
        grad_loss = 0
        nce_loss = 0
        if train:

            img_mu, img_1alpha, img_beta = self.img_BayesCap(image_features)
            txt_mu, txt_1alpha, txt_beta = self.txt_BayesCap(text_features)

            loss_i = self.Cri(img_mu, img_1alpha, img_beta, image_features, T1=1e0, T2=1e-4)  #改成1e-4损失会变小
            loss_t = self.Cri(txt_mu, txt_1alpha, txt_beta, text_features, T1=1e0, T2=1e-4)
            #cross modal terms
            loss_i4t = self.Cri(img_mu, img_1alpha, img_beta, text_features, T1=1e0, T2=1e-4)
            loss_t4i = self.Cri(txt_mu, txt_1alpha, txt_beta, image_features, T1=1e0, T2=1e-4)
            
            loss_cl = loss_i + loss_t + 1e-6*(loss_i4t + loss_t4i)


            evi_alp = self.softplus(logits) + 1.0
            # Calculate loss
            # print(torch.zeros_like(preds_proxy).shape)
            # print(targets.unsqueeze(-1).shape)
            
            labels_1hot = torch.zeros_like(logits).scatter_(0, targets.unsqueeze(-1), 0)
                # IEDL -> fisher_mse
            loss_mse, loss_var, loss_fisher = self.compute_fisher_mse(labels_1hot, evi_alp)

            evi_alp = (evi_alp - self.target_con) * (1 - labels_1hot) + self.target_con
            
            loss_kl = self.compute_kl_loss(evi_alp, targets, self.target_con)

            grad_loss = loss_mse + loss_var + self.fisher_c * loss_fisher + 1.0 * loss_kl
            #############
            nce_loss = self.nce_loss(image_features, text_features, targets)
            #print(nce_loss)

        
        output = {}
        # print(loss_cl)
        # print(logits)
        #print(loss_cl)
        output['loss'] = torch.nn.BCEWithLogitsLoss()(logits, torch.unsqueeze(targets.float(), dim=1)) + 0.0001 * loss_cl + 0.1 * grad_loss  +  0.01 * nce_loss
        #output['accuracy'] = self.acc(preds, targets)
        #output['auroc'] = self.auroc(preds_proxy, targets)
        output['preds'] = preds
        output['preds_proxy'] = preds_proxy

        return output 