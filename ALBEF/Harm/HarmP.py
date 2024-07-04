import argparse
import os
import ruamel.yaml as yaml
#import ruamel_yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path
import json
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import sys
from model_final import ALBEF
from vit import interpolate_pos_embed
from tokenization_bert import BertTokenizer
sys.path.append("../../")
from transformers import CLIPTokenizer, CLIPTextModel

import utils
from dataset import create_dataset, create_sampler, create_loader
from scheduler import create_scheduler
from optim import create_optimizer

from sklearn.metrics import f1_score,recall_score,precision_score, accuracy_score
from sklearn.metrics import mean_absolute_error



#os.environ["CUDA_VISIBLE_DEVICES"] = "7"
seed = 42 + utils.get_rank()
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
cudnn.benchmark = True

def calculate_mmae(expected, predicted, classes):
    NUM_CLASSES = len(classes)
    # expected = expected.flatten()
    count_dict = {}
    dist_dict = {}
    for i in range(NUM_CLASSES):
        count_dict[i] = 0
        dist_dict[i] = 0.0
    for i in range(len(expected)):
        dist_dict[expected[i]] += abs(expected[i] - predicted[i])
        count_dict[expected[i]] += 1
    overall = 0.0
    for claz in range(NUM_CLASSES): 
        class_dist =  1.0 * dist_dict[claz] / count_dict[claz] 
        overall += class_dist
    overall /= NUM_CLASSES
#     return overall[0]
    return overall


def train(model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, scheduler, config):
    # train
    model.train()  
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.9f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50   
    step_size = 100
    warmup_iterations = warmup_steps*step_size  
 
    for i,(images, text, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        images, targets = images.to(device,non_blocking=True), targets.to(device,non_blocking=True)
        
        text_inputs = tokenizer(text, padding='max_length', max_length=100, truncation=True, return_tensors="pt").to(device) 
        
        if epoch>0 or not config['warm_up']:
            alpha = config['alpha']
        else:
            alpha = config['alpha']*min(1,i/len(data_loader))

        output = model(images, text_inputs, targets=targets, train=True, alpha=alpha)    
        loss = output['loss']
        optimizer.zero_grad()

        # with amp.scale_loss(loss, optimizer) as scaled_loss:
        #     scaled_loss.backward()

        loss.backward()
        optimizer.step()    
               
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(loss=loss.item())
        
        if epoch==0 and i%step_size==0 and i<=warmup_iterations: 
            scheduler.step(i//step_size)         
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.4f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}    


@torch.no_grad()
def evaluate(model, data_loader, tokenizer, device, config):
    # test
    model.eval()
            
    metric_logger = utils.MetricLogger(delimiter="  ")

    header = 'Evaluation:'
    print_freq = 50

    pre_list=[]
    label_list=[]

    for images, text, targets in metric_logger.log_every(data_loader, print_freq, header):
        
        images, targets = images.to(device,non_blocking=True), targets.to(device,non_blocking=True)   
        
        text_inputs = tokenizer(text, padding='max_length', max_length=100, truncation=True, return_tensors="pt").to(device) 

        output = model(images, text_inputs, targets=targets, train=False) 
        
        pred_class = output['preds']
        #高版本pytorch的tensor和int之间的除法不能直接用'/'
        accuracy = torch.true_divide((torch.unsqueeze(targets.float(), dim=1)==pred_class).sum(), targets.size(0))  
        metric_logger.meters['acc'].update(accuracy.item(), n=images.size(0))
    
        pre_list.append(output['preds'])
        label_list.append(targets)

    pre_list = torch.cat(pre_list, 0)
    
    # for i in pre_list : 
    #     if i >= 0.5 :
    #         pre.append(1) 
    #     else:
    #         pre.append(0) 
    label_list = torch.cat(label_list, 0)
    # pre_list = torch.tensor(pre_list).to(device) 
    # label_list = torch.tensor(label_list).to(device) 
    # print(pre_list)
    # print(label_list)
    # sklearn中的auc_roc计算
    #pre_list = F.softmax(pre_list, dim=-1)[:, 1]#[size, 1]
    # label_list = label_list.unsqueeze(1) 
    # f1 = torchmetrics.F1Score(task="binary", average='macro')(pre_list.cpu(), label_list.cpu())
    # rec = torchmetrics.Recall(task="binary", average='macro')(pre_list.cpu(), label_list.cpu())
    # pre = torchmetrics.Precision(task="binary", average='macro')(pre_list.cpu(), label_list.cpu())
    # mae = torchmetrics.MeanAbsoluteError()(pre_list.cpu(), label_list.cpu())
    # mmae = torchmetrics.M
    f1 = f1_score(label_list.cpu().numpy(), pre_list.cpu().numpy(), average='macro')
    rec = recall_score(label_list.cpu().numpy(), pre_list.cpu().numpy(), average='macro')
    pre = precision_score(label_list.cpu().numpy(), pre_list.cpu().numpy(), average='macro')
    mae = mean_absolute_error(label_list.cpu().numpy(), pre_list.cpu().numpy())
    mmae = calculate_mmae(label_list.cpu().numpy(), pre_list.cpu().numpy(), [0,1])
    #acc = acc.item()
    # metrics中auc_roc计算
    f1 = f1.item()
    rec = rec.item()
    pre = pre.item()
    mae = mae.item()
    mmae = mmae.item()
    #print(acc)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())   
    print("Averaged stats: f1: ", f1)  
    print("Averaged stats: rec: ", rec)  
    print("Averaged stats: pre: ", pre)  
    print("Averaged stats: mae: ", mae)  
    print("Averaged stats: mmae: ", mmae) 

    score_dic = {k: "{:.4f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}
    score_dic['f1'] = f1
    score_dic['rec'] = rec
    score_dic['pre'] = pre
    score_dic['mae'] = mae
    score_dic['mmae'] = mmae

    return score_dic
    
    
def main(args, config):
    utils.init_distributed_mode(args)    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    

    #### Dataset #### 
    print("Creating dataset")
    #需要更改
    datasets = create_dataset('harmC', config) 
    
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler(datasets, [True, False, False], num_tasks, global_rank)         
    else:
        samplers = [None, None, None]

    train_loader, val_seen_loader, test_seen_loader = create_loader(datasets,samplers,
                                                          batch_size=[config['batch_size_train']]+[config['batch_size_test']]*2,
                                                          num_workers=[2,2,2],is_trains=[True,False,False], 
                                                          collate_fns=[None,None,None])

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    #### Model #### 
    print("Creating model")
    #需要更改
    model = ALBEF(config=config, text_encoder="bert-base-uncased", tokenizer=tokenizer)
    
    if args.checkpoint:    
        checkpoint = torch.load(args.checkpoint, map_location='cpu') 
        state_dict = checkpoint['model']
        
        # reshape positional embedding to accomodate for image resolution change
        pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],model.visual_encoder)         
        state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped
        
        if not args.evaluate:
            if config['distill']:
                m_pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'],model.visual_encoder_m)   
                state_dict['visual_encoder_m.pos_embed'] = m_pos_embed_reshaped 

            for key in list(state_dict.keys()):                
                if 'bert' in key:
                    new_key = key.replace('bert.','')
                    state_dict[new_key] = state_dict[key] 
                    del state_dict[key]
                
        msg = model.load_state_dict(state_dict,strict=False)
        print('load checkpoint from %s'%args.checkpoint)
        print(msg)    
 
    model = model.to(device)   
    
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module     
    
    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)
    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer, flag=False)  
    
    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']
    best_seen = 0
    best_seen_epoch = 0
    
    best_seen_f1 = 0
    best_seen_f1_epoch = 0

    best_seen_rec = 0
    best_seen_rec_epoch = 0

    best_seen_pre = 0
    best_seen_pre_epoch = 0

    best_seen_mae = 1
    best_seen_mae_epoch = 0

    best_seen_mmae = 1
    best_seen_mmae_epoch = 0

    print("Start training")
    start_time = time.time()


    for epoch in range(0, max_epoch):
        if not args.evaluate:
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
            train_stats = train(model, train_loader, optimizer, tokenizer, epoch, warmup_steps, device, lr_scheduler, config)  

        val_seen_stats = evaluate(model, val_seen_loader, tokenizer, device, config)
        test_seen_stats = evaluate(model, test_seen_loader, tokenizer, device, config)

        if utils.is_main_process():  
            if args.evaluate:
                log_stats = {**{f'val_seen_{k}': v for k, v in val_seen_stats.items()},
                            **{f'test_seen_{k}': v for k, v in test_seen_stats.items()},
                            'epoch': epoch,
                            }

                with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                    f.write(json.dumps(log_stats) + "\n")                
            else:    
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                            **{f'val_seen_{k}': v for k, v in val_seen_stats.items()},
                            **{f'test_seen_{k}': v for k, v in test_seen_stats.items()},
                            'epoch': epoch,
                            }

                with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                if float(val_seen_stats['acc'])>best_seen:
                    save_obj = {
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'config': config,
                        'epoch': epoch,
                    }
                    torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_seen_best_acc.pth')) 
                    best_seen = float(val_seen_stats['acc'])
                    best_seen_epoch=epoch

                if float(val_seen_stats['f1'])>best_seen_f1:
                    save_obj = {
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'config': config,
                        'epoch': epoch,
                    }
                    torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best_val_seen_f1.pth')) 
                    best_seen_f1 = float(val_seen_stats['f1'])
                    best_seen_f1_epoch = epoch

                if float(val_seen_stats['rec'])>best_seen_rec:
                    save_obj = {
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'config': config,
                        'epoch': epoch,
                    }
                    torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best_val_seen_rec.pth')) 
                    best_seen_rec = float(val_seen_stats['rec'])
                    best_seen_rec_epoch = epoch

                if float(val_seen_stats['pre'])>best_seen_pre:
                    save_obj = {
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'config': config,
                        'epoch': epoch,
                    }
                    torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best_val_seen_pre.pth')) 
                    best_seen_pre = float(val_seen_stats['pre'])
                    best_seen_pre_epoch = epoch
                
                if float(val_seen_stats['mae'])<best_seen_mae:
                    save_obj = {
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'config': config,
                        'epoch': epoch,
                    }
                    torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best_val_seen_mae.pth')) 
                    best_seen_mae = float(val_seen_stats['mae'])
                    best_seen_mae_epoch = epoch

                if float(val_seen_stats['mmae'])<best_seen_mmae:
                    save_obj = {
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'config': config,
                        'epoch': epoch,
                    }
                    torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best_val_seen_mmae.pth')) 
                    best_seen_mmae = float(val_seen_stats['mmae'])
                    best_seen_mmae_epoch = epoch

        if args.evaluate:
            break
        lr_scheduler.step(epoch+warmup_steps+1)  
                
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 

    if utils.is_main_process():   
        with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
            f.write("best val seen acc step: %d"%best_seen_epoch)         
            f.write("best val seen f1 step: %d"%best_seen_f1_epoch)
            f.write("best val seen rec step: %d"%best_seen_rec_epoch)
            f.write("best val seen pre step: %d"%best_seen_pre_epoch)
            f.write("best val seen mae step: %d"%best_seen_mae_epoch)
            f.write("best val seen mmae step: %d"%best_seen_mmae_epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='/workspace/project/ALBEF/configs/Hateful.yaml')
    parser.add_argument('--output_dir', default='/workspace/project/ALBEF/output/Hateful_test')  
    parser.add_argument('--checkpoint', default='/workspace/project/ALBEF/output/Hateful/ALBEF.pth')   
    parser.add_argument('--text_encoder', default='/home/yangcp/ALBEF/bert-base-uncased')
#   例如 python train.py --eval  那么你用了这个eval 那这个eval就是true
#   如果 python train.py   你没有用那个 eval 此时 eval 为false
    parser.add_argument('--evaluate', action='store_true')    
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=False, type=bool)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    #将hateful配置文件输出一份到output
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)
