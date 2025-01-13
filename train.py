import argparse
import os
import pickle
import pprint

import numpy as np
import torch
import tqdm
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F
from model.model_factory import get_model
from parameters import parser

# from test import *
import test as test
from dataset import CompositionDataset
from utils import *
import sys
### 可视化
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns
from collections import Counter
from torch.cuda.amp import autocast, GradScaler
import shutil
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

def train_model(model, optimizer, config, train_dataset, val_dataset, test_dataset):
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.num_workers
    )

    model.train()
    best_metric = 0
    best_loss = 1e5
    best_epoch = 0
    final_model_state = None
    
    val_results = []
    
    scheduler = get_scheduler(optimizer, config, len(train_dataloader))
    attr2idx = train_dataset.attr2idx
    obj2idx = train_dataset.obj2idx

    train_pairs = torch.tensor([(attr2idx[attr], obj2idx[obj])
                                for attr, obj in train_dataset.train_pairs]).cuda()
                                
    train_losses = []
    
    for i in range(config.epoch_start, config.epochs):
        progress_bar = tqdm.tqdm(
            total=len(train_dataloader), desc="epoch % 3d" % (i + 1)
        )
        
        epoch_train_losses = []
        for bid, batch in enumerate(train_dataloader):

            
            #predict, loss_kg = model(batch, train_pairs, training=True, pairs = train_dataset.train_pairs)
            with autocast():
                predict = model(batch, train_pairs)
                # predict, loss_kg = model(batch, train_pairs, training=True, pairs = train_dataset.train_pairs)

                loss = model.loss_calu(predict, batch) #+ loss_kg

            # normalize loss to account for batch accumulation
            loss = loss / config.gradient_accumulation_steps

            # backward pass
            # loss.backward()
            scaler.scale(loss.type(torch.float)).backward()

            # weights update
            if ((bid + 1) % config.gradient_accumulation_steps == 0) or (bid + 1 == len(train_dataloader)):
                # optimizer.step()
                scaler.step(optimizer)
                scaler.update()

                optimizer.zero_grad()
            scheduler = step_scheduler(scheduler, config, bid, len(train_dataloader))

            epoch_train_losses.append(loss.item())
            progress_bar.set_postfix({"train loss": np.mean(epoch_train_losses[-50:])})
            progress_bar.update()

        progress_bar.close()
        progress_bar.write(f"epoch {i+1} train loss {np.mean(epoch_train_losses)}")
        train_losses.append(np.mean(epoch_train_losses))

        if (i + 1) % config.save_every_n == 0:
            torch.save(model.state_dict(), os.path.join(config.save_path, f"epoch_{i}.pt"))

        print("Evaluating val dataset:")
        val_result = evaluate(model, val_dataset, config)
        val_results.append(val_result)  

        output = "Step [%d] Train loss: %f " % (i+1, np.mean(epoch_train_losses))
        with open(os.path.join(config.save_path, 'logs.txt'),"a+") as f:
            f.write(output+'\n')
            f.write("Evaluating val dataset:" + '\n')
            for k,v in val_result.items():
                f.write(k+':'+str(v)+ ";  ")
            f.write('\n')
            f.close()

        print("Evaluating test dataset:")
        test_result = evaluate(model, test_dataset, config)

        with open(os.path.join(config.save_path, 'logs.txt'),"a+") as f:
            f.write("Evaluating test dataset:" + '\n')
            for k,v in test_result.items():
                f.write(k+':'+str(v)+ ";  ")
            f.write('\n')
            f.close()

        if config.val_metric == 'best_loss' and val_result[config.val_metric] < best_loss:
            best_loss = val_result['best_loss']
            best_epoch = i
            torch.save(model.state_dict(), os.path.join(
                config.save_path, "val_best.pt"))
            
        if config.val_metric != 'best_loss' and val_result[config.val_metric] > best_metric:
            best_metric = val_result[config.val_metric]
            best_epoch = i
            torch.save(model.state_dict(), os.path.join(
                config.save_path,  f"val_best.pt"))

        final_model_state = model.state_dict()
        if i + 1 == config.epochs:
            print("--- Evaluating test dataset on Closed World ---")
            model.load_state_dict(torch.load(os.path.join(
                config.save_path, "val_best.pt"
            )))
            evaluate(model, test_dataset, config)

    if config.save_final_model:
        torch.save(final_model_state, os.path.join(config.save_path, f'final_model.pt'))


def evaluate(model, dataset, config):
    model.eval()
    evaluator = test.Evaluator(dataset, model=None)
    all_logits, all_attr_gt, all_obj_gt, all_pair_gt, loss_avg = test.predict_logits(
            model, dataset, config)
    test_stats = test.test(
            dataset,
            evaluator,
            all_logits,
            all_attr_gt,
            all_obj_gt,
            all_pair_gt,
            config
        )
    test_saved_results = dict()
    result = ""
    key_set = ["best_seen", "best_unseen", "best_hm", "AUC", "attr_acc", "obj_acc"]
    for key in key_set:
        result = result + key + "  " + str(round(test_stats[key], 4)) + "| "
        test_saved_results[key] = round(test_stats[key], 4)
    print(result)
    print('all results\n')
    print(test_stats)
    test_saved_results['best_loss'] = loss_avg
    return test_saved_results


# def save_args(args, log_path, argfile):
#     shutil.copy('train.py', log_path)
#     modelfiles = ospj(log_path, 'models')
#     try:
#         shutil.copy(argfile, log_path)
#     except:
#         print('Config exists')
#     try:
#         shutil.copytree('models/', modelfiles)
#     except:
#         print('Already exists')
#     with open(ospj(log_path,'args_all.yaml'),'w') as f:
#         yaml.dump(args, f, default_flow_style=False, allow_unicode=True)
#     with open(ospj(log_path, 'args.txt'), 'w') as f:
#         f.write('\n'.join(sys.argv[1:]))

if __name__ == "__main__":
    config = parser.parse_args()
    if config.yml_path:
        load_args(config.yml_path, config)

    os.makedirs(config.save_path, exist_ok=True)
    shutil.copy('train.py', config.save_path)
    shutil.copy('./config/troika/{}.yml'.format(config.dataset),config.save_path)
    # shutil.copy('./model/troika_em.py', config.save_path)
    # f = open(os.path.join(config.save_path,'logs.log'),'a')
    # sys.stdout = f
    # sys.stderr = f

    print(config)
    # set the seed value
    set_seed(config.seed)

    dataset_path = config.dataset_path

    train_dataset = CompositionDataset(dataset_path,
                                       phase='train',
                                       split='compositional-split-natural',
                                       same_prim_sample=config.same_prim_sample)

    val_dataset = CompositionDataset(dataset_path,
                                     phase='val',
                                     split='compositional-split-natural')

    test_dataset = CompositionDataset(dataset_path,
                                       phase='test',
                                       split='compositional-split-natural')

    allattrs = train_dataset.attrs
    allobj = train_dataset.objs
    classes = [cla.replace(".", " ").lower() for cla in allobj]
    attributes = [attr.replace(".", " ").lower() for attr in allattrs]
    offset = len(attributes)

    model = get_model(config, attributes=attributes, classes=classes, offset=offset).cuda()
    optimizer = get_optimizer(model, config)

    scaler = GradScaler()
    #############
    # print("Evaluating test dataset:")
    # test_result = evaluate(model, test_dataset, config)
    # val_result = evaluate(model, val_dataset, config)
    
    if config.load_model is not None:
        model.load_state_dict(torch.load(config.load_model))
        print("Evaluating val dataset:")
        val_result = evaluate(model, val_dataset, config)
        print("Evaluating test dataset:")
        test_result = evaluate(model, test_dataset, config)
    #########
    train_model(model, optimizer, config, train_dataset, val_dataset, test_dataset)

    with open(os.path.join(config.save_path, "config.pkl"), "wb") as fp:
        pickle.dump(config, fp)
    write_json(os.path.join(config.save_path, "config.json"), vars(config))
    print("done!")
