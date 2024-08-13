import os
import copy
import time
import pickle
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from utils import get_num_classes, initialize_networks, get_dataloader
from sklearn.mixture import GaussianMixture
from blocks import AdaMixAdapter
from loss import GeneralizedCrossEntropy
import logging
logging.getLogger("imported_module").setLevel(logging.WARNING)

from torch.cuda.amp import GradScaler 
from loss import FocalLoss

class RoutingLoRASolver(object):

    def __init__(self, args, dataset):
        """ Initialize configurations. """
        self.args = args
        self.dataset = dataset
        self.num_class = get_num_classes(args.dataset)

        # Load training networks
        self.model = initialize_networks(dataset=args.dataset, model=args.model, adapter=args.adapter)
        # print(self.model)
        
        # Optimizer
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.device = torch.device('cuda')

    def testing_plm(self, global_loader, model=None):
        writer = {'loss': 0., 'acc': 0., 'step': 0}
        if model is None:
            model = self.model
        net = model.to(self.device)
        net.eval()

        writer['loss'] = 0.
        writer['acc'] = 0.
        writer['step'] = 0.

        predictions = []
        labels = []
        with torch.no_grad():
            for batch in global_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                
                # Summary
                writer['acc'] += torch.eq(logits.argmax(dim=-1), labels).float().sum()
                writer['step'] += len(logits)

        return float(writer['acc'] / writer['step'])


    def JSdivergence(self, output1, output2):
        divergence = (F.kl_div(output1.log_softmax(dim=-1), output2.softmax(dim=-1), reduce=False) + F.kl_div(output2.log_softmax(dim=-1), output1.softmax(dim=-1), reduce=False))/2
        return divergence.mean(dim=-1)


    def modeling_loss(self, model, epoch, dataloader):

        sample_indexes = []
        sample_losses = []
        sample_correct_labels = []
        sample_preds = []
        model = model.eval()
        with torch.no_grad():
            for batch in dataloader:
                indexes = batch['id'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                noisy_labels = batch['noise_label'].to(self.device)

                preds = []
                probs = []
                for _ in range(2):
                    outputs = model(input_ids, attention_mask=attention_mask)
                    logits = outputs.logits
                    preds.append(logits.unsqueeze(0))
                    probs.append(torch.softmax(logits, dim=-1).unsqueeze(0))
                preds = torch.cat(preds, dim=0).mean(dim=0)
                probs = torch.cat(probs, dim=0).mean(dim=0)

                # Model Updates
                loss = (-F.one_hot(noisy_labels, self.num_class) *  preds.log_softmax(dim=-1)).sum(dim=-1)
                sample_indexes.append(indexes)
                sample_losses.append(loss)
                sample_correct_labels.append((labels == noisy_labels).float())
                sample_preds.append(preds)
        sample_preds = torch.cat(sample_preds, dim=0)
        sample_indexes = torch.cat(sample_indexes, dim=0)
        sample_correct_labels = torch.cat(sample_correct_labels, dim=0).cpu()
        sample_losses = torch.cat(sample_losses, dim=0)

        sample_losses = sample_losses.unsqueeze(-1).detach().cpu().numpy()

        confidence_decision, decision_probs = self.estimate_MM(sample_losses, min_prob=0.5, is_min_clean=True)
        # sample_losses /= sample_losses.max()

        self.plot_MM(clean=sample_losses[sample_correct_labels == 1.0],
                     noisy=sample_losses[sample_correct_labels == 0.0], title='{}_Confidence_{}'.format(self.args.dataset, epoch))
        print('[{}]: # of clean samples: {} (acc {})'.format('Confidence', confidence_decision.sum(), sample_correct_labels[confidence_decision == 1].mean()))
        confidence_decision[sample_indexes] = copy.deepcopy(confidence_decision)
        decision_probs[sample_indexes] = copy.deepcopy(decision_probs)
        sample_preds[sample_indexes] = copy.deepcopy(sample_preds)

        return confidence_decision.float(), decision_probs, sample_preds
    
    def modeling_loss_pos_neg(self, pos_model, neg_model, epoch, dataloader):

        sample_indexes = []
        sample_losses = []
        sample_correct_labels = []
        sample_preds = []
        pos_model = pos_model.eval()
        neg_model = neg_model.eval()
        with torch.no_grad():
            for batch in dataloader:
                indexes = batch['id'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                noisy_labels = batch['noise_label'].to(self.device)


                outputs = pos_model(input_ids, attention_mask=attention_mask)
                pos_logits = outputs.logits
                
                # Model Updates
                loss = (-F.one_hot(noisy_labels, self.num_class) *  pos_logits.log_softmax(dim=-1)).sum(dim=-1)
                sample_indexes.append(indexes)
                sample_losses.append(loss)
                sample_correct_labels.append((labels == noisy_labels).float())
                # sample_preds.append(probs)
        # sample_preds = torch.cat(sample_preds, dim=0)
        sample_indexes = torch.cat(sample_indexes, dim=0)
        sample_correct_labels = torch.cat(sample_correct_labels, dim=0).cpu()
        sample_losses = torch.cat(sample_losses, dim=0)
        sample_losses = sample_losses.unsqueeze(-1).detach().cpu().numpy()

        confidence_decision, decision_probs = self.estimate_MM(sample_losses, min_prob=0.5, is_min_clean=True)
        # sample_losses /= sample_losses.max()

        self.plot_MM(clean=sample_losses[sample_correct_labels == 1.0],
                     noisy=sample_losses[sample_correct_labels == 0.0], title='{}_Confidence_{}'.format(self.args.dataset, epoch))
        print('[{}]: # of clean samples: {} (acc {})'.format('Confidence', confidence_decision.sum(), sample_correct_labels[confidence_decision == 1].mean()))
        confidence_decision[sample_indexes] = copy.deepcopy(confidence_decision)
        decision_probs[sample_indexes] = copy.deepcopy(decision_probs)
        # sample_preds[sample_indexes] = copy.deepcopy(sample_preds)
        return confidence_decision.float()#, sample_preds
    

    def estimate_MM(self, data, min_prob, is_min_clean=True):
        gm = GaussianMixture(n_components=2, random_state=0).fit(data)
        if is_min_clean:
            clean_label_index = np.argmin(gm.means_)
        else:
            clean_label_index = np.argmax(gm.means_)

        probs = gm.predict_proba(data)[:, clean_label_index]
        decision = (probs > min_prob)
        decision = torch.from_numpy(decision).float()
        return decision, torch.tensor(probs).float().to(self.device)

    def plot_MM(self, clean, noisy, title):
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker

        fig = plt.figure(figsize=(9, 3))

        plt.style.use('ggplot')
        plt.rcParams['axes.facecolor']='#EAEAF1'
        COLOR = 'black'
        plt.rcParams['text.color'] = COLOR
        plt.rcParams['axes.labelcolor'] = COLOR
        plt.rcParams['xtick.color'] = COLOR
        plt.rcParams['ytick.color'] = COLOR
        plt.rcParams.update({'font.size': 14})

        plt.hist(clean, bins=50, density=True, color='blue', alpha=0.65, label='True identification')
        plt.hist(noisy, bins=50, density=True, color='red', alpha=0.65, label='False identification')
        plt.legend()
        plt.xlabel('Predictive Confidence')
        plt.ylabel('Empirical PDF')
        plt.grid(True)
        plt.savefig('./plot/{}.pdf'.format(title.split('/')[-1]), bbox_inches='tight')
        plt.close()

    def jenson_shannon_divergence(self, net_1_logits, net_2_logits):
        from torch.functional import F
        net_1_probs = F.softmax(net_1_logits, dim=0)
        net_2_probs = F.softmax(net_2_logits, dim=0)
        
        total_m = 0.5 * (net_1_probs + net_1_probs)
        
        loss = 0.0
        loss += F.kl_div(F.log_softmax(net_1_logits, dim=0), total_m, reduce=False).sum(dim=-1)
        loss += F.kl_div(F.log_softmax(net_2_logits, dim=0), total_m, reduce=False).sum(dim=-1)
        return (0.5 * loss)

    def run(self):
        
        """ Start federated learning scenario """
        # Load global validation set
        pos_train_loader, test_loader, _ = get_dataloader(dataset=self.dataset, train_bs=self.args.batch_size, test_bs=self.args.batch_size)

        # self.model = self.load_model('sst5_2_0.4281249940395355')

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr)    
        criterion = nn.CrossEntropyLoss(reduce=False)
        focal_criterion = FocalLoss(alpha=2.0)
        gce = GeneralizedCrossEntropy(self.num_class)
        
        scaler = GradScaler()
        def update_ema_variables(model, ema_model, alpha):
            # Use the true average until the exponential average is more correct
            for ema_param, param in zip(ema_model.parameters(), model.parameters()):
                ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
                
        self.model_hist = []
        self.acc_hist = []

        # print(peft_modules)
        # exit()
        self.model = self.model.to(self.device)
        self.prev_ensemble_preds = None
        training_history = {'cka': [], 'test_acc': [], 'clean_acc': [], 'noise_acc': []}

        # self.model = copy.deepcopy()
        for epoch in range(self.args.epochs):
            writer = {'loss': 0., 'acc': 0., 'step': 0, 'cka': 0, 'clean_acc': 0, 'nclean': 0, 'noise_acc': 0, 'nnoise': 0}
            self.model.train()

            if epoch >= self.args.warm_up:
                decision, decision_probs, ensemble_preds = self.modeling_loss(self.model, epoch, pos_train_loader)
                
            for name, param in self.model.named_parameters():
                if 'fix' not in name and ('lora' in name or 'classifier' in name): 
                    param.requires_grad = True
                else:
                    param.requires_grad = False

            for batch in tqdm(pos_train_loader):

                indexes = batch['id'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                noisy_labels = batch['noise_label'].to(self.device)

                # decision probability updates
                if epoch >= self.args.warm_up:
                    self.model.bert.encoder.routing_probs.data = decision_probs[indexes]
                    self.model.bert.encoder.start_routing = True

                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                # Model Updates
                optimizer.zero_grad()
                if epoch >= self.args.warm_up:
                    temperature = 1.0
                    teacher_preds = (ensemble_preds[indexes]/temperature).softmax(dim=-1)
                    
                    distillation_loss = (-teacher_preds.detach() * (logits/temperature).log_softmax(dim=-1)).sum(dim=-1) * (temperature) ** 2
                    loss = (-F.one_hot(noisy_labels, self.num_class) * logits.log_softmax(dim=-1)).sum(dim=-1) 
                    loss = loss.mean() + distillation_loss.mean()
                else:
                    loss = (-F.one_hot(noisy_labels, self.num_class) * logits.log_softmax(dim=-1)).sum(dim=-1)
                loss = loss.mean()
                loss.backward(retain_graph=True)
                optimizer.step()

                writer['clean_acc']   += torch.eq(logits[noisy_labels == labels].argmax(dim=-1), labels[noisy_labels == labels]).float().sum()
                writer['nclean']   += (noisy_labels == labels).float().sum()
                
                writer['noise_acc']   += torch.eq(logits[noisy_labels != labels].argmax(dim=-1), noisy_labels[noisy_labels != labels]).float().sum()
                writer['nnoise']   += (noisy_labels != labels).float().sum()
                
                writer['step']  += len(logits)
            
            # Evaluate the global model
            self.model.bert.encoder.start_routing = False

            clean_acc = writer['clean_acc'] / writer['nclean']
            noise_acc = writer['noise_acc'] / writer['nnoise']
            test_acc = self.testing_plm(global_loader=test_loader)

            training_history['clean_acc'].append(float(clean_acc))
            training_history['noise_acc'].append(float(noise_acc))
            training_history['test_acc'].append(float(test_acc))

            print('Epoch ({}) Test accuracy {}'.format(epoch, test_acc))
        for key in training_history:
            print(f'{key}=', training_history[key])
            
            # dataset_name = self.args.dataset.split('/')[-1]
            # self.save_model(self.model, f'{dataset_name}_{epoch}_{test_acc}')

    def load_model(self, file_name):
        model = pickle.load(open(os.path.join(self.args.modeldir, f'{file_name}.pkl'), 'rb'))
        return model

    def save_model(self, model, file_name):
        pickle.dump(model, open(os.path.join(self.args.modeldir, f'{file_name}.pkl'), 'wb'))
