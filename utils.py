from lib2to3.pgen2 import token
import os
import sys
import copy
import pickle
import numpy as np
import pandas as pd

# from data.sampler import SubsetSequentialSampler
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

# Huggingface imports
# import datasets
import loralib
from datasets import load_dataset, Dataset, concatenate_datasets, DatasetDict
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, AutoModel
from transformers import AutoModelForTokenClassification, AutoModelForMultipleChoice
# from transformers.adapters import CompacterConfig

from transformer_model import BaseAdapterTransformer, BaseAdapterBertEncoder, BertAdapterSelfOutput, BertAdapterOutput, BertSelectOutput, \
                                BertSelectSelfOutput, BertMixAdapterOutput, BertMixAdapterSelfOutput, MixClassifier, \
                                BertFusionAdapterOutput, BertFusionAdapterSelfOutput, FusionClassifier, BertPrefixEncoder, BertRoutingPrefixEncoder, BertRoutingBitFitEncoder, BertRoutingLoRAEncoder
# from blocks import BaseAdapterTransformerBlock, NormalizedAdapterTransformerBlock

# Global variable
nlp_dataset = None

class SubsetSequentialSampler(torch.utils.data.Sampler):
    r"""Samples elements sequentially from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (int(self.indices[i]) for i in range(len(self.indices)))
    
    def __len__(self):
        return len(self.indices)  

class Logger(object):
    def __init__(self, location):
        self.terminal = sys.stdout
        self.log = open(location, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass
        

def set_random_seed(seed):
    import torch
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed+1)
    torch.manual_seed(seed+2)
    torch.cuda.manual_seed(seed+3)
    torch.cuda.manual_seed_all(seed+4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_num_classes(dataset: str) -> int:
    """
    Return the number of classes in the given dataset
    Args:
        dataset: Dataset name (e.g., cifar10, cifar100)
    Returns:
        the number of classes
    """
    n_classes = 0
    if dataset == 'ag_news':
        n_classes = 4
    elif dataset == 'hatexplain':
        n_classes = 3
    elif dataset == 'trec':
        n_classes = 6
    elif dataset == 'SetFit/sst5':
        n_classes = 5
    elif dataset == 'tweet_eval':
        n_classes = 20
    elif dataset == 'SetFit/20_newsgroups':
        n_classes = 20
    elif dataset == 'multi_sent':
        n_classes = 10
    elif dataset == 'wiki':
        n_classes = 158
    elif dataset == 'ner':
        n_classes = 9
    elif dataset == 'qa':
        n_classes = 1
    elif dataset == 'real':
        n_classes = 5
    elif dataset == 'PolyAI/banking77':
        n_classes = 77
    if not n_classes:
        print('No {} dataset in data directory'.format(dataset))
        exit()
    return n_classes


def get_tokenizer(model: str, max_length: int):
    return AutoTokenizer.from_pretrained(model, model_max_length=max_length)


def other_class(n_classes, current_class):
    """
    Returns a list of class indices excluding the class indexed by class_ind
    :param nb_classes: number of classes in the task
    :param class_ind: the class index to be omitted
    :return: one random class that != class_ind
    """
    if current_class < 0 or current_class >= n_classes:
        error_str = "class_ind must be within the range (0, nb_classes - 1)"
        raise ValueError(error_str)

    other_class_list = list(range(n_classes))
    other_class_list.remove(current_class)
    other_class = np.random.choice(other_class_list)
    return other_class

    

def make_noisy_general(clean_data, noise_matrix, num_classes):
    for row in noise_matrix:
        assert np.isclose(np.sum(row), 1)

    assert len(noise_matrix) == num_classes

    noisy_data = copy.deepcopy(clean_data)
    for i in range(len(noisy_data)):
        probability_row = noise_matrix[noisy_data[i]]
        noisy_data[i] = np.random.choice(num_classes, p=probability_row)
    return noisy_data

def make_noisy_uniform(y, noise_level, num_classes, noise_type='sym'):
    # assert num_classes == len(set(y))
    clean_label_probability = 1 - noise_level
    uniform_noise_probability = noise_level / num_classes  # distribute noise_level across all other labels
    clean_label_probability += uniform_noise_probability

    if noise_type == 'sym':
        true_noise_matrix = np.empty((num_classes, num_classes))
        true_noise_matrix.fill(uniform_noise_probability)
        for true_label in range(num_classes):
            true_noise_matrix[true_label][true_label] = clean_label_probability
    elif noise_type == 'asym':
        true_noise_matrix = get_single_flip_mat(noise_level, num_classes)
    else:
        print('no specified noise types', noise_type)
        exit(0)
        
    noisy_data = make_noisy_general(y, true_noise_matrix, num_classes)
    return noisy_data

def get_single_flip_mat(noise_level, num_classes):
    flips = np.arange(num_classes)
    flips = np.roll(flips, 1)

    true_noise_matrix = np.zeros((num_classes, num_classes))
    for true_label in range(num_classes):
        true_noise_matrix[true_label][true_label] = 1 - noise_level
        true_noise_matrix[true_label][flips[true_label]] = noise_level
    return true_noise_matrix

def load_noisy_ner_dataset(dataset: str, model: str, noise_rate: float = 0.3, max_length: int = 128, num_class=20):
    global nlp_dataset
    tokenizer = get_tokenizer(model=model, max_length=max_length)
    
    
    tag_dict = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8}
    
    # load text
    train_raw_text = []
    train_raw_labels = []
    with open('./RoSTER/data/conll/train_text.txt', 'r') as f:
        for line in f:
            line = line[:-1]
            train_raw_text.append(line.split(' '))
    with open('./RoSTER/data/conll/train_label_dist.txt', 'r') as f:
        for line in f:
            line = line[:-1]
            train_raw_labels.append([tag_dict[t] for t in line.split(' ')])
            
    test_raw_text = []
    test_raw_labels = []
    with open('./RoSTER/data/conll/valid_text.txt', 'r') as f:
        for line in f:
            line = line[:-1]
            test_raw_text.append(line.split(' '))
    with open('./RoSTER/data/conll/valid_label_true.txt', 'r') as f:
        for line in f:
            line = line[:-1]
            test_raw_labels.append([tag_dict[t] for t in line.split(' ')])

    train_data = Dataset.from_dict({"tokens" : train_raw_text, "ner_tags" : train_raw_labels})
    test_data = Dataset.from_dict({"tokens" : test_raw_text, "ner_tags" : test_raw_labels})

    raw_dataset = DatasetDict(
        {
            "train" : train_data, "test" : test_data
        }
    )                             
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(examples["tokens"], padding='max_length', truncation=True, is_split_into_words=True)

        labels = []
        for i, label in enumerate(examples[f"ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:  # Set the special tokens to -100.
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs
    

    tokenized_dataset = raw_dataset.map(tokenize_and_align_labels, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(['tokens', 'ner_tags'])

    nlp_dataset = tokenized_dataset
    df_train = pd.DataFrame(nlp_dataset['train'])
    df_train['id'] = [i for i in range(len(nlp_dataset['train']))]

    nlp_dataset['train'] = Dataset.from_pandas(df_train)
    nlp_dataset.set_format("torch")
    return nlp_dataset

# def load_noisy_ner_dataset(dataset: str, model: str, noise_rate: float = 0.3, max_length: int = 128, num_class=20):
#     global nlp_dataset
#     tokenizer = get_tokenizer(model=model, max_length=max_length)

#     def tokenize_and_align_labels(examples):
#         tokenized_inputs = tokenizer(examples["tokens"], padding='max_length', truncation=True, is_split_into_words=True)

#         labels = []
#         for i, label in enumerate(examples[f"ner_tags"]):
#             word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
#             previous_word_idx = None
#             label_ids = []
#             for word_idx in word_ids:  # Set the special tokens to -100.
#                 if word_idx is None:
#                     label_ids.append(-100)
#                 elif word_idx != previous_word_idx:  # Only label the first token of a given word.
#                     label_ids.append(label[word_idx])
#                 else:
#                     label_ids.append(-100)
#                 previous_word_idx = word_idx
#             labels.append(label_ids)

#         tokenized_inputs["labels"] = labels
#         return tokenized_inputs
    
#     raw_dataset = load_dataset('phucdev/noisyner', f'NoisyNER_labelset{int(noise_rate)}')
#     clean_dataset = load_dataset('phucdev/noisyner', 'estner_clean')

#     raw_dataset['test'] = clean_dataset['test']

#     tokenized_dataset = raw_dataset.map(tokenize_and_align_labels, batched=True)
#     tokenized_dataset = tokenized_dataset.remove_columns(['tokens', 'lemmas', 'grammar', 'ner_tags'])

#     nlp_dataset = tokenized_dataset
#     df_train = pd.DataFrame(nlp_dataset['train'])
#     df_train['id'] = [i for i in range(len(nlp_dataset['train']))]

#     nlp_dataset['train'] = Dataset.from_pandas(df_train)
#     nlp_dataset.set_format("torch")
#     return nlp_dataset


# def load_noisy_ner_dataset(dataset: str, model: str, noise_rate: float = 0.3, max_length: int = 128, num_class=20):
#     global nlp_dataset
#     tokenizer = get_tokenizer(model=model, max_length=max_length)

#     def tokenize_and_align_labels(examples):
#         tokenized_inputs = tokenizer(examples["tokens"], padding='max_length', truncation=True, is_split_into_words=True)

#         labels = []
#         for i, label in enumerate(examples[f"ner_tags"]):
#             word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
#             previous_word_idx = None
#             label_ids = []
#             for word_idx in word_ids:  # Set the special tokens to -100.
#                 if word_idx is None:
#                     label_ids.append(-100)
#                 elif word_idx != previous_word_idx:  # Only label the first token of a given word.
#                     label_ids.append(label[word_idx])
#                 else:
#                     label_ids.append(-100)
#                 previous_word_idx = word_idx
#             labels.append(label_ids)

#         tokenized_inputs["labels"] = labels
#         return tokenized_inputs
    
#     raw_dataset = load_dataset('conll2003')

#     tokenized_dataset = raw_dataset.map(tokenize_and_align_labels, batched=True)
#     tokenized_dataset = tokenized_dataset.remove_columns(['tokens', 'pos_tags', 'chunk_tags', 'ner_tags'])

#     nlp_dataset = tokenized_dataset
#     df_train = pd.DataFrame(nlp_dataset['train'])
#     df_train['id'] = [i for i in range(len(nlp_dataset['train']))]

#     # r = 0.3
#     a = torch.ones(num_class, num_class) * (noise_rate / (num_class-1))
#     for i in range(len(a)):
#         a[i, i] = 1-noise_rate
#     a = a.numpy().astype('float64')
#     for i in range(len(a)):
#         a[i] /= a[i].sum()

#     noisy_labels = []
#     for i in range(len(df_train)):
#         current_labels = df_train.iloc[i]['labels']
#         corrupted_labels = []
#         for tag_id in current_labels:
#             if tag_id == -100:
#                 corrupted_labels.append(tag_id)
#                 continue
#             # random flipping
#             # print(a[tag_id].sum())
#             flip_tag_id = np.random.choice(num_class, 1, p=a[tag_id])
#             corrupted_labels.append(flip_tag_id)
#         noisy_labels.append(current_labels)
        
#     df_train['id'] = [i for i in range(len(nlp_dataset['train']))]
#     df_train['noisy_labels'] = noisy_labels

#     nlp_dataset['train'] = Dataset.from_pandas(df_train)
#     nlp_dataset.set_format("torch")
#     return nlp_dataset

def load_real_noise_dataset(dataset: str, datadir: str, model: str, noise_rate: float = 0.3, max_length: int = 128, num_class=20):
    global nlp_dataset
    
    tokenizer = get_tokenizer(model=model, max_length=max_length)
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True)

    train_data = pd.read_csv('./data/hausa.tsv',sep='\t')
    label_set = train_data['label'].unique()
    label_dict = {}
    for l in label_set:
        label_dict[l] = len(label_dict)
    label_ids = [label_dict[l] for l in train_data['label']]
    ids = [i for i in range(len(train_data['label']))]

    train_data = Dataset.from_dict({"text" : train_data['news_title'], "label" : label_ids, "noise_label" : label_ids, "id": ids})

    test_data = pd.read_csv('./data/hausa_test.tsv',sep='\t')
    label_ids = [label_dict[l] for l in test_data['label']]
    ids = [i for i in range(len(test_data['label']))]

    test_data = Dataset.from_dict({"text" : test_data['news_title'], "label" : label_ids, "noise_label" : label_ids, "id": ids})
    raw_dataset = DatasetDict(
        {
            "train" : train_data, "test" : test_data
        }
    )           
    nlp_dataset = raw_dataset.map(tokenize_function, batched=True)
    nlp_dataset.set_format("torch")
    return nlp_dataset


def load_noise_dataset(dataset: str, datadir: str, model: str, noise_rate: float = 0.3, noise_type: str = 'sym', max_length: int = 128, num_class=20):
    global nlp_dataset
    
    tokenizer = get_tokenizer(model=model, max_length=max_length)
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True)
    
    def mode(lst):
        return max(set(lst), key=lst.count)
    
    def update_data_hatexplain(example):
        example["label"] = mode(example["annotators"]["label"])
        example["text"] = " ".join(example["post_tokens"]) # content
        return example

    if 'sst' in dataset or 'hatexplain' in dataset or 'bank' in dataset:
        raw_dataset = load_dataset(dataset)
        if 'hatexplain' in dataset:
            raw_dataset = raw_dataset.map(update_data_hatexplain)
        # raw_dataset = raw_dataset.remove_columns(['id', 'annotators', 'rationales', 'post_tokens'])

    elif 'wiki' in dataset:
        no = round(noise_rate,1)
        with open("./NoisywikiHow-dataset/cat158.csv", "r") as f:
            y = f.readlines()

        label2id = dict()
        for idx, label in enumerate(y[1:]):
            label = label.split(",")[0]
            label2id[label] = idx

        pand_x = pd.read_csv(f"./NoisywikiHow-dataset/noisy/mix_{round(noise_rate, 1)}.csv")
        val = pd.read_csv("./NoisywikiHow-dataset/noisy/val.csv")
        test = pd.read_csv("./NoisywikiHow-dataset/noisy/test.csv")
        
        idx = 0
        mixed_idx= 0
        sentences = []
        labels = []
        noisy_labels = []
        is_true = []
        for s, l, n_l, n_s, t in zip(pand_x["step"], pand_x["cat"], pand_x["noisy_cat"], pand_x["noisy_step"], pand_x["noisy_label"]):
            idx += 1
            '''
            if t == -1:
                mixed_idx += 1
                sentences.append(s)
                labels.append(label2id[l])
            else:        
                sentences.append(n_s)            
                labels.append(label2id[n_l])
            '''
            # if t == -1:
            sentences.append(n_s)
            labels.append(label2id[l])
            noisy_labels.append(label2id[l])
                
            if l == n_l:
                is_true.append(1)
            else:
                mixed_idx += 1
                is_true.append(0)
        
        print(mixed_idx / idx)
        
        val_sentences = []
        val_labels = []
        for s, l in zip(val["step"], val["cat"]):
            val_sentences.append(s)
            val_labels.append(label2id[l])

        train_data = Dataset.from_dict({"text" : sentences, "label" : labels, "noise_label": noisy_labels})
        test_data = Dataset.from_dict({"text" : val_sentences, "label" : val_labels})

        raw_dataset = DatasetDict(
            {
                "train" : train_data, "test" : test_data
            }
        )                 
                
    tokenized_dataset = raw_dataset.map(tokenize_function, batched=True)
    
    df_train = pd.DataFrame(tokenized_dataset['train'])
    y_train = df_train['label'].to_numpy()
    noisy_y_train = np.copy(y_train)

    nlp_dataset = tokenized_dataset
    df_train = pd.DataFrame(nlp_dataset['train'])
    if 'wiki' not in dataset:
        if noise_rate > 0:
            noisy_y_train = make_noisy_uniform(noisy_y_train, noise_rate, num_class, noise_type)
        df_train = pd.DataFrame(nlp_dataset['train'])
        df_train['noise_label'] = noisy_y_train
    
    df_train['id'] = [i for i in range(len(nlp_dataset['train']))]

    nlp_dataset['train'] = Dataset.from_pandas(df_train)
    if dataset == 'SetFit/20_newsgroups':
        nlp_dataset = nlp_dataset.remove_columns(['text', 'label_text'])
    elif dataset == 'ag_news' or dataset == 'trec':
        nlp_dataset = nlp_dataset.remove_columns(['text'])
    elif dataset == 'ag_news' or dataset == 'trec':
        nlp_dataset = nlp_dataset.remove_columns(['text'])

    nlp_dataset.set_format("torch")
    return nlp_dataset

def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp

    data_list=[]
    for net_id, data in net_cls_counts.items():
        n_total=0
        for class_id, n_data in data.items():
            n_total += n_data
        data_list.append(n_total)
    print('mean:', np.mean(data_list))
    print('std:', np.std(data_list))

    print('* Dataset statistics:')
    for net_id in net_cls_counts:
        cd = []
        for c in net_cls_counts[net_id]:
            cd.append(net_cls_counts[net_id][c])
        print('- Client {}: {})'.format(net_id, net_cls_counts[net_id]))
    return net_cls_counts

def get_dataloader(dataset, train_bs: int, test_bs: int, dataidxs=None):
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    if dataidxs is None:
        train_dl = DataLoader(train_dataset, batch_size=train_bs, pin_memory=True, shuffle=True)
        test_dl = DataLoader(test_dataset, batch_size=train_bs, pin_memory=True)
        local_dl = DataLoader(train_dataset, batch_size=train_bs, pin_memory=True)
    else:
        train_dl = DataLoader(train_dataset, batch_size=train_bs, sampler=SubsetRandomSampler(dataidxs), pin_memory=True)
        test_dl = DataLoader(test_dataset, batch_size=train_bs, pin_memory=True)
        local_dl = DataLoader(train_dataset, batch_size=train_bs, sampler=SubsetSequentialSampler(dataidxs), pin_memory=True)
    return train_dl, test_dl, local_dl

def get_multi_dataloader(dataset: str, datadir: str, train_bs: int, test_bs: int, dataidxs=None, client_id: int=None, lang=None, n_parties=100):
    langs = ['en', 'de', 'es', 'fr', 'ru']

    if client_id is not None:
        n_clients_per_language = int(n_parties / len(langs))
        if n_clients_per_language == 0:
            n_clients_per_language = 1

        lang_idx = int(client_id/n_clients_per_language)
        current_lang = langs[lang_idx]
        train_dataset, test_dataset = nlp_dataset[current_lang]

    if lang is not None:
        train_dataset, test_dataset = nlp_dataset[lang]

    if n_parties == 1: # Union mode
        all_lang_train_dataset = []
        all_lang_test_dataset = []
        for lang in langs:
            train_dataset, test_dataset = nlp_dataset[lang]
            all_lang_train_dataset.append(train_dataset)
            all_lang_test_dataset.append(test_dataset)
        all_lang_train_dataset = concatenate_datasets(all_lang_train_dataset)
        all_lang_test_dataset = concatenate_datasets(all_lang_test_dataset)
        
        train_dl = DataLoader(all_lang_train_dataset, batch_size=train_bs, pin_memory=True, shuffle=True)
        test_dl = DataLoader(all_lang_test_dataset, batch_size=train_bs, pin_memory=True)
        local_dl = DataLoader(all_lang_train_dataset, batch_size=train_bs, pin_memory=True)
        
        return train_dl, test_dl, local_dl
    else:
        
        if dataidxs is None:
            train_dl = DataLoader(train_dataset, batch_size=train_bs, pin_memory=True, shuffle=True)
            test_dl = DataLoader(test_dataset, batch_size=train_bs, pin_memory=True)
            local_dl = DataLoader(train_dataset, batch_size=train_bs, pin_memory=True)
        else:
            train_dl = DataLoader(train_dataset, batch_size=train_bs, sampler=SubsetRandomSampler(dataidxs), pin_memory=True)
            test_dl = DataLoader(test_dataset, batch_size=train_bs, pin_memory=True)
            local_dl = DataLoader(train_dataset, batch_size=train_bs, sampler=SubsetSequentialSampler(dataidxs), pin_memory=True)
    return train_dl, test_dl, local_dl

def initialize_networks(dataset: str, model: str, device: str ='cpu', adapter: str = '', rank=16, task=16):
    """ Initialize the network based on the given dataset and model specification. """
    #from pabee import BertForSequenceClassificationWithPabee
    n_classes = get_num_classes(dataset)
    
    if dataset == 'ner':
        model_class = AutoModelForTokenClassification
    elif dataset == 'qa':
        model_class = AutoModelForMultipleChoice
    else:
        model_class = AutoModelForSequenceClassification
        
    plm = model_class.from_pretrained(model, num_labels=n_classes)

    if adapter == 'base':
        base_plm = model_class.from_pretrained(model, num_labels=n_classes)
        layers = plm.bert.encoder.layer
        for i in reversed(range(len(layers))):
            layers[i].attention.output = BertAdapterSelfOutput(plm.config)
            layers[i].output = BertAdapterOutput(plm.config)
        plm.load_state_dict(base_plm.state_dict(), strict=False)
    elif adapter == 'mix': #  BertMixAdapterOutput, BertMixAdapterSelfOutput
        base_plm = model_class.from_pretrained(model, num_labels=n_classes)
        layers = plm.bert.encoder.layer
        for i, roberta_layer in enumerate(layers):
            layers[i].attention.output = BertMixAdapterSelfOutput(plm.config)
            layers[i].output = BertMixAdapterOutput(plm.config)
        plm.load_state_dict(base_plm.state_dict(), strict=False)
        plm.classifier = MixClassifier(plm.config, n_classes)
        
    elif adapter == 'routing_adapter': #  BertMixAdapterOutput, BertMixAdapterSelfOutput
        base_plm = model_class.from_pretrained(model, num_labels=n_classes)
        layers = plm.bert.encoder.layer
        for i in reversed(range(len(layers))):
            layers[i].attention.output = BertFusionAdapterSelfOutput(plm.config)
            layers[i].output = BertFusionAdapterOutput(plm.config)
        plm.load_state_dict(base_plm.state_dict(), strict=False)
        plm.classifier = FusionClassifier(plm.config, n_classes)
    
    elif adapter == 'prefix':
        base_plm = model_class.from_pretrained(model, num_labels=n_classes)
        plm.bert.encoder = BertPrefixEncoder(plm.config)
        plm.load_state_dict(base_plm.state_dict(), strict=False)
        
    elif adapter == 'lora':
        rank = 4
        for i in range(len(plm.bert.encoder.layer)):
            # query
            in_dim = plm.bert.encoder.layer[i].attention.self.query.in_features
            out_dim = plm.bert.encoder.layer[i].attention.self.query.out_features
            weight = plm.bert.encoder.layer[i].attention.self.query.weight
            bias = plm.bert.encoder.layer[i].attention.self.query.bias

            plm.bert.encoder.layer[i].attention.self.query = loralib.Linear(in_dim, out_dim, r=rank)
            plm.bert.encoder.layer[i].attention.self.query.weight = weight
            plm.bert.encoder.layer[i].attention.self.query.bias = bias

            # v_lin
            in_dim = plm.bert.encoder.layer[i].attention.self.value.in_features
            out_dim = plm.bert.encoder.layer[i].attention.self.value.out_features
            weight = plm.bert.encoder.layer[i].attention.self.value.weight
            bias = plm.bert.encoder.layer[i].attention.self.value.bias

            plm.bert.encoder.layer[i].attention.self.value = loralib.Linear(in_dim, out_dim, r=rank)
            plm.bert.encoder.layer[i].attention.self.value.weight = weight
            plm.bert.encoder.layer[i].attention.self.value.bias = bias

            # k_lin
            in_dim = plm.bert.encoder.layer[i].attention.self.key.in_features
            out_dim = plm.bert.encoder.layer[i].attention.self.key.out_features
            weight = plm.bert.encoder.layer[i].attention.self.key.weight
            bias = plm.bert.encoder.layer[i].attention.self.key.bias

            plm.bert.encoder.layer[i].attention.self.key = loralib.Linear(in_dim, out_dim, r=rank)
            plm.bert.encoder.layer[i].attention.self.key.weight = weight
            plm.bert.encoder.layer[i].attention.self.key.bias = bias

            # out_lin
            in_dim = plm.bert.encoder.layer[i].attention.output.dense.in_features
            out_dim = plm.bert.encoder.layer[i].attention.output.dense.out_features
            weight = plm.bert.encoder.layer[i].attention.output.dense.weight
            bias = plm.bert.encoder.layer[i].attention.output.dense.bias

            plm.bert.encoder.layer[i].attention.output.dense = loralib.Linear(in_dim, out_dim, r=rank)
            plm.bert.encoder.layer[i].attention.output.dense.weight = weight
            plm.bert.encoder.layer[i].attention.output.dense.bias = bias
            
    elif adapter == 'routing_lora': # BertRoutingLoRAEncoder
        base_plm = model_class.from_pretrained(model, num_labels=n_classes)
        plm.bert.encoder = BertRoutingLoRAEncoder(plm.config)
        plm.load_state_dict(base_plm.state_dict(), strict=False)
        plm.bert.encoder.fixed_layer = copy.deepcopy(plm.bert.encoder.layer)
        
        rank = 4
        for i in range(len(plm.bert.encoder.layer)):
            # query
            in_dim = plm.bert.encoder.layer[i].attention.self.query.in_features
            out_dim = plm.bert.encoder.layer[i].attention.self.query.out_features
            weight = plm.bert.encoder.layer[i].attention.self.query.weight
            bias = plm.bert.encoder.layer[i].attention.self.query.bias

            plm.bert.encoder.layer[i].attention.self.query = loralib.Linear(in_dim, out_dim, r=rank)
            plm.bert.encoder.layer[i].attention.self.query.weight = weight
            plm.bert.encoder.layer[i].attention.self.query.bias = bias

            # v_lin
            in_dim = plm.bert.encoder.layer[i].attention.self.value.in_features
            out_dim = plm.bert.encoder.layer[i].attention.self.value.out_features
            weight = plm.bert.encoder.layer[i].attention.self.value.weight
            bias = plm.bert.encoder.layer[i].attention.self.value.bias

            plm.bert.encoder.layer[i].attention.self.value = loralib.Linear(in_dim, out_dim, r=rank)
            plm.bert.encoder.layer[i].attention.self.value.weight = weight
            plm.bert.encoder.layer[i].attention.self.value.bias = bias

            # # k_lin
            # in_dim = plm.bert.encoder.layer[i].attention.self.key.in_features
            # out_dim = plm.bert.encoder.layer[i].attention.self.key.out_features
            # weight = plm.bert.encoder.layer[i].attention.self.key.weight
            # bias = plm.bert.encoder.layer[i].attention.self.key.bias

            # plm.bert.encoder.layer[i].attention.self.key = loralib.Linear(in_dim, out_dim, r=rank)
            # plm.bert.encoder.layer[i].attention.self.key.weight = weight
            # plm.bert.encoder.layer[i].attention.self.key.bias = bias

            # # out_lin
            # in_dim = plm.bert.encoder.layer[i].attention.output.dense.in_features
            # out_dim = plm.bert.encoder.layer[i].attention.output.dense.out_features
            # weight = plm.bert.encoder.layer[i].attention.output.dense.weight
            # bias = plm.bert.encoder.layer[i].attention.output.dense.bias

            # plm.bert.encoder.layer[i].attention.output.dense = loralib.Linear(in_dim, out_dim, r=rank)
            # plm.bert.encoder.layer[i].attention.output.dense.weight = weight
            # plm.bert.encoder.layer[i].attention.output.dense.bias = bias
    
    elif adapter == 'routing_prefix': #BertRoutingBitFitEncoder
        base_plm = model_class.from_pretrained(model, num_labels=n_classes)
        plm.bert.encoder = BertRoutingPrefixEncoder(plm.config)
        plm.load_state_dict(base_plm.state_dict(), strict=False)
    
    elif adapter == 'routing_bitfit': #BertRoutingBitFitEncoder
        base_plm = model_class.from_pretrained(model, num_labels=n_classes)
        plm.bert.encoder = BertRoutingBitFitEncoder(plm.config)
        plm.load_state_dict(base_plm.state_dict(), strict=False)
        plm.bert.encoder.fixed_layer = copy.deepcopy(plm.bert.encoder.layer)
        
    return plm