'''
Multitask BERT class, starter training code, evaluation, and test code.

Of note are:
* class MultitaskBERT: Your implementation of multitask BERT.
* function train_multitask: Training procedure for MultitaskBERT. Starter code
    copies training procedure from classifier.py (single-task SST).
* function test_multitask: Test procedure for MultitaskBERT. This function generates
    the required files for submission.

Running python multitask_classifier.py trains and tests your MultitaskBERT and
writes all required submission files.
'''

import random, numpy as np, argparse
from types import SimpleNamespace

# import gc
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


from bert import BertModel
from optimizer import AdamW
from tqdm import tqdm
import math
import copy
from datasets import (
    SentenceClassificationDataset,
    SentenceClassificationTestDataset,
    SentencePairDataset,
    SentencePairTestDataset,
    TrainSNLIDataSet,
    load_multitask_data,
    SNLIDataSet
)
from tokenizer import BertTokenizer

from evaluation import model_eval_sst, model_eval_multitask, model_eval_test_multitask, model_eval_paraphrase, \
    model_eval_sts, model_eval_simcse

TQDM_DISABLE=False


# Fix the random seed.
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


BERT_HIDDEN_SIZE = 768
N_SENTIMENT_CLASSES = 5



class MultitaskBERT(nn.Module):
    '''
    This module should use BERT for 3 tasks:

    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    '''
    def __init__(self, config):
        super(MultitaskBERT, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        # last-linear-layer mode does not require updating BERT paramters.
        assert config.fine_tune_mode in ["last-linear-layer", "full-model"]
        for param in self.bert.parameters():
            if config.fine_tune_mode == 'last-linear-layer':
                param.requires_grad = False
            elif config.fine_tune_mode == 'full-model':
                param.requires_grad = True
        # You will want to add layers here to perform the downstream tasks.
        ### TODO
        self.dropout_layer = torch.nn.Dropout(config.hidden_dropout_prob)
        self.sentiment_classifier_layer = torch.nn.Linear(self.bert.config.hidden_size, 5)
        self.paraphrase_classifier_layer = torch.nn.Linear(self.bert.config.hidden_size*3, 1)
        self.similarity_classifier_layer = torch.nn.Linear(self.bert.config.hidden_size*2, 1)
        self.SNLI_classifier_layer = torch.nn.Linear(self.bert.config.hidden_size*3, 3)



        # raise NotImplementedError

    def forward(self, input_ids, attention_mask,noise=0):
        'Takes a batch of sentences and produces embeddings for them.'
        # The final BERT embedding is the hidden state of [CLS] token (the first token)
        # Here, you can start by just returning the embeddings straight from BERT.
        # When thinking of improvements, you can later try modifying this
        # (e.g., by adding other layers).
        ### TODO
        embedding_output = self.bert(input_ids, attention_mask,noise)  # Using forward method in bert.py file
        return embedding_output

    def supervised_simcse_loss(self,
                              input_ids_1, attention_mask_1,
                              input_ids_2, attention_mask_2,
                              input_ids_3, attention_mask_3,pooling='mean',
                              temperature=0.05):
        """
        Compute supervised SimCSE loss for triplet input:
        anchor (input_ids_1), positive (input_ids_2), negative (input_ids_3)
        """
        # Encode inputs
        embedding_output1=self.forward(input_ids_1, attention_mask_1)
        embedding_output2=self.forward(input_ids_2, attention_mask_2)
        embedding_output3=self.forward(input_ids_3, attention_mask_3)
        if pooling=='mean':
            hidden_states1=embedding_output1['last_hidden_state']
            hidden_states2=embedding_output2['last_hidden_state']
            hidden_states3=embedding_output3['last_hidden_state']

            mask1=attention_mask_1.unsqueeze(-1).expand(-1, -1, hidden_states1.shape[2]).float()
            mask2=attention_mask_2.unsqueeze(-1).expand(-1, -1, hidden_states2.shape[2]).float()
            mask3=attention_mask_3.unsqueeze(-1).expand(-1, -1, hidden_states3.shape[2]).float()

            den1=torch.sum(attention_mask_1,dim=1).unsqueeze(-1).expand(-1, hidden_states1.shape[2])+1e-8
            den2=torch.sum(attention_mask_2,dim=1).unsqueeze(-1).expand(-1, hidden_states2.shape[2])+1e-8
            den3=torch.sum(attention_mask_3,dim=1).unsqueeze(-1).expand(-1, hidden_states3.shape[2])+1e-8
            
            vector1 = torch.sum(mask1*(embedding_output1['last_hidden_state']), dim=1)/den1
            vector2 = torch.sum(mask2*(embedding_output2['last_hidden_state']), dim=1)/den2
            vector3 = torch.sum(mask3*(embedding_output3['last_hidden_state']), dim=1)/den3
        elif pooling=='cls':
            vector1=embedding_output1['pooler_output']
            vector2=embedding_output2['pooler_output']
            vector3=embedding_output3['pooler_output']
        anchor = self.dropout_layer(vector1)  # a
        positive = self.dropout_layer(vector2)  # b
        negative = self.dropout_layer(vector3)  # c


        # Normalize embeddings for cosine similarity
        anchor = F.normalize(anchor, dim=-1)
        positive = F.normalize(positive, dim=-1)
        negative = F.normalize(negative, dim=-1)

        # Compute cosine similarities
        sim_ap = torch.matmul(anchor, positive.T) / temperature  # anchor-positive
        sim_an = torch.matmul(anchor, negative.T) / temperature  # anchor-negative

        # Combine positive and negative similarities
        exp_ap = torch.exp(sim_ap.diag())  # correct pairs
        exp_all = torch.exp(sim_ap) + torch.exp(sim_an)  # all (positive + negatives)

        # Sum over rows and compute loss
        denom = exp_all.sum(dim=1)
        loss = -1*torch.log(exp_ap / denom)
        return loss.mean()


    def unsupervised_simcse_loss(self,
                              input_ids_1, attention_mask_1,pooling='mean',
                              temperature=0.05):
        """
        Compute supervised SimCSE loss for triplet input:
        anchor (input_ids_1), positive (input_ids_2), negative (input_ids_3)
        """
        # Encode inputs
        embedding_output1=self.forward(input_ids_1, attention_mask_1)
        embedding_output2=self.forward(input_ids_1, attention_mask_1)
        if pooling=='mean':
            hidden_states1=embedding_output1['last_hidden_state']
            hidden_states2=embedding_output2['last_hidden_state']

            mask1=attention_mask_1.unsqueeze(-1).expand(-1, -1, hidden_states1.shape[2]).float()

            den1=torch.sum(attention_mask_1,dim=1).unsqueeze(-1).expand(-1, hidden_states1.shape[2])+1e-8

            vector1 = torch.sum(mask1*(embedding_output1['last_hidden_state']), dim=1)/den1
            vector2 = torch.sum(mask1*(embedding_output2['last_hidden_state']), dim=1)/den1
            # vector3 = torch.sum(mask3*(embedding_output3['last_hidden_state']), dim=1)/den3
        elif pooling=='cls':
            vector1=embedding_output1['pooler_output']

        h = self.dropout_layer(vector1)
        h_plus = self.dropout_layer(vector2)

        # Normalize embeddings for cosine similarity
        h = F.normalize(h, dim=-1)
        h_plus = F.normalize(h_plus, dim=-1)

        # Compute cosine similarities
        sim_ap = torch.matmul(h, h_plus.T) / temperature  # anchor-positive

        exp_ap = torch.exp(sim_ap.diag())  # correct pairs
        exp_all = torch.exp(sim_ap)

        # Sum over rows and compute loss
        denom = exp_all.sum(dim=1)
        loss = -1*torch.log(exp_ap / denom)
        return loss.mean()

        
    def predict_SNLI(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2,noise1=0,noise2=0,pooling='mean'):
        '''Given a batch of pairs of sentences, outputs a single logit for predicting whether they are paraphrases.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation.
        '''
        ### TODO
        embedding_output1=self.forward(input_ids_1, attention_mask_1,noise1)
        embedding_output2=self.forward(input_ids_2, attention_mask_2,noise2)
        if pooling=='mean':
            hidden_states1=embedding_output1['last_hidden_state']
            hidden_states2=embedding_output2['last_hidden_state']
            mask1=attention_mask_1.unsqueeze(-1).expand(-1, -1, hidden_states1.shape[2]).float()
            mask2=attention_mask_2.unsqueeze(-1).expand(-1, -1, hidden_states1.shape[2]).float()
            den1=torch.sum(attention_mask_1,dim=1).unsqueeze(-1).expand(-1, hidden_states1.shape[2])+1e-8
            den2=torch.sum(attention_mask_2,dim=1).unsqueeze(-1).expand(-1, hidden_states1.shape[2])+1e-8
            vector1 = torch.sum(mask1*(embedding_output1['last_hidden_state']), dim=1)/den1
            vector2 = torch.sum(mask2*(embedding_output2['last_hidden_state']), dim=1)/den2
        elif pooling=='cls':
            vector1=embedding_output1['pooler_output']
            vector2=embedding_output2['pooler_output']

        Concatenated_result=torch.concat((vector1,vector2,abs(vector1-vector2)),dim=1)
        logits=self.SNLI_classifier_layer(Concatenated_result)
        return logits


    def predict_sentiment(self, input_ids, attention_mask,noise=0,pooling='mean'):
        '''Given a batch of sentences, outputs logits for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1- somewhat negative, 2- neutral, 3- somewhat positive, 4- positive)
        Thus, your output should contain 5 logits for each sentence.
        '''
        ### TODO
        embedding_output=self.forward(input_ids, attention_mask,noise=noise)
        if pooling=='mean':
            hidden_state=embedding_output['last_hidden_state']
            mask=attention_mask.unsqueeze(-1).expand(-1, -1, hidden_state.shape[2]).float()
            den=torch.sum(attention_mask,dim=1).unsqueeze(-1).expand(-1, hidden_state.shape[2])+1e-8
            vector = torch.sum(mask*(embedding_output['last_hidden_state']), dim=1)/den
        elif pooling=='cls':
            vector=embedding_output['pooler_output']
        Dropout_layer_output=self.dropout_layer(vector)
        logits=self.sentiment_classifier_layer(Dropout_layer_output)
        if isinstance(noise, torch.Tensor):
            if pooling=='mean':
                vector_wn = torch.sum(mask*(embedding_output['last_hidden_state_wn']), dim=1)/den
            elif pooling=='cls':
                vector_wn=embedding_output['pooler_output_wn']
            Dropout_layer_output_wn=self.dropout_layer(vector_wn)
            logits_wn=self.sentiment_classifier_layer(Dropout_layer_output_wn)
            Embedding_in=embedding_output['embedding_output']
            Embedding_in_wn=embedding_output['embedding_output_with_noise']
            return logits,logits_wn,Embedding_in,Embedding_in_wn
        return logits

    def predict_paraphrase(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2,noise1=0,noise2=0,pooling='mean'):
        '''Given a batch of pairs of sentences, outputs a single logit for predicting whether they are paraphrases.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation.
        '''
        ### TODO
        embedding_output1=self.forward(input_ids_1, attention_mask_1,noise1)
        embedding_output2=self.forward(input_ids_2, attention_mask_2,noise2)
        if pooling=='mean':
            hidden_states1=embedding_output1['last_hidden_state']
            hidden_states2=embedding_output2['last_hidden_state']
            mask1=attention_mask_1.unsqueeze(-1).expand(-1, -1, hidden_states1.shape[2]).float()
            mask2=attention_mask_2.unsqueeze(-1).expand(-1, -1, hidden_states1.shape[2]).float()
            den1=torch.sum(attention_mask_1,dim=1).unsqueeze(-1).expand(-1, hidden_states1.shape[2])+1e-8
            den2=torch.sum(attention_mask_2,dim=1).unsqueeze(-1).expand(-1, hidden_states1.shape[2])+1e-8
            vector1 = torch.sum(mask1*(embedding_output1['last_hidden_state']), dim=1)/den1
            vector2 = torch.sum(mask2*(embedding_output2['last_hidden_state']), dim=1)/den2
        elif pooling=='cls':
            vector1=embedding_output1['pooler_output']
            vector2=embedding_output2['pooler_output']
        Dropout_layer_output1=self.dropout_layer(vector1)
        Dropout_layer_output2=self.dropout_layer(vector2)

        Concatenated_result=torch.concat((Dropout_layer_output1,Dropout_layer_output2,abs(Dropout_layer_output1-Dropout_layer_output2)),dim=1)
        logit=self.paraphrase_classifier_layer(Concatenated_result)
        if isinstance(noise1, torch.Tensor) and isinstance(noise2, torch.Tensor):
            if pooling=='mean':
                vector1_wn = torch.sum(mask1*(embedding_output1['last_hidden_state_wn']), dim=1)/den1
                vector2_wn = torch.sum(mask2*(embedding_output2['last_hidden_state_wn']), dim=1)/den2
            elif pooling=='cls':
                vector1_wn=embedding_output1['pooler_output_wn']
                vector2_wn=embedding_output2['pooler_output_wn']        
            Dropout_layer_output1_wn=self.dropout_layer(vector1_wn)
            Dropout_layer_output2_wn=self.dropout_layer(vector2_wn)
            Concatenated_result_wn=torch.concat((Dropout_layer_output1_wn,Dropout_layer_output2_wn,abs(Dropout_layer_output1_wn-Dropout_layer_output2_wn)),dim=1)
            logit_wn=self.paraphrase_classifier_layer(Concatenated_result_wn)
            Embedding_in1=embedding_output1['embedding_output']
            Embedding_in2=embedding_output2['embedding_output']
            Embedding_in1_wn=embedding_output1['embedding_output_with_noise']
            Embedding_in2_wn=embedding_output2['embedding_output_with_noise']
            return logit,logit_wn,Embedding_in1,Embedding_in2,Embedding_in1_wn,Embedding_in2_wn
        return logit

    def predict_similarity(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2,noise1=0,noise2=0,pooling='mean'):
        '''Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
        Note that your output should be unnormalized (a logit).
        '''
        ### TODO
        embedding_output1=self.forward(input_ids_1, attention_mask_1,noise1)
        embedding_output2=self.forward(input_ids_2, attention_mask_2,noise2)
        if pooling == 'mean':
            hidden_states1=embedding_output1['last_hidden_state']
            hidden_states2=embedding_output2['last_hidden_state']
            mask1=attention_mask_1.unsqueeze(-1).expand(-1, -1, hidden_states1.shape[2]).float()
            mask2=attention_mask_2.unsqueeze(-1).expand(-1, -1, hidden_states1.shape[2]).float()
            den1=torch.sum(attention_mask_1,dim=1).unsqueeze(-1).expand(-1, hidden_states1.shape[2])+1e-8
            den2=torch.sum(attention_mask_2,dim=1).unsqueeze(-1).expand(-1, hidden_states1.shape[2])+1e-8
            vector1 = torch.sum(mask1*(embedding_output1['last_hidden_state']), dim=1)/den1
            vector2 = torch.sum(mask2*(embedding_output2['last_hidden_state']), dim=1)/den2
        elif pooling=='cls':
            vector1=embedding_output1['pooler_output']
            vector2=embedding_output2['pooler_output']
        Dropout_layer_output1=self.dropout_layer(vector1)
        Dropout_layer_output2=self.dropout_layer(vector2)
        Cosine_similarity=(torch.sum(Dropout_layer_output1*Dropout_layer_output2,dim=1)/((torch.norm(Dropout_layer_output1,dim=1)*torch.norm(Dropout_layer_output2,dim=1))+1e-8)+1)*2.5
        if isinstance(noise1, torch.Tensor) and isinstance(noise2, torch.Tensor):
            if pooling=='mean':
                vector1_wn = torch.sum(mask1*(embedding_output1['last_hidden_state_wn']), dim=1)/den1
                vector2_wn = torch.sum(mask2*(embedding_output2['last_hidden_state_wn']), dim=1)/den2
            elif pooling=='cls':
                vector1_wn=embedding_output1['pooler_output_wn']
                vector2_wn=embedding_output2['pooler_output_wn']        
            Dropout_layer_output1_wn=self.dropout_layer(vector1_wn)
            Dropout_layer_output2_wn=self.dropout_layer(vector2_wn)
            Cosine_similarity_wn=(torch.sum(Dropout_layer_output1_wn*Dropout_layer_output2_wn,dim=1)/((torch.norm(Dropout_layer_output1_wn,dim=1)*torch.norm(Dropout_layer_output2_wn,dim=1))+1e-8)+1)*2.5
            Embedding_in1=embedding_output1['embedding_output']
            Embedding_in2=embedding_output2['embedding_output'] 
            Embedding_in1_wn=embedding_output1['embedding_output_with_noise']
            Embedding_in2_wn=embedding_output2['embedding_output_with_noise']
            return Cosine_similarity,Cosine_similarity_wn,Embedding_in1,Embedding_in2,Embedding_in1_wn,Embedding_in2_wn
        return Cosine_similarity

def save_model(model, optimizer, args, config, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")


def train_multitask(args):
    '''Train MultitaskBERT.

    Currently only trains on SST dataset. The way you incorporate training examples
    from other datasets into the training procedure is up to you. To begin, take a
    look at test_multitask below to see how you can use the custom torch `Dataset`s
    in datasets.py to load in examples from the Quora and SemEval datasets.
    '''
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    # Create the data and its corresponding datasets and dataloader.
    sst_train_data, num_labels,para_train_data, sts_train_data, snli_train_data = load_multitask_data(args.sst_train,args.para_train,args.sts_train, args.snli_train, split ='train')
    sst_dev_data, num_labels,para_dev_data, sts_dev_data, snli_dev_data = load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev, args.snli_dev, split ='dev')

    sst_train_data = SentenceClassificationDataset(sst_train_data, args)
    sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)
    
    sst_train_dataloader = DataLoader(sst_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sst_train_data.collate_fn)
    sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sst_dev_data.collate_fn)
    
    para_train_data = SentencePairDataset(para_train_data, args)
    para_dev_data = SentencePairDataset(para_dev_data, args)
    
    para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=para_train_data.collate_fn)
    para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=para_dev_data.collate_fn)
    
    sts_train_data = SentencePairDataset(sts_train_data, args, isRegression=True)
    sts_dev_data = SentencePairDataset(sts_dev_data, args, isRegression=True)
    
    sts_train_dataloader = DataLoader(sts_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sts_train_data.collate_fn)
    sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sts_dev_data.collate_fn)

    snli_train_data = TrainSNLIDataSet(snli_train_data, args)
    snli_dev_data = SNLIDataSet(snli_dev_data, args)

    snli_train_dataloader = DataLoader(snli_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=snli_train_data.collate_fn)
    snli_dev_dataloader = DataLoader(snli_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=snli_dev_data.collate_fn)

    # Init model.
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'num_labels': num_labels,
              'hidden_size': 768,
              'data_dir': '.',
              'fine_tune_mode': args.fine_tune_mode}

    config = SimpleNamespace(**config)

    model = MultitaskBERT(config)
    model = model.to(device)

    # model = MultitaskBERT(config)
    # checkpoint = torch.load('simcse_Para1.pt', map_location=device, weights_only=False)
    # model.load_state_dict(checkpoint["model"])
    # model = model.to(device)
    best_simcse_dev_acc=0.644
    best_sst_dev_acc = 0
    best_paraphrase_dev_acc = 0
    best_sts_corr = 0

    lr = args.lr
    optimizer = AdamW(model.parameters(), lr=lr)

    for epoch in range(args.epochs):
        model.train()
        # Train SimCSE using SNLI dataset
        simcse_train_loss = 0
        num_batches = 0

        for batch in tqdm(snli_train_dataloader, desc=f'SimCSE-{epoch}', disable=TQDM_DISABLE):
            b_ids1, b_mask1, b_ids2, b_mask2, b_ids3, b_mask3 = (
                batch['token_ids_1'], batch['attention_mask_1'],
                batch['token_ids_2'], batch['attention_mask_2'],
                batch['token_ids_3'], batch['attention_mask_3']
            )

            b_ids1, b_mask1 = b_ids1.to(device), b_mask1.to(device)
            b_ids2, b_mask2 = b_ids2.to(device), b_mask2.to(device)
            b_ids3, b_mask3 = b_ids3.to(device), b_mask3.to(device)

            optimizer.zero_grad()

            # Compute contrastive loss
            loss = model.supervised_simcse_loss(b_ids1, b_mask1,b_ids2, b_mask2,b_ids3, b_mask3)

            loss.backward()
            optimizer.step()

            simcse_train_loss += loss.item()
            num_batches += 1

        simcse_train_loss /= num_batches
        simcse_dev_acc = model_eval_simcse(snli_dev_dataloader, model, device)

        dev_sts_corr= model_eval_sts(sts_dev_dataloader, model, device)
        if dev_sts_corr > best_sts_corr:
            best_sts_corr = dev_sts_corr
            save_model(model, optimizer, args, config, args.filepath)
        print(f"Epoch {epoch}: dev_sts_corr :: {dev_sts_corr :.3f}")

    model = MultitaskBERT(config)
    checkpoint = torch.load(args.filepath, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model = model.to(device)

    paraphrase_dev_acc,paraphrase_dev_f1 = model_eval_paraphrase(para_dev_dataloader, model, device)
    dev_sts_corr= model_eval_sts(sts_dev_dataloader, model, device)
    sst_dev_acc, sst_dev_f1, *_ = model_eval_sst(sst_dev_dataloader, model, device)

    print('dev sts corr = ',dev_sts_corr)
    print('dev sst acc = ',sst_dev_acc)
    print('dev sst f1 = ',sst_dev_f1)
    print('dev paraphrase acc = ',paraphrase_dev_acc)
    print('dev paraphrase f1 = ',paraphrase_dev_f1)

    overall_perf_acc=((dev_sts_corr+1)/2+sst_dev_acc+paraphrase_dev_acc)/3
    print('Overall Performance = ',overall_perf_acc)
    best_overall_perf=overall_perf_acc
    lr = args.lr
    optimizer = AdamW(model.parameters(), lr=lr)
    epsilon=1e-5
    eps = 1e-8
    lambda_s=0
    std=1e-5
    mean=0
    S=1
    Tx=1
    eta=1e-3
    mu=0
    model_copy = copy.deepcopy(model)
    # Run for the specified number of epochs.
    for epoch in range(1):
        model.train()
        train_loss = 0
        num_batches = 0
        theta_t=copy.deepcopy(model.state_dict())
        theta_tilde_t=copy.deepcopy(theta_t)
        Beta=0.99
        n=0
        for batch in tqdm(para_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
            b_ids1, b_mask1,b_ids2, b_mask2,b_labels= (batch['token_ids_1'], batch['attention_mask_1'],batch['token_ids_2'], batch['attention_mask_2'],batch['labels'])


            b_ids1 = b_ids1.to(device)
            b_mask1 = b_mask1.to(device)
            b_ids2 = b_ids2.to(device)
            b_mask2 = b_mask2.to(device)
            b_labels=b_labels.to(device)

            if lambda_s!=0 or mu!=0:
                theta_tilde_s=copy.deepcopy(theta_t)
                if n==3538:
                    Beta=0.999
                n+=1
                model_copy.load_state_dict(theta_tilde_t)
                for s in range(0,S):
                    noise1 = torch.randn(b_ids1.shape[0], b_ids1.shape[1], 768)*std+mean
                    noise1=noise1.to(device)
                    noise2 = torch.randn(b_ids2.shape[0], b_ids2.shape[1], 768)*std+mean
                    noise2=noise2.to(device)
                    for m in range(0,Tx):
                        logit,logit_wn,_,_,Embedding_in1_wn,Embedding_in2_wn = model.predict_paraphrase(b_ids1, b_mask1, b_ids2, b_mask2, noise1, noise2)
                        prob = torch.sigmoid(logit)    # shape: (batch_size, 1)
                        prob_wn = torch.sigmoid(logit_wn)  # with noise
                        P1 = torch.cat([1 - prob, prob], dim=1)  # [P(not paraphrase), P(paraphrase)]
                        Q = torch.cat([1 - prob_wn, prob_wn], dim=1)
                        KL_divergence_PQ = P1 * torch.log((P1 + eps) / (Q + eps))
                        KL_divergence_QP = Q * torch.log((Q + eps) / (P1 + eps))
                        ls=KL_divergence_PQ+KL_divergence_QP
                        ls=ls.sum()
                        ls.backward() # retain_graph=True to allow backward being called again
                        gi1_tilde=Embedding_in1_wn.grad/Embedding_in1_wn.shape[0]
                        gi2_tilde=Embedding_in2_wn.grad/Embedding_in2_wn.shape[0]
                        gi1_tilde/=(torch.norm(gi1_tilde,p=float('inf'),dim=(1, 2)).view(gi1_tilde.shape[0],1,1)+eps)
                        gi2_tilde/=(torch.norm(gi2_tilde,p=float('inf'),dim=(1, 2)).view(gi2_tilde.shape[0],1,1)+eps)
                        noise1 += eta * gi1_tilde
                        noise2 += eta * gi2_tilde
                    logit,logit_wn,*rest = model.predict_paraphrase(b_ids1, b_mask1, b_ids2, b_mask2, noise1, noise2) 
                    prob = torch.sigmoid(logit)    # shape: (batch_size, 1)
                    prob_wn = torch.sigmoid(logit_wn)  # with noise
                    P1 = torch.cat([1 - prob, prob], dim=1)  # [P(not paraphrase), P(paraphrase)]
                    Q = torch.cat([1 - prob_wn, prob_wn], dim=1)

                    KL_divergence_PQ = P1 * torch.log((P1 + eps) / (Q + eps))
                    KL_divergence_QP = Q * torch.log((Q + eps) / (P1 + eps))
                    ls=KL_divergence_PQ+KL_divergence_QP
                    max_ls,_=torch.max(ls,-1)
                    ls=ls.sum()
                    Rs=torch.sum(max_ls)/max_ls.shape[0]
                    logit2,*rest = model_copy.predict_paraphrase(b_ids1, b_mask1, b_ids2, b_mask2, noise1, noise2)
                    prob2 = torch.sigmoid(logit2)    # shape: (batch_size, 1)
                    P2 = torch.cat([1 - prob2, prob2], dim=1)  # [P(not paraphrase), P(paraphrase)]
                    KL_divergence_P1P2 = P1 * torch.log((P1 + eps) / (P2 + eps))
                    KL_divergence_P2P1 = P2 * torch.log((P2 + eps) / (P1 + eps))
                    ls2=KL_divergence_P1P2+KL_divergence_P2P1
                    D_Breg=ls2.sum()/args.batch_size
                    optimizer.zero_grad()
                    loss=F.binary_cross_entropy_with_logits(logit.squeeze(), b_labels.view(-1).float(), reduction='sum') / args.batch_size + lambda_s*Rs + mu*D_Breg
                    loss.backward()
                    optimizer.step()
                    theta_tilde_s=copy.deepcopy(model.state_dict())
                theta_t=copy.deepcopy(theta_tilde_s)
                theta_t_tilde = copy.deepcopy(theta_t)
                for key in theta_tilde_s.keys():
                    theta_t_tilde[key] = (1 - Beta) * theta_tilde_s[key] + Beta * theta_t_tilde[key]
            else:
                optimizer.zero_grad()
                logit = model.predict_paraphrase(b_ids1, b_mask1, b_ids2, b_mask2)
                loss = F.binary_cross_entropy_with_logits(logit.squeeze(), b_labels.view(-1).float(), reduction='sum') / args.batch_size
                loss.backward()
                optimizer.step()
            train_loss += loss.item()
            num_batches += 1
        train_loss = train_loss / (num_batches)

        save_model(model, optimizer, args, config, args.filepath)

        paraphrase_train_acc,paraphrase_train_f1 = model_eval_paraphrase(para_train_dataloader, model, device)
        paraphrase_dev_acc,paraphrase_dev_f1 = model_eval_paraphrase(para_dev_dataloader, model, device)

        if paraphrase_dev_acc > best_paraphrase_dev_acc:
            best_paraphrase_dev_acc = paraphrase_dev_acc
            save_model(model, optimizer, args, config, args.filepath)

        print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, paraphrase train acc :: {paraphrase_train_acc :.3f}, paraphrase dev acc :: {paraphrase_dev_acc :.3f}")

    model = MultitaskBERT(config)
    checkpoint = torch.load(args.filepath, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model = model.to(device)

    paraphrase_dev_acc,paraphrase_dev_f1 = model_eval_paraphrase(para_dev_dataloader, model, device)
    dev_sts_corr= model_eval_sts(sts_dev_dataloader, model, device)
    sst_dev_acc, sst_dev_f1, *_ = model_eval_sst(sst_dev_dataloader, model, device)

    print('dev sts corr = ',dev_sts_corr)
    print('dev sst acc = ',sst_dev_acc)
    print('dev sst f1 = ',sst_dev_f1)
    print('dev paraphrase acc = ',paraphrase_dev_acc)
    print('dev paraphrase f1 = ',paraphrase_dev_f1)

    overall_perf=((dev_sts_corr+1)/2+sst_dev_acc+paraphrase_dev_acc)/3
    print('Overall Performance = ',overall_perf)
    best_overall_perf=overall_perf

    for i in range(10):
        lr = args.lr
        optimizer = AdamW(model.parameters(), lr=lr)

        epsilon=1e-5
        eps = 1e-8
        lambda_s=0
        std=1e-5
        mean=0
        S=1
        Tx=1
        eta=1e-3
        mu=0
        model_copy = copy.deepcopy(model)
        # Run for the specified number of epochs.
        for epoch in range(1):
            model.train()
            train_loss = 0
            num_batches = 0
            theta_t=copy.deepcopy(model.state_dict())
            theta_tilde_t=copy.deepcopy(theta_t)
            Beta=0.99
            n=0
            for batch in tqdm(sts_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
                b_ids1, b_mask1,b_ids2, b_mask2, b_labels = (batch['token_ids_1'], batch['attention_mask_1'],batch['token_ids_2'],batch['attention_mask_2'],batch['labels'])

                b_ids1 = b_ids1.to(device)
                b_mask1 = b_mask1.to(device)
                b_ids2 = b_ids2.to(device)
                b_mask2 = b_mask2.to(device)
                b_labels=b_labels.to(device)
                if lambda_s!=0 or mu!=0:
                    theta_tilde_s=copy.deepcopy(theta_t)
                    # optimizer.zero_grad()
                    if n==76:
                        Beta=0.999
                    n+=1
                    model_copy.load_state_dict(theta_tilde_t)
                    for s in range(0,S):
                        # model.load_state_dict(copy.deepcopy(theta_tilde_s))
                        noise1 = torch.randn(b_ids1.shape[0], b_ids1.shape[1], 768)*std+mean
                        noise1=noise1.to(device)
                        noise2 = torch.randn(b_ids2.shape[0], b_ids2.shape[1], 768)*std+mean
                        noise2=noise2.to(device)
                        for m in range(0,Tx):
                            logit0,logit0_wn,Embedding_in1,Embedding_in2,Embedding_in1_wn,Embedding_in2_wn = model.predict_similarity(b_ids1, b_mask1, b_ids2, b_mask2, noise1, noise2)
                            ls=(logit0_wn-logit0)**2
                            max_ls=ls
                            ls=ls.sum()
                            Rs=torch.sum(max_ls)/max_ls.shape[0]
                            Embedding_in1_wn.retain_grad()
                            Embedding_in2_wn.retain_grad()
                            ls.backward(retain_graph=True) # retain_graph=True to allow backward being called again
                            gi1=Embedding_in1_wn.grad/Embedding_in1_wn.shape[0]
                            gi2=Embedding_in2_wn.grad/Embedding_in2_wn.shape[0]
                            gi1_tilde=gi1/(torch.norm(gi1,p=float('inf'),dim=(1, 2)).view(gi1.shape[0],1,1)+eps)
                            gi2_tilde=gi2/(torch.norm(gi2,p=float('inf'),dim=(1, 2)).view(gi2.shape[0],1,1)+eps)
                        

                            noise1 = noise1.clone() + eta * gi1_tilde.clone()
                            noise2 = noise2.clone() + eta * gi2_tilde.clone()
                        logit,logit_wn,Embedding_in1_1,Embedding_in2_1,Embedding_in1_wn_1,Embedding_in2_wn_1 = model.predict_similarity(b_ids1, b_mask1, b_ids2, b_mask2, noise1, noise2) 
                        ls=(logit-logit_wn)**2
                        max_ls=ls
                        ls=ls.sum()
                        Rs=torch.sum(max_ls)/max_ls.shape[0]
                        logit2,logit2_wn,Embedding_in1_2,Embedding_in2_2,Embedding_in1_wn_2,Embedding_in2_wn_2 = model_copy.predict_similarity(b_ids1, b_mask1, b_ids2, b_mask2, noise1, noise2)
                        ls2=(logit2-logit)**2
                        D_Breg=ls2.sum()/args.batch_size
                        optimizer.zero_grad()
                        loss=F.mse_loss(logit, b_labels.view(-1).float(), reduction='sum') / args.batch_size + lambda_s*Rs +mu*D_Breg
                        loss.backward()
                        optimizer.step()
                        theta_tilde_s=copy.deepcopy(model.state_dict())
                    theta_t=copy.deepcopy(theta_tilde_s)
                    theta_t_tilde = copy.deepcopy(theta_t)
                    for key in theta_tilde_s.keys():
                        theta_t_tilde[key] = (1 - Beta) * theta_tilde_s[key] + Beta * theta_t_tilde[key]
                else:
                    optimizer.zero_grad()
                    logit = model.predict_similarity(b_ids1, b_mask1, b_ids2, b_mask2)
                    loss = F.mse_loss(logit, b_labels.view(-1).float(), reduction='sum') / args.batch_size
                    loss.backward()
                    optimizer.step()
        paraphrase_dev_acc,paraphrase_dev_f1 = model_eval_paraphrase(para_dev_dataloader, model, device)
        dev_sts_corr= model_eval_sts(sts_dev_dataloader, model, device)
        sst_dev_acc, sst_dev_f1, *_ = model_eval_sst(sst_dev_dataloader, model, device)

        print('dev sts corr = ',dev_sts_corr)
        print('dev sst acc = ',sst_dev_acc)
        print('dev sst f1 = ',sst_dev_f1)
        print('dev paraphrase acc = ',paraphrase_dev_acc)
        print('dev paraphrase f1 = ',paraphrase_dev_f1)

        overall_perf=((dev_sts_corr+1)/2+sst_dev_acc+paraphrase_dev_acc)/3
        print('Overall Performance = ',overall_perf)

        if overall_perf > best_overall_perf:
            best_overall_perf = overall_perf
            save_model(model, optimizer, args, config, args.filepath)
            print('New best model saved with overall performance =', overall_perf)



        lr = args.lr
        optimizer = AdamW(model.parameters(), lr=lr)

        epsilon=1e-5
        eps = 1e-8
        lambda_s=0
        std=1e-5
        mean=0
        S=1
        Tx=1
        eta=1e-3
        mu=0
        model_copy = copy.deepcopy(model)
        # Run for the specified number of epochs.
        for epoch in range(1):
            model.train()
            train_loss = 0
            num_batches = 0
            theta_t=copy.deepcopy(model.state_dict())
            theta_tilde_t=copy.deepcopy(theta_t)
            Beta=0.99
            n=0
            for batch in tqdm(sst_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
                b_ids, b_mask, b_labels = (batch['token_ids'],
                                        batch['attention_mask'], batch['labels'])

                b_ids = b_ids.to(device)
                b_mask = b_mask.to(device)
                b_labels = b_labels.to(device)
                if lambda_s!=0 or mu!=0:
                    theta_tilde_s=copy.deepcopy(theta_t)
                    if n==107:
                        Beta=0.999
                    n+=1
                    model_copy.load_state_dict(theta_tilde_t)
                    for s in range(0,S):
                        noise = torch.randn(b_ids.shape[0], b_ids.shape[1], 768)*std+mean
                        noise=noise.to(device)
                        for m in range(0,Tx):
                            logits,logits_wn,Embedding_in,Embedding_in_wn = model.predict_sentiment(b_ids,b_mask,noise)
                            P1=F.softmax(logits, dim=1)
                            Q=F.softmax(logits_wn, dim=1)
                            KL_divergence_PQ = P1 * torch.log((P1 + eps) / (Q + eps))
                            KL_divergence_QP = Q * torch.log((Q + eps) / (P1 + eps))
                            ls=KL_divergence_PQ+KL_divergence_QP
                            max_ls,_=torch.max(ls,-1)
                            ls=ls.sum()
                            Rs=torch.sum(max_ls)/max_ls.shape[0]
                            Embedding_in_wn.retain_grad()
                            ls.backward(retain_graph=True) # retain_graph=True to allow backward being called again
                            gi=Embedding_in_wn.grad/Embedding_in_wn.shape[0]
                            gi_tilde=gi/(torch.norm(gi,p=float('inf'),dim=(1, 2)).view(gi.shape[0],1,1)+eps)
                            noise = noise.clone() + eta * gi_tilde.clone()
                        logits,logits_wn,Embedding_in,Embedding_in_wn = model.predict_sentiment(b_ids,b_mask,noise)
                        P1=F.softmax(logits, dim=1)
                        Q=F.softmax(logits_wn, dim=1)
                        KL_divergence_PQ = P1 * torch.log((P1 + eps) / (Q + eps))
                        KL_divergence_QP = Q * torch.log((Q + eps) / (P1 + eps))
                        ls=KL_divergence_PQ+KL_divergence_QP
                        max_ls,_=torch.max(ls,-1)
                        ls=ls.sum()
                        Rs=torch.sum(max_ls)/max_ls.shape[0]
                        logits2,logits2_wn,Embedding_in_2,Embedding_in_wn_2 = model_copy.predict_sentiment(b_ids,b_mask,noise)
                        P2=F.softmax(logits, dim=1)
                        KL_divergence_P1P2 = P1 * torch.log((P1 + eps) / (P2 + eps))
                        KL_divergence_P2P1 = P2 * torch.log((P2 + eps) / (P1 + eps))
                        ls2=KL_divergence_P1P2+KL_divergence_P2P1
                        D_Breg=ls2.sum()/args.batch_size
                        optimizer.zero_grad()
                        loss=F.cross_entropy(logits, b_labels.view(-1), reduction='sum') / args.batch_size + lambda_s*Rs +mu*D_Breg
                        loss.backward()
                        optimizer.step()
                        theta_tilde_s=copy.deepcopy(model.state_dict())
                    theta_t=copy.deepcopy(theta_tilde_s)
                    theta_t_tilde = copy.deepcopy(theta_t)
                    for key in theta_tilde_s.keys():
                        theta_t_tilde[key] = (1 - Beta) * theta_tilde_s[key] + Beta * theta_t_tilde[key]
                else:
                    optimizer.zero_grad()
                    logits = model.predict_sentiment(b_ids, b_mask)
                    loss = F.cross_entropy(logits, b_labels.view(-1), reduction='sum') / args.batch_size
                    loss.backward()
                    optimizer.step()
        paraphrase_dev_acc,paraphrase_dev_f1 = model_eval_paraphrase(para_dev_dataloader, model, device)
        dev_sts_corr= model_eval_sts(sts_dev_dataloader, model, device)
        sst_dev_acc, sst_dev_f1, *_ = model_eval_sst(sst_dev_dataloader, model, device)

        print('dev sts corr = ',dev_sts_corr)
        print('dev sst acc = ',sst_dev_acc)
        print('dev sst f1 = ',sst_dev_f1)
        print('dev paraphrase acc = ',paraphrase_dev_acc)
        print('dev paraphrase f1 = ',paraphrase_dev_f1)

        overall_perf=((dev_sts_corr+1)/2+sst_dev_acc+paraphrase_dev_acc)/3
        print('Overall Performance = ',overall_perf)

        if overall_perf > best_overall_perf:
            best_overall_perf = overall_perf
            save_model(model, optimizer, args, config, args.filepath)
            print('New best model saved with overall performance =', overall_perf)


        lr = args.lr
        optimizer = AdamW(model.parameters(), lr=lr)
        epsilon=1e-5
        eps = 1e-8
        lambda_s=0
        std=1e-5
        mean=0
        S=1
        Tx=1
        eta=1e-3
        mu=0
        model_copy = copy.deepcopy(model)
        # Run for the specified number of epochs.
        for epoch in range(1):
            model.train()
            train_loss = 0
            num_batches = 0
            theta_t=copy.deepcopy(model.state_dict())
            theta_tilde_t=copy.deepcopy(theta_t)
            Beta=0.99
            n=0
            para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size, collate_fn=para_train_data.collate_fn)
            for batch in tqdm(para_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
                b_ids1, b_mask1,b_ids2, b_mask2,b_labels= (batch['token_ids_1'], batch['attention_mask_1'],batch['token_ids_2'], batch['attention_mask_2'],batch['labels'])

                if n==1001:
                    break
                b_ids1 = b_ids1.to(device)
                b_mask1 = b_mask1.to(device)
                b_ids2 = b_ids2.to(device)
                b_mask2 = b_mask2.to(device)
                b_labels=b_labels.to(device)

                if lambda_s!=0 or mu!=0:
                    theta_tilde_s=copy.deepcopy(theta_t)
                    if n==100:
                        Beta=0.999
                    n+=1
                    model_copy.load_state_dict(theta_tilde_t)
                    for s in range(0,S):
                        noise1 = torch.randn(b_ids1.shape[0], b_ids1.shape[1], 768)*std+mean
                        noise1=noise1.to(device)
                        noise2 = torch.randn(b_ids2.shape[0], b_ids2.shape[1], 768)*std+mean
                        noise2=noise2.to(device)
                        for m in range(0,Tx):
                            logit,logit_wn,_,_,Embedding_in1_wn,Embedding_in2_wn = model.predict_paraphrase(b_ids1, b_mask1, b_ids2, b_mask2, noise1, noise2)
                            prob = torch.sigmoid(logit)    # shape: (batch_size, 1)
                            prob_wn = torch.sigmoid(logit_wn)  # with noise
                            P1 = torch.cat([1 - prob, prob], dim=1)  # [P(not paraphrase), P(paraphrase)]
                            Q = torch.cat([1 - prob_wn, prob_wn], dim=1)
                            KL_divergence_PQ = P1 * torch.log((P1 + eps) / (Q + eps))
                            KL_divergence_QP = Q * torch.log((Q + eps) / (P1 + eps))
                            ls=KL_divergence_PQ+KL_divergence_QP
                            ls=ls.sum()
                            ls.backward()
                            gi1_tilde=Embedding_in1_wn.grad/Embedding_in1_wn.shape[0]
                            gi2_tilde=Embedding_in2_wn.grad/Embedding_in2_wn.shape[0]
                            gi1_tilde/=(torch.norm(gi1_tilde,p=float('inf'),dim=(1, 2)).view(gi1_tilde.shape[0],1,1)+eps)
                            gi2_tilde/=(torch.norm(gi2_tilde,p=float('inf'),dim=(1, 2)).view(gi2_tilde.shape[0],1,1)+eps)
                            noise1 += eta * gi1_tilde
                            noise2 += eta * gi2_tilde
                        logit,logit_wn,*rest = model.predict_paraphrase(b_ids1, b_mask1, b_ids2, b_mask2, noise1, noise2)
                        prob = torch.sigmoid(logit)    # shape: (batch_size, 1)
                        prob_wn = torch.sigmoid(logit_wn)  # with noise
                        P1 = torch.cat([1 - prob, prob], dim=1)  # [P(not paraphrase), P(paraphrase)]
                        Q = torch.cat([1 - prob_wn, prob_wn], dim=1)
                        KL_divergence_PQ = P1 * torch.log((P1 + eps) / (Q + eps))
                        KL_divergence_QP = Q * torch.log((Q + eps) / (P1 + eps))
                        ls=KL_divergence_PQ+KL_divergence_QP
                        max_ls,_=torch.max(ls,-1)
                        ls=ls.sum()
                        Rs=torch.sum(max_ls)/max_ls.shape[0]
                        logit2,*rest = model_copy.predict_paraphrase(b_ids1, b_mask1, b_ids2, b_mask2, noise1, noise2)
                        prob2 = torch.sigmoid(logit2)    # shape: (batch_size, 1)
                        P2 = torch.cat([1 - prob2, prob2], dim=1)  # [P(not paraphrase), P(paraphrase)]
                        KL_divergence_P1P2 = P1 * torch.log((P1 + eps) / (P2 + eps))
                        KL_divergence_P2P1 = P2 * torch.log((P2 + eps) / (P1 + eps))
                        ls2=KL_divergence_P1P2+KL_divergence_P2P1
                        D_Breg=ls2.sum()/args.batch_size
                        optimizer.zero_grad()
                        loss=F.binary_cross_entropy_with_logits(logit.squeeze(), b_labels.view(-1).float(), reduction='sum') / args.batch_size + lambda_s*Rs + mu*D_Breg
                        loss.backward()
                        optimizer.step()
                        theta_tilde_s=copy.deepcopy(model.state_dict())
                    theta_t=copy.deepcopy(theta_tilde_s)
                    theta_t_tilde = copy.deepcopy(theta_t)
                    for key in theta_tilde_s.keys():
                        theta_t_tilde[key] = (1 - Beta) * theta_tilde_s[key] + Beta * theta_t_tilde[key]
                else:
                    optimizer.zero_grad()
                    logit = model.predict_paraphrase(b_ids1, b_mask1, b_ids2, b_mask2)
                    loss = F.binary_cross_entropy_with_logits(logit.squeeze(), b_labels.view(-1).float(), reduction='sum') / args.batch_size
                    loss.backward()
                    optimizer.step()
        paraphrase_dev_acc,paraphrase_dev_f1 = model_eval_paraphrase(para_dev_dataloader, model, device)
        dev_sts_corr= model_eval_sts(sts_dev_dataloader, model, device)
        sst_dev_acc, sst_dev_f1, *_ = model_eval_sst(sst_dev_dataloader, model, device)

        print('dev sts corr = ',dev_sts_corr)
        print('dev sst acc = ',sst_dev_acc)
        print('dev sst f1 = ',sst_dev_f1)
        print('dev paraphrase acc = ',paraphrase_dev_acc)
        print('dev paraphrase f1 = ',paraphrase_dev_f1)

        overall_perf=((dev_sts_corr+1)/2+sst_dev_acc+paraphrase_dev_acc)/3
        print('Overall Performance = ',overall_perf)

        if overall_perf > best_overall_perf:
            best_overall_perf = overall_perf
            save_model(model, optimizer, args, config, args.filepath)
            print('New best model saved with overall performance =', overall_perf)


def manual_test(args):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    num_labels=1
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'num_labels': num_labels,
              'hidden_size': 768,
              'data_dir': '.',
              'fine_tune_mode': args.fine_tune_mode}

    config = SimpleNamespace(**config)

    model = MultitaskBERT(config)
    checkpoint = torch.load(args.filepath, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model = model.to(device)
    BT=BertTokenizer.from_pretrained('bert-base-uncased')
    while(1):
        Task=input("Which Task? (choose one from sst5,qqp,stsb)")
        if Task=="sst5":
            Sentence=input("Enter Sentence: ")
            encoding = BT(Sentence, return_tensors='pt', padding=True, truncation=True)
            token_ids = torch.LongTensor(encoding['input_ids']).to(device)
            attention_mask = torch.LongTensor(encoding['attention_mask']).to(device)
            pred_label=torch.argmax(model.predict_sentiment(token_ids,attention_mask),dim=1)
            if pred_label==0:
                print("The Sentence is very negative")
            elif pred_label==1:
                print("The Sentence is negative")
            elif pred_label==2:
                print("The Sentence is neutral")
            elif pred_label==3:
                print("The Sentence is positive")
            elif pred_label==4:
                print("The Sentence is very positive")
        elif Task=="qqp":
            Sentence1=input("Enter Sentence1: ")
            Sentence2=input("Enter Sentence2: ")
            
            encoding1 = BT(Sentence1, return_tensors='pt', padding=True, truncation=True)
            token_ids1 = torch.LongTensor(encoding1['input_ids']).to(device)
            attention_mask1 = torch.LongTensor(encoding1['attention_mask']).to(device)
            
            encoding2 = BT(Sentence2, return_tensors='pt', padding=True, truncation=True)
            token_ids2 = torch.LongTensor(encoding2['input_ids']).to(device)
            attention_mask2 = torch.LongTensor(encoding2['attention_mask']).to(device)

            pred_label=model.predict_paraphrase(token_ids1,attention_mask1,token_ids2,attention_mask2).sigmoid().round()
            if pred_label==0:
                print("This pair IS NOT Paraphrase")
            elif pred_label==1:
                print("This pair IS a Paraphrase")
        elif Task=="stsb":
            Sentence1=input("Enter Sentence1: ")
            Sentence2=input("Enter Sentence2: ")
            
            encoding1 = BT(Sentence1, return_tensors='pt', padding=True, truncation=True)
            token_ids1 = torch.LongTensor(encoding1['input_ids']).to(device)
            attention_mask1 = torch.LongTensor(encoding1['attention_mask']).to(device)
            
            encoding2 = BT(Sentence2, return_tensors='pt', padding=True, truncation=True)
            token_ids2 = torch.LongTensor(encoding2['input_ids']).to(device)
            attention_mask2 = torch.LongTensor(encoding2['attention_mask']).to(device)

            pred_label=model.predict_similarity(token_ids1,attention_mask1,token_ids2,attention_mask2)
            print("Similarity between the two sentences on a scale from 0 to 5 = ", pred_label.item())
            if pred_label>=0 and pred_label<1:
                print("The two sentences are on different topics")
            elif pred_label>=1 and pred_label<2:
                print("The two sentences are not equivalent, but are on the same topic")
            elif pred_label>=2 and pred_label<3:
                print("The two sentences are not equivalent, but do share some details")
            elif pred_label>=3 and pred_label<4:
                print("The two sentences are mostly equivalent but some unimportant details differ")
            elif pred_label>=4 and pred_label<5:
                print("The sentences are completely equivalent, as they mean the same thing")
        else:
            print("Invalid Task")

def test_multitask(args):
    '''Test and save predictions on the dev and test sets of all three tasks.'''
    with torch.no_grad():
        device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        saved = torch.load(args.filepath,weights_only=False)
        config = saved['model_config']

        model = MultitaskBERT(config)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        print(f"Loaded model to test from {args.filepath}")

        sst_test_data, num_labels,para_test_data, sts_test_data, snli_test_data = \
            load_multitask_data(args.sst_test,args.para_test, args.sts_test, args.snli_test, split='test')

        sst_dev_data, num_labels,para_dev_data, sts_dev_data, snli_dev_data = \
            load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev, args.snli_dev,split='dev')

        sst_test_data = SentenceClassificationTestDataset(sst_test_data, args)
        sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

        sst_test_dataloader = DataLoader(sst_test_data, shuffle=True, batch_size=args.batch_size,
                                         collate_fn=sst_test_data.collate_fn)
        sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=sst_dev_data.collate_fn)

        para_test_data = SentencePairTestDataset(para_test_data, args)
        para_dev_data = SentencePairDataset(para_dev_data, args)

        para_test_dataloader = DataLoader(para_test_data, shuffle=True, batch_size=args.batch_size,
                                          collate_fn=para_test_data.collate_fn)
        para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                         collate_fn=para_dev_data.collate_fn)

        sts_test_data = SentencePairTestDataset(sts_test_data, args)
        sts_dev_data = SentencePairDataset(sts_dev_data, args, isRegression=True)

        sts_test_dataloader = DataLoader(sts_test_data, shuffle=True, batch_size=args.batch_size,
                                         collate_fn=sts_test_data.collate_fn)
        sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=sts_dev_data.collate_fn)

        dev_sentiment_accuracy,dev_sst_y_pred, dev_sst_sent_ids, \
            dev_paraphrase_accuracy, dev_para_y_pred, dev_para_sent_ids, \
            dev_sts_corr, dev_sts_y_pred, dev_sts_sent_ids = model_eval_multitask(sst_dev_dataloader,
                                                                    para_dev_dataloader,
                                                                    sts_dev_dataloader, model, device)

        test_sst_y_pred, \
            test_sst_sent_ids, test_para_y_pred, test_para_sent_ids, test_sts_y_pred, test_sts_sent_ids = \
                model_eval_test_multitask(sst_test_dataloader,
                                          para_test_dataloader,
                                          sts_test_dataloader, model, device)

        with open(args.sst_dev_out, "w+") as f:
            print(f"dev sentiment acc :: {dev_sentiment_accuracy :.3f}")
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(dev_sst_sent_ids, dev_sst_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sst_test_out, "w+") as f:
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(test_sst_sent_ids, test_sst_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.para_dev_out, "w+") as f:
            print(f"dev paraphrase acc :: {dev_paraphrase_accuracy :.3f}")
            f.write(f"id \t Predicted_Is_Paraphrase \n")
            for p, s in zip(dev_para_sent_ids, dev_para_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.para_test_out, "w+") as f:
            f.write(f"id \t Predicted_Is_Paraphrase \n")
            for p, s in zip(test_para_sent_ids, test_para_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sts_dev_out, "w+") as f:
            print(f"dev sts corr :: {dev_sts_corr :.3f}")
            f.write(f"id \t Predicted_Similiary \n")
            for p, s in zip(dev_sts_sent_ids, dev_sts_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sts_test_out, "w+") as f:
            f.write(f"id \t Predicted_Similiary \n")
            for p, s in zip(test_sts_sent_ids, test_sts_y_pred):
                f.write(f"{p} , {s} \n")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sst_train", type=str, default="data/ids-sst-train.csv")
    parser.add_argument("--sst_dev", type=str, default="data/ids-sst-dev.csv")
    parser.add_argument("--sst_test", type=str, default="data/ids-sst-test-student.csv")

    parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
    parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")

    parser.add_argument("--sts_train", type=str, default="data/sts-train.csv")
    parser.add_argument("--sts_dev", type=str, default="data/sts-dev.csv")
    parser.add_argument("--sts_test", type=str, default="data/sts-test-student.csv")

    parser.add_argument("--snli_train", type=str, default="data/snli-train.csv")
    parser.add_argument("--snli_dev", type=str, default="data/snli-dev.csv")
    parser.add_argument("--snli_test", type=str, default="data/snli-test-student.csv")


    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--fine-tune-mode", type=str,
                        help='last-linear-layer: the BERT parameters are frozen and the task specific head parameters are updated; full-model: BERT parameters are updated as well',
                        choices=('last-linear-layer', 'full-model'), default="last-linear-layer")
    parser.add_argument("--use_gpu", action='store_true')

    parser.add_argument("--sst_dev_out", type=str, default="predictions/sst-dev-output.csv")
    parser.add_argument("--sst_test_out", type=str, default="predictions/sst-test-output.csv")

    parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
    parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

    parser.add_argument("--sts_dev_out", type=str, default="predictions/sts-dev-output.csv")
    parser.add_argument("--sts_test_out", type=str, default="predictions/sts-test-output.csv")

    parser.add_argument("--snli_dev_out", type=str, default="predictions/snli-dev-output.csv")
    parser.add_argument("--snli_test_out", type=str, default="predictions/snli-test-output.csv")

    parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.1)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    args.filepath = f'SBERT_SMART.pt' # Save path.
    seed_everything(args.seed)  # Fix the seed for reproducibility.
    train_multitask(args)
    test_multitask(args)
    manual_test(args)
