#!/usr/bin/env python3

'''
Multitask BERT evaluation functions.

When training your multitask model, you will find it useful to call
model_eval_multitask to evaluate your model on the 3 tasks' dev sets.
'''

import torch
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
import numpy as np


TQDM_DISABLE = False


# Evaluate multitask model on SST only.
def model_eval_sst(dataloader, model, device):
    model.eval()  # Switch to eval model, will turn off randomness like dropout.
    y_true = []
    y_pred = []
    sents = []
    sent_ids = []
    for step, batch in enumerate(tqdm(dataloader, desc=f'eval', disable=TQDM_DISABLE)):
        b_ids, b_mask, b_labels, b_sents, b_sent_ids = batch['token_ids'],batch['attention_mask'],  \
                                                        batch['labels'], batch['sents'], batch['sent_ids']

        b_ids = b_ids.to(device)
        b_mask = b_mask.to(device)

        logits = model.predict_sentiment(b_ids, b_mask)
        logits = logits.detach().cpu().numpy()
        preds = np.argmax(logits, axis=1).flatten()

        b_labels = b_labels.flatten()
        y_true.extend(b_labels)
        y_pred.extend(preds)
        sents.extend(b_sents)
        sent_ids.extend(b_sent_ids)

    f1_sst = f1_score(y_true, y_pred, average='weighted')
    acc = accuracy_score(y_true, y_pred)

    return acc, f1_sst, y_pred, y_true, sents, sent_ids

# Evaluate multitask model on Paraphrase Dataset only.
def model_eval_paraphrase(dataloader, model, device):
    model.eval()  # Switch to eval model, will turn off randomness like dropout.
    para_y_true = []
    para_y_pred = []
    para_sent_ids = []
    for step, batch in enumerate(tqdm(dataloader, desc=f'eval', disable=TQDM_DISABLE)):
        (b_ids1, b_mask1,
            b_ids2, b_mask2,
            b_labels, b_sent_ids) = (batch['token_ids_1'], batch['attention_mask_1'],
                        batch['token_ids_2'], batch['attention_mask_2'],
                        batch['labels'], batch['sent_ids'])

        b_ids1 = b_ids1.to(device)
        b_mask1 = b_mask1.to(device)
        b_ids2 = b_ids2.to(device)
        b_mask2 = b_mask2.to(device)

        logits = model.predict_paraphrase(b_ids1, b_mask1, b_ids2, b_mask2)
        y_hat = logits.sigmoid().round().flatten().detach().cpu().numpy()
        b_labels = b_labels.flatten().cpu().numpy()

        para_y_pred.extend(y_hat)
        para_y_true.extend(b_labels)
        para_sent_ids.extend(b_sent_ids)
    f1_para = f1_score(para_y_true, para_y_pred, average='weighted')
    paraphrase_accuracy = np.mean(np.array(para_y_pred) == np.array(para_y_true))
    return paraphrase_accuracy,f1_para

# Evaluate multitask model on STS Dataset only.
def model_eval_sts(dataloader, model, device):
    model.eval()  # Switch to eval model, will turn off randomness like dropout.
    sts_y_true = []
    sts_y_pred = []
    sts_sent_ids = []
    for step, batch in enumerate(tqdm(dataloader, desc=f'eval', disable=TQDM_DISABLE)):
        (b_ids1, b_mask1,
            b_ids2, b_mask2,
            b_labels, b_sent_ids) = (batch['token_ids_1'], batch['attention_mask_1'],
                        batch['token_ids_2'], batch['attention_mask_2'],
                        batch['labels'], batch['sent_ids'])

        b_ids1 = b_ids1.to(device)
        b_mask1 = b_mask1.to(device)
        b_ids2 = b_ids2.to(device)
        b_mask2 = b_mask2.to(device)

        logits = model.predict_similarity(b_ids1, b_mask1, b_ids2, b_mask2)
        y_hat = logits.flatten().detach().cpu().numpy()
        b_labels = b_labels.flatten().cpu().numpy()

        sts_y_pred.extend(y_hat)
        sts_y_true.extend(b_labels)
        sts_sent_ids.extend(b_sent_ids)
    pearson_mat = np.corrcoef(sts_y_pred,sts_y_true)
    sts_corr = pearson_mat[1][0]
    return sts_corr


def model_eval_simcse(dataloader, model, device):
    model.eval()  # Switch to eval model, will turn off randomness like dropout.
    para_y_true = []
    para_y_pred = []
    para_sent_ids = []
    for step, batch in enumerate(tqdm(dataloader, desc=f'eval', disable=TQDM_DISABLE)):
        (b_ids1, b_mask1,
            b_ids2, b_mask2,
            b_labels) = (batch['token_ids_1'], batch['attention_mask_1'],
                        batch['token_ids_2'], batch['attention_mask_2'],
                        batch['labels'])

        b_ids1 = b_ids1.to(device)
        b_mask1 = b_mask1.to(device)
        b_ids2 = b_ids2.to(device)
        b_mask2 = b_mask2.to(device)

        logits = model.predict_SNLI(b_ids1, b_mask1, b_ids2, b_mask2)
        y_hat = np.argmax(logits.detach().cpu().numpy(), axis=1).flatten()
        b_labels = b_labels.flatten().cpu().numpy()

        para_y_pred.extend(y_hat)
        para_y_true.extend(b_labels)
    paraphrase_accuracy = np.mean(np.array(para_y_pred) == np.array(para_y_true))
    return paraphrase_accuracy

# # Evaluate SimCSE model using contrastive loss.
# def model_eval_simcse(dataloader, model, device):
#     """
#     Evaluates the SimCSE model on a dataset using NT-Xent contrastive loss.

#     Args:
#         dataloader (DataLoader): Evaluation dataset loader.
#         model (MultitaskBERT): Trained SimCSE model.
#         device (torch.device): Device (CPU/GPU).

#     Returns:
#         float: Average contrastive loss over the dataset.
#     """
#     model.eval()
#     total_loss = 0
#     num_batches = 0

#     with torch.no_grad():  # Disable gradient computation
#         for batch in tqdm(dataloader, desc="Evaluating SimCSE", disable=TQDM_DISABLE):
#             # Extract input sentences
#             b_ids = batch['token_ids'].to(device)
#             b_mask = batch['attention_mask'].to(device)

#             # Generate two embeddings with different dropout masks
#             z_i = model(b_ids, b_mask)  # First forward pass
#             z_j = model(b_ids, b_mask)  # Second forward pass (dropout applied differently)

#             # Compute contrastive loss
#             loss = model.contrastive_loss(z_i, z_j)

#             total_loss += loss.item()
#             num_batches += 1

#     avg_loss = total_loss / num_batches
#     print(f"SimCSE Evaluation: Avg Contrastive Loss = {avg_loss:.4f}")
#     return avg_loss  # Return average loss over all batches


# Evaluate multitask model on dev sets.
def model_eval_multitask(sentiment_dataloader,
                         paraphrase_dataloader,
                         sts_dataloader,
                         model, device):
    model.eval()  # Switch to eval model, will turn off randomness like dropout.

    with torch.no_grad():
        # Evaluate sentiment classification.
        sst_y_true = []
        sst_y_pred = []
        sst_sent_ids = []
        for step, batch in enumerate(tqdm(sentiment_dataloader, desc=f'eval', disable=TQDM_DISABLE)):
            b_ids, b_mask, b_labels, b_sent_ids = batch['token_ids'], batch['attention_mask'], batch['labels'], batch['sent_ids']

            b_ids = b_ids.to(device)
            b_mask = b_mask.to(device)

            logits = model.predict_sentiment(b_ids, b_mask)
            y_hat = logits.argmax(dim=-1).flatten().cpu().numpy()
            b_labels = b_labels.flatten().cpu().numpy()

            sst_y_pred.extend(y_hat)
            sst_y_true.extend(b_labels)
            sst_sent_ids.extend(b_sent_ids)
        f1_sst = f1_score(sst_y_true, sst_y_pred, average='weighted')
        sentiment_accuracy = np.mean(np.array(sst_y_pred) == np.array(sst_y_true))

        # Evaluate paraphrase detection.
        para_y_true = []
        para_y_pred = []
        para_sent_ids = []
        for step, batch in enumerate(tqdm(paraphrase_dataloader, desc=f'eval', disable=TQDM_DISABLE)):
            (b_ids1, b_mask1,
             b_ids2, b_mask2,
             b_labels, b_sent_ids) = (batch['token_ids_1'], batch['attention_mask_1'],
                          batch['token_ids_2'], batch['attention_mask_2'],
                          batch['labels'], batch['sent_ids'])

            b_ids1 = b_ids1.to(device)
            b_mask1 = b_mask1.to(device)
            b_ids2 = b_ids2.to(device)
            b_mask2 = b_mask2.to(device)

            logits = model.predict_paraphrase(b_ids1, b_mask1, b_ids2, b_mask2)
            y_hat = logits.sigmoid().round().flatten().cpu().numpy()
            b_labels = b_labels.flatten().cpu().numpy()

            para_y_pred.extend(y_hat)
            para_y_true.extend(b_labels)
            para_sent_ids.extend(b_sent_ids)
        f1_para = f1_score(para_y_true, para_y_pred, average='weighted')    
        paraphrase_accuracy = np.mean(np.array(para_y_pred) == np.array(para_y_true))

        # Evaluate semantic textual similarity.
        sts_y_true = []
        sts_y_pred = []
        sts_sent_ids = []
        for step, batch in enumerate(tqdm(sts_dataloader, desc=f'eval', disable=TQDM_DISABLE)):
            (b_ids1, b_mask1,
             b_ids2, b_mask2,
             b_labels, b_sent_ids) = (batch['token_ids_1'], batch['attention_mask_1'],
                          batch['token_ids_2'], batch['attention_mask_2'],
                          batch['labels'], batch['sent_ids'])

            b_ids1 = b_ids1.to(device)
            b_mask1 = b_mask1.to(device)
            b_ids2 = b_ids2.to(device)
            b_mask2 = b_mask2.to(device)

            logits = model.predict_similarity(b_ids1, b_mask1, b_ids2, b_mask2)
            y_hat = logits.flatten().cpu().numpy()
            b_labels = b_labels.flatten().cpu().numpy()

            sts_y_pred.extend(y_hat)
            sts_y_true.extend(b_labels)
            sts_sent_ids.extend(b_sent_ids)
        pearson_mat = np.corrcoef(sts_y_pred,sts_y_true)
        sts_corr = pearson_mat[1][0]


        print(f'Sentiment classification accuracy: {sentiment_accuracy:.3f}  Sentiment classification F1 score: {f1_sst:.3f}')
        print(f'Paraphrase detection accuracy: {paraphrase_accuracy:.3f}  Paraphrase detection F1 score: {f1_para:.3f}')
        print(f'Semantic Textual Similarity correlation: {sts_corr:.3f}')

        return (sentiment_accuracy, sst_y_pred, sst_sent_ids,
                paraphrase_accuracy, para_y_pred, para_sent_ids,
                sts_corr, sts_y_pred, sts_sent_ids,)

# Evaluate multitask model on test sets.
def model_eval_test_multitask(sentiment_dataloader,
                         paraphrase_dataloader,
                         sts_dataloader,
                         model, device):
    model.eval()  # Switch to eval model, will turn off randomness like dropout.

    with torch.no_grad():
        # Evaluate sentiment classification.
        sst_y_pred = []
        sst_sent_ids = []
        for step, batch in enumerate(tqdm(sentiment_dataloader, desc=f'eval', disable=TQDM_DISABLE)):
            b_ids, b_mask, b_sent_ids = batch['token_ids'], batch['attention_mask'],  batch['sent_ids']

            b_ids = b_ids.to(device)
            b_mask = b_mask.to(device)

            logits = model.predict_sentiment(b_ids, b_mask)
            y_hat = logits.argmax(dim=-1).flatten().cpu().numpy()

            sst_y_pred.extend(y_hat)
            sst_sent_ids.extend(b_sent_ids)

        # Evaluate paraphrase detection.
        para_y_pred = []
        para_sent_ids = []
        for step, batch in enumerate(tqdm(paraphrase_dataloader, desc=f'eval', disable=TQDM_DISABLE)):
            (b_ids1, b_mask1,
             b_ids2, b_mask2,
             b_sent_ids) = (batch['token_ids_1'], batch['attention_mask_1'],
                          batch['token_ids_2'], batch['attention_mask_2'],
                          batch['sent_ids'])

            b_ids1 = b_ids1.to(device)
            b_mask1 = b_mask1.to(device)
            b_ids2 = b_ids2.to(device)
            b_mask2 = b_mask2.to(device)

            logits = model.predict_paraphrase(b_ids1, b_mask1, b_ids2, b_mask2)
            y_hat = logits.sigmoid().round().flatten().cpu().numpy()

            para_y_pred.extend(y_hat)
            para_sent_ids.extend(b_sent_ids)

        # Evaluate semantic textual similarity.
        sts_y_pred = []
        sts_sent_ids = []
        for step, batch in enumerate(tqdm(sts_dataloader, desc=f'eval', disable=TQDM_DISABLE)):
            (b_ids1, b_mask1,
             b_ids2, b_mask2,
             b_sent_ids) = (batch['token_ids_1'], batch['attention_mask_1'],
                          batch['token_ids_2'], batch['attention_mask_2'],
                          batch['sent_ids'])

            b_ids1 = b_ids1.to(device)
            b_mask1 = b_mask1.to(device)
            b_ids2 = b_ids2.to(device)
            b_mask2 = b_mask2.to(device)

            logits = model.predict_similarity(b_ids1, b_mask1, b_ids2, b_mask2)
            y_hat = logits.flatten().cpu().numpy()

            sts_y_pred.extend(y_hat)
            sts_sent_ids.extend(b_sent_ids)
        
        return (sst_y_pred, sst_sent_ids,
                para_y_pred, para_sent_ids,
                sts_y_pred, sts_sent_ids,)
