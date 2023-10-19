import os
import torch
from torch.utils.data import RandomSampler
from transformers import AdamW
from transformers import BertTokenizer
from tqdm import trange
import numpy as np
import json
import warnings
warnings.filterwarnings("ignore")
from ner_model import NER_Model
from log import configure_logger
from utils import read_ner_data_from_file
from sklearn.metrics import classification_report

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def accuracy_per_tag(preds, labels, label_map):
    total_match = preds[preds == labels]
    logger.info("PRF1 per each tag:")
    for label, label_idx in label_map.items():
        labeli = np.sum(labels == label_idx)
        predsi = np.sum(preds == label_idx)
        matchi = np.sum(total_match == label_idx)
        pre = matchi / (predsi + 1e-8)
        rec = matchi / (labeli + 1e-8)
        logger.info("\ttag: {:<10}, P:{:<.2f}, R:{:<.2f}, F1:{:<.2f}".format(
            label, pre, rec, 2 * pre * rec / (pre + rec + 1e-8)))

curr_dir = os.path.dirname(__file__)
log_path = os.path.join(curr_dir, "ccks2019.log")
bert_path = os.path.join(curr_dir, './model/bert-base-chinese')
bert_path_ner = os.path.join(curr_dir, '/model/ner_model/')
#######################################ccks2019#######################################
train_filepath = os.path.join(curr_dir, "./data/ccks2019/train.txt")
test_filepath = os.path.join(curr_dir, "./data/ccks2019/test.txt")
dev_filepath = os.path.join(curr_dir, "./data/ccks2019/dev.txt")
#######################################ccks2018#######################################
# train_filepath = os.path.join(curr_dir, "./data/ccks2018/train.txt")
# test_filepath = os.path.join(curr_dir, "./data/ccks2018/test.txt")
# dev_filepath = os.path.join(curr_dir, "./data/ccks2018/dev.txt")
# #######################################WEIBO#######################################
# train_filepath = os.path.join(curr_dir, "./data/weibo/train.txt")
# test_filepath = os.path.join(curr_dir, "./data/weibo/test.txt")
# dev_filepath = os.path.join(curr_dir, "./data/weibo/dev.txt")

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
logger.info("CUDA INFO: %s", os.getenv('CUDA_VISIBLE_DEVICES', ""))
batch_size = 16 
max_seq_length = 128
train_loss_epoch = []
global_loss = float("inf")
epochs = 100
use_pinyin = True  
use_pos = True   
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载切词器
tokenizer = BertTokenizer.from_pretrained(bert_path)

label_map = {'O': 0, 'B-DISEASE': 1, 'I-DISEASE': 2, 'B-OPERATION': 3, 'I-OPERATION': 4, 'B-ANATOMY': 5,'I-ANATOMY': 6, 'B-TESTIMAGE': 7, 'I-TESTIMAGE': 8, 'B-DRUG': 9, 'I-DRUG': 10,'B-TESTLAB': 11, 'I-TESTLAB': 12}
# label_map = {'O': 0, "B-身体部位": 1, "I-身体部位": 2, "B-检查和检验": 3, "I-检查和检验": 4, "B-症状和体征": 5, "I-症状和体征": 6, "B-治疗": 7, "I-治疗": 8, "B-疾病和诊断": 9, "I-疾病和诊断": 10}
# label_map = { "O": 0, "B-PER": 1, "I-PER": 2, "B-LOC": 3, "I-LOC": 4, "B-GPE": 5, "I-GPE": 6, "B-ORG": 7, "I-ORG": 8}

label_map_rev = {v: k for k, v in label_map.items()}
logger.info("LABELS: %s", json.dumps(label_map, ensure_ascii=False))
logger.info("BATCH SIZE: %d", batch_size)
logger.info("MAX SEQUENCE LENGTH: %d", max_seq_length)
logger.info("MODEL INFO: %s", bert_path)

train_data,train_words,train_labels = read_ner_data_from_file(train_filepath,tokenizer,label_map=label_map,model="bert",max_seq_length=max_seq_length,splitor=" ",sp_token_label="O",batch_size=batch_size,sampler=RandomSampler,logger=logger)
test_data,test_words,test_labels = read_ner_data_from_file(test_filepath,tokenizer,label_map=label_map,model="bert",max_seq_length=max_seq_length,splitor=" ",sp_token_label="O",batch_size=batch_size,sampler=RandomSampler,logger=logger)
dev_data,dev_words,dev_labels = read_ner_data_from_file(dev_filepath,tokenizer,label_map=label_map,model="bert",max_seq_length=max_seq_length,splitor=" ",sp_token_label="O",batch_size=batch_size,sampler=RandomSampler,logger=logger)

def train():
    global global_loss
    model = NER_Model.from_pretrained(bert_path, num_labels=len(label_map),use_pinyin=use_pinyin,use_pos=use_pos)
    model.to(device)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        'weight_decay_rate': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5)
    for epoch in trange(epochs, desc="Epoch"):
        print("training...")
        model.train()
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(train_data):
            if step > 1e5:
                break
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_seg, b_labels, b_index, b_pinyin, b_pos = batch 
            optimizer.zero_grad()
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels, pinyin_ids_tensor=b_pinyin, pos_ids_tensor=b_pos) 
            loss = outputs[0]
            sumloss = loss.sum()
            sumloss.backward()
            if step % 100 == 0:
                logger.info("step: %d ==> LOSS: %f", step, sumloss.item())
            optimizer.step()
            tr_loss += sumloss
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1
        tr_loss = tr_loss / nb_tr_steps 
        train_loss_epoch.append(tr_loss)
        logger.info("Epoch: %d, Train loss: %f", epoch, tr_loss)
        if tr_loss < global_loss:
            model.save_pretrained(os.path.join(curr_dir, bert_path_ner))
            global_loss = tr_loss
        print("testing...")
        model.eval()
        eval_accuracy = 0
        nb_eval_steps = 0
        total_labels = np.array([])
        total_preds = np.array([])
        for idx, batch in enumerate(dev_data):
            if idx > 1e5:
                break
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_seg, b_labels, b_index, b_pinyin, b_pos = batch
            with torch.no_grad():
                outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, pinyin_ids_tensor=b_pinyin, pos_ids_tensor=b_pos) 
            logits = outputs[0]
            active_loss = b_input_mask.view(-1) == 1
            active_logits = logits.view(-1, len(label_map))[active_loss]
            active_labels = b_labels.view(-1)[active_loss]
            label_preds = active_logits.detach().cpu().numpy()
            label_ids = active_labels.to('cpu').numpy()
            pred_flat = np.argmax(label_preds, axis=1).flatten()
            labels_flat = label_ids.flatten()
            tmp_eval_accuracy = np.sum(pred_flat == labels_flat) / len(labels_flat)
            total_labels = np.hstack((total_labels, labels_flat))
            total_preds = np.hstack((total_preds, pred_flat))
            eval_accuracy += tmp_eval_accuracy
            nb_eval_steps += 1
        report = classification_report(total_labels, total_preds, target_names=label_map_rev.values(),digits=4)
        logger.info("Classification Report:\n{}".format(report))
        logger.info("Validation Accuracy: %f", eval_accuracy / nb_eval_steps)

def test():
    logger.debug("\ntest data-----------------\n")
    model = NER_Model.from_pretrained(bert_path_ner, num_labels=len(label_map),use_pinyin=use_pinyin)
    model.eval()
    total_map = {}
    for idx, batch in enumerate(test_data):
        b_input_ids, b_input_mask, b_seg, b_labels, b_index, b_pinyin, b_pos = batch
        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, pinyin_ids_tensor=b_pinyin,pos_ids_tensor=b_pos)
        logits = outputs[0]
        active_mask = b_input_mask == 1
        preds = np.argmax(logits, axis=2).numpy()
        for i in range(preds.shape[0]):
            preds_indices = preds[i][active_mask[i]][1:-1]
            preds_labels = [label_map_rev[k] for k in preds_indices]
            logger.debug("\n-----------------\n")
            raw_words = test_words[b_index[i]]
            raw_labels = test_labels[b_index[i]]
            logger.debug("raw_words: %s, %d", "|".join(raw_words), len(raw_words))
            logger.debug("raw_label: %s, %d", "|".join(raw_labels), len(raw_labels))
            logger.debug("pred_labels: %s, %d", "|".join(preds_labels), len(preds_labels))
    logger.debug("TEST result: %s", json.dumps(total_map))
    for k, v in total_map.items():
        logger.debug("%s: P[%.2f], R[%.2f]", k, v[2]/(v[1]+.001), v[2]/(v[0]+.001))


if __name__ == "__main__":
    train()
