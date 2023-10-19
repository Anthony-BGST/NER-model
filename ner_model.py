import torch
from torch import nn
from transformers import BertPreTrainedModel, BertModel
from TorchCRF import CRF
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    def __init__(self, attention_dropout=0.3):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, scale=None, attn_mask=None):
        attention = torch.matmul(q, k.transpose(-1,-2))
        if scale:
            attention = attention * scale
        if attn_mask is not None:
            attention = attention.masked_fill_(attn_mask, -np.inf)
        attention = self.softmax(attention)  
        context = torch.matmul(attention, v) 
        return context

class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim=768, num_heads=8, dropout=0.3):
        super(MultiHeadAttention, self).__init__()
        self.model_dim = model_dim
        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.dot_product_attention = ScaledDotProductAttention(dropout)
        self.linear_final = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.dim_per_head)
        return x.permute([0, 2, 1, 3])

    def forward(self, key, value, query, attn_mask=None):
        residual = query
        num_heads = self.num_heads
        batch_size = key.size(0)
        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)
        query = self.split_heads(query, batch_size)  
        key = self.split_heads(key, batch_size)  
        value = self.split_heads(value, batch_size)
        scale = (key.size(-1) // num_heads) ** -0.5
        context = self.dot_product_attention(
            query, key, value, scale,
            attn_mask)
        context = context.permute([0, 2, 1, 3])
        context = context.reshape(batch_size, -1, self.model_dim)
        output = self.linear_final(context)
        output = self.dropout(output)
        output = self.layer_norm(residual + output)
        return output


class NER_Model(BertPreTrainedModel):
    def __init__(self, config, use_pinyin=None, use_pos=None):
        super(NER_Model, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.lstm = nn.LSTM(input_size=config.hidden_size,hidden_size=config.hidden_size//2,num_layers=1,bidirectional=True,batch_first=True)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.pinyin_embedding = nn.Embedding(540, 768)  
        self.pos_embedding = nn.Embedding(54, 768)  
        self.crf = CRF(config.num_labels)  
        self.init_weights()
        self.use_pinyin = use_pinyin
        self.use_pos = use_pos
        self.attention = MultiHeadAttention(model_dim=768, num_heads=8, dropout=0.3)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None, pinyin_ids_tensor=None, pos_ids_tensor=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        pinyin_embedding = self.pinyin_embedding(pinyin_ids_tensor)
        pos_embedding = self.pos_embedding(pos_ids_tensor)
        fuse_embedding = self.attention(pinyin_embedding,pos_embedding,sequence_output)
        fuse_embedding = torch.add(pinyin_embedding,pos_embedding)
        sequence_output = torch.add(fuse_embedding, sequence_output)
        sequence_output, _ = self.lstm(sequence_output)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        if labels is not None: 
            loss_mask = labels.gt(-1)
            loss = self.crf(logits, labels, loss_mask) * (-1)
            outputs = (loss,) 
        else:
            outputs = (logits,)
        return outputs
