import torch
import pypinyin
import jieba.posseg as pseg
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

with open('./data/feature_file/pinyin.txt', 'r', encoding='utf-8') as f:
    pinyin_vocab = {pinyin.strip(): i for i, pinyin in enumerate(f)}
pos_vocab = {'a': 0, 'n': 1, 'v': 2, 'x': 3, 'i': 4, 'd': 5, 'k': 6, 'c': 7, 'j': 8, 'f': 9, 'l': 10, 'ng': 11, 'vg': 12, 'ug': 13, 'm': 14, 'vn': 15, 'nr': 16, 'eng': 17, 'g': 18, 'q': 19, 'r': 20, 't': 21, 'p': 22, 'u': 23, 'b': 24, 'nz': 25, 'nrt': 26, 'tg': 27, 'y': 28, 's': 29, 'zg': 30, 'z': 31, 'ad': 32, 'uj': 33, 'ns': 34, 'yg': 35, 'an': 36, 'uz': 37, 'uv': 38, 'nt': 39, 'vd': 40, 'h': 41, 'mq': 42, 'dg': 43, 'o': 44, 'ul': 45, 'ud': 46, 'nrfg': 47, 'vi': 48, 'ag': 49, 'e': 50, 'df': 51, 'rr': 52, 'vq': 53}
def read_ner_data_from_file(filepath,tokenizer,label_map={},model='bert',max_seq_length=256,splitor="\t",
                            sp_token_label="O",batch_size=32,sampler=RandomSampler,logger=None):
    logger.info("Reading NER data...")
    def mysplit(wordtag, splitor):
        try:
            word, label = wordtag.rstrip().split(splitor)
        except:
            word = "\t"
            label = "O"
        return word, label
    input_ids = []
    input_mask = []
    segment_ids = []
    index_ids = []
    input_words = []
    input_labels = []
    input_labelids = []
    pinyin_ids = []  
    pos_ids = []  
    max_len = 0
    line_cnt = 0
    with open(filepath, encoding='utf-8') as fobj:
        sents = fobj.read().split("\n\n")
        seg_sents = [item.split("\n") for item in sents]
        for idx, seg in enumerate(seg_sents):
            line_cnt += len(seg) + 1
            if line_cnt > 100000000:
                break
            seg = [item for item in seg if len(item) > 2]
            if not seg:
                continue
            raw_words, raw_labels = list(zip(*[mysplit(item, splitor) for item in seg]))
            assert len(raw_words) == len(raw_labels)
            if len(raw_words) > max_seq_length - 2:
                continue
            text = "".join(raw_words)
            pinyin_seq = []
            for char in text:
                pinyin = pypinyin.lazy_pinyin(char, style=pypinyin.Style.NORMAL)[0]
                pinyin_id = pinyin_vocab.get(pinyin, 0)
                pinyin_seq.append(pinyin_id)
            pinyin_seq += [0] * (max_seq_length - len(text))  
            pinyin_ids.append(pinyin_seq)
            pos_ids_ = []
            for word, pos in pseg.cut(text):
                pos_id = pos_vocab.get(pos, 0)
                for char in word:
                    pos_ids_.append(pos_id)
            pos_ids_ += [0] * (max_seq_length - len(text))
            pos_ids.append(pos_ids_)
            tokens_id = tokenizer.convert_tokens_to_ids(raw_words)
            labels_id = [label_map[label] for label in raw_labels]
            if model == "bert":
                tokens_id = [tokenizer.cls_token_id] + tokens_id + [tokenizer.sep_token_id]
                labels_id = [label_map[sp_token_label]] + labels_id + [label_map[sp_token_label]]
                inputs_mask = [1] * len(tokens_id)
                tokens_id += [tokenizer.pad_token_id] * (max_seq_length - len(tokens_id))
                inputs_mask += [0] * (max_seq_length - len(inputs_mask))
                segments_id = [0] * len(tokens_id)
                labels_id += [label_map[sp_token_label]] * (max_seq_length - len(labels_id))
            else:
                raise ValueError("Not supported model name for [%s]" % model)
            assert len(tokens_id) == max_seq_length
            assert len(inputs_mask) == max_seq_length
            assert len(labels_id) == max_seq_length
            assert len(segments_id) == max_seq_length
            input_ids.append(tokens_id)
            input_mask.append(inputs_mask)
            segment_ids.append(segments_id)
            input_labels.append(raw_labels)
            input_labelids.append(labels_id)
            input_words.append(raw_words)
            if len(raw_words) > max_len:
                max_len = len(raw_words)
    index_ids = list(range(len(input_ids)))
    input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)
    input_mask_tensor = torch.tensor(input_mask, dtype=torch.long)
    segment_ids_tensor = torch.tensor(segment_ids, dtype=torch.long)
    index_ids_tensor = torch.tensor(index_ids, dtype=torch.long)
    input_labelids_tensor = torch.tensor(input_labelids, dtype=torch.long)
    pinyin_ids_tensor = torch.tensor(pinyin_ids, dtype=torch.long)
    pos_ids_tensor = torch.tensor(pos_ids, dtype=torch.long)
    dataset = TensorDataset(input_ids_tensor, input_mask_tensor, segment_ids_tensor, input_labelids_tensor, index_ids_tensor, pinyin_ids_tensor, pos_ids_tensor) 
    seq_sampler = sampler(dataset)
    pred_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=seq_sampler)
    logger.info("Data length: %d", len(input_words))
    return pred_dataloader, input_words, input_labels
