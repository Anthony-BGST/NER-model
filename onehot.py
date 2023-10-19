import pypinyin
import numpy as np

def get_all_pinyin():
    pinyin_set = set()
    for ch in range(0x4e00, 0x9fff): 
        char = chr(ch)
        pinyin = pypinyin.pinyin(char, style=pypinyin.Style.TONE3)[0][0]
        pinyin_set.add(pinyin)
    return list(pinyin_set)

def create_pinyin_to_index(all_pinyin):
    return {pinyin: index for index, pinyin in enumerate(all_pinyin)}

def one_hot_encode_pinyin_list(pinyin_list, pinyin_to_index):
    num_pinyin = len(pinyin_to_index)
    one_hot_list = []
    for pinyin in pinyin_list:
        one_hot = np.zeros(num_pinyin)
        if pinyin in pinyin_to_index:
            index = pinyin_to_index[pinyin]
            one_hot[index] = 1
        one_hot_list.append(one_hot)
    return one_hot_list