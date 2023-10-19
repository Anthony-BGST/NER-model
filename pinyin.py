import re
import pypinyin

def read_file_and_segment(file_path):
    pinyin_list = []
    with open(file_path, 'r', encoding='utf-8') as f:
        sents = f.read().split("\n\n")
        seg_sents = [item.split("\n") for item in sents]
        for idx, line in enumerate(seg_sents):
            raw_labels = "".join([item.split(" ")[0] for item in line])
            pinyin_ = pypinyin.pinyin(raw_labels, style=pypinyin.NORMAL)
            for pinyin in pinyin_:
                pinyin = ''.join(pinyin)
                if not re.search("[A-Z;Ⅱβ　Ⅲ\[>Ｉ\:\]\\‘＂ＢｒａｕｎＣＴ\d,。！？?.%、\+（）\)\(×/*#；，”“＜\-’°：\"－~＋【】]", pinyin):
                    if pinyin not in pinyin_list:
                        pinyin_list.append(pinyin)
    return pinyin_list

if __name__ == '__main__':
    pinyin_list = read_file_and_segment('./data.txt')
    pinyin_vocab = open("pinyin.txt", 'w', encoding='utf-8')
    for py in list(set(pinyin_list)):
        pinyin_vocab.write(py+"\n")

