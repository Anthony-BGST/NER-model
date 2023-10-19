import jieba.posseg as pseg

def read_file_and_segment(file_path):
    pos_list = []
    with open(file_path, 'r', encoding='utf-8') as f:
        sents = f.read().split("\n\n")
        seg_sents = [item.split("\n") for item in sents]
        for idx, line in enumerate(seg_sents):
            raw_labels = "".join([item.split(" ")[0] for item in line])
            for word, pos in pseg.cut(raw_labels):
                if pos not in pos_list:
                    pos_list.append(pos)
    return pos_list

if __name__ == '__main__':
    pos_list = read_file_and_segment('./data.txt')
    pos_dict = {pos: i for i, pos in enumerate(pos_list)}

