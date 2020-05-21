import re
import os
import sys
import spacy
import numpy as np
import errno
import os.path as osp
import natsort
from torchtext.data import Field, BucketIterator
from torchtext.datasets import Multi30k


def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

def load_dataset(batch_size):
    spacy_de = spacy.load('de_core_news_sm')
    spacy_en = spacy.load('en_core_web_sm')
    url = re.compile('(<url>.*</url>)')

    def tokenize_de(text):
        return [tok.text for tok in spacy_de.tokenizer(url.sub('@URL@', text))]

    def tokenize_en(text):
        return [tok.text for tok in spacy_en.tokenizer(url.sub('@URL@', text))]

    DE = Field(tokenize=tokenize_de, include_lengths=True,
               init_token='<sos>', eos_token='<eos>')
    EN = Field(tokenize=tokenize_en, include_lengths=True,
               init_token='<sos>', eos_token='<eos>')
    train, val, test = Multi30k.splits(exts=('.de', '.en'), fields=(DE, EN))
    DE.build_vocab(train.src, min_freq=2)
    EN.build_vocab(train.trg, max_size=10000)
    train_iter, val_iter, test_iter = BucketIterator.splits(
            (train, val, test), batch_size=batch_size, repeat=False)
    return train_iter, val_iter, test_iter, DE, EN

def cross_tracklet(file_npy):

    all_tracklets=file_npy
    print("len of all_tracklets",len(all_tracklets))
    new_tracklets=[]

    step =4
    seq_len=24
    for i in range(all_tracklets.shape[0]):
        start_index,end_index,_,_,_, = all_tracklets[i]
        start_index = int(start_index)
        end_index = int(end_index)
        len_tracklet=end_index - start_index
        n =(len_tracklet-seq_len)//step + 1   ###滑动窗口滑动的次数（包含起始位）
        for j in range(n):
            start_index1=start_index +step*j
            end_index1 =start_index1+seq_len-1
            new_tracklet =[start_index1,end_index1,all_tracklets[i][2],all_tracklets[i][3],all_tracklets[i][4]]
            new_tracklets.append(new_tracklet)
    print("len of new_tracklets",len(new_tracklets))
    return new_tracklets

class Logger(object):
    """
    Write console output to external text file.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()

class AverageMeter(object):
    """Computes and stores the average and current value.

       Code imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



if __name__ == '__main__':

    sys.stdout = Logger(os.path.join("./", 'log_train_lstm.txt'))
