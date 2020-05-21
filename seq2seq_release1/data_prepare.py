import numpy as np
import os
import natsort
import json
import pandas as pd

import torch
from torch.utils.data import Dataset

class SkDataset(Dataset):
########initialize the dataset######
    def __init__(self):
        dir ="/home/tonner/Downloads/pytorch-openpose/data1"
        trian_sk, val_sk, query_sk, gallery_sk=self._process_data(dir)
        self.train =trian_sk
        self.val = val_sk
        self.query = query_sk
        self.gallery = gallery_sk
        self.mum_class = 124

    def _read_json(self,pid,dir):
        sk_tracklet=[]
        sk1_tracklet =[]
        i=0
        json_files = natsort.natsorted(os.listdir(os.path.join(dir, pid)))
        for json_file in json_files:
            if i <=2:
                with open(os.path.join(dir, pid, json_file), "r") as file_obj:
                    skeleton_t1 = json.load(file_obj)
                    skeleton_t_l1 = [skeleton_t1, int(pid) - 1]
                    sk1_tracklet.append(skeleton_t_l1)
                    i+=1
            else:
                with open(os.path.join(dir, pid, json_file), "r") as file_obj:
                    skeleton_t = json.load(file_obj)
                    skeleton_t_l = [skeleton_t, int(pid) - 1]
                sk_tracklet.append(skeleton_t_l)
        return sk_tracklet,sk1_tracklet
    def _test_read_json(self,pid,dir):
        q_tracklets=[]
        g_tracklets =[]

        json_files =natsort.natsorted(os.listdir(os.path.join(dir, pid)))
        for json_file in json_files:
            state_id=json_file[4:9]
            tracklet_id =json_file.split(".")[0][-3:]
            ##########gallery dataset##########
            if (state_id=="nm-04"or state_id=="nm-05"or state_id=="nm-06") and int(tracklet_id)==0:
                with open(os.path.join(dir, pid, json_file), "r") as file_obj:
                    skeleton_t1 = json.load(file_obj)
                    skeleton_t_l1 = [skeleton_t1, int(pid),state_id]
                    g_tracklets.append(skeleton_t_l1)
            elif (state_id=="cl-01"or state_id=="cl-02") and int(tracklet_id)==0:
                with open(os.path.join(dir, pid, json_file), "r") as file_obj:
                    skeleton_t = json.load(file_obj)
                    skeleton_t_l = [skeleton_t, int(pid),state_id]
                q_tracklets.append(skeleton_t_l)
        return q_tracklets ,g_tracklets

    def _process_data(self,dir):
        train_sk=[]
        val_sk=[]
        query_sk =[]
        gallery_sk =[]
        pids_list = natsort.natsorted(os.listdir(dir))
        for pid in pids_list:
            if int(pid) <=84:     ###>=44 dataset cross
                train_sk0,val_sk0=self._read_json(pid,dir)
                train_sk.append(np.array(train_sk0))
                val_sk.append(np.array(val_sk0))
            else:
                #####gallery and query#######
                query_sk0,gallery_sk0 = self._test_read_json(pid,dir)
                query_sk.append(np.array(query_sk0))
                gallery_sk.append(np.array(gallery_sk0))
        print("finish init dataset")
        return train_sk, val_sk, query_sk,gallery_sk
if __name__ == '__main__':
    dataset=SkDataset()
    print("dataset")
