import os
import argparse
import sys
from data_prepare import SkDataset
from utils import load_dataset, Logger
import numpy as np
import torch
from torch.autograd import Variable
from model import LSTMGait,RNN
from eval_metrics import evaluate
import pandas as pd
from tensorboardX import SummaryWriter

def parse_arguments():
    p = argparse.ArgumentParser(description='Hyperparams')
    p.add_argument('-epochs', type=int, default=200,
                   help='number of epochs for train')
    p.add_argument('-batch_size', type=int, default=32,
                   help='number of epochs for train')
    p.add_argument('-seq_size', type=int, default=24,
                   help='number of epochs for train')
    p.add_argument('-embed_size', type=int, default=36,
                   help='number of epochs for train')
    p.add_argument('-hidden_size', type=int, default=256,
                   help='number of epochs for train')
    p.add_argument('-lr', type=float, default=0.0001,
                   help='initial learning rate')      ##学习速率为0.001时候会loss出现震荡
    p.add_argument('-grad_clip', type=float, default=10.0,
                   help='in case of gradient explosion')      ###预防梯度爆炸
    p.add_argument('-seed', type=float, default=1.0,
                   help='in case of gradient explosion')
    p.add_argument('--gpu-devices', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
    p.add_argument('--weight-decay', default=5e-04, type=float,
                        help="weight decay (default: 5e-04)")
    p.add_argument('--stepsize', default=200, type=int,
                        help="stepsize to decay learning rate (>0 means this is enabled)")
    p.add_argument('--gamma', default=0.1, type=float,
                        help="learning rate decay")
    p.add_argument('--save_dir', default="./log", type=str,
                   help="save model checkpoint")
    p.add_argument('--eval_step', default=50, type=int,
                   help="run evaluation for every N epochs")
    p.add_argument("--test_batch",default=1,type=int,help="has to be 1")
    return p.parse_args()

def data_to_tensor(data):
    seq_tracklets=[]
    for i in range(len(data)):
        for j in range(len(data[i])-1):
            if j==0:
                data_tracklet=np.expand_dims(np.array(data[i][j]),0)
                data_tracklet1 =np.expand_dims(np.array(data[i][j+1]),0)
                data_tracklets=np.concatenate((data_tracklet,data_tracklet1),0)
            else:
                data_tracklet1 = np.expand_dims(np.array(data[i][j + 1]), 0)
                data_tracklets = np.concatenate((data_tracklets, data_tracklet1), 0)

        #####abnormal data detect and process#####
        PD=pd.DataFrame(data_tracklets)
        nan_num =PD.isnull().sum()
        if nan_num.sum() >0:
            print("nan eixist########",nan_num.sum())
            data_tracklets=np.array(PD.fillna(0))
        # print("max",np.max(data_tracklets))
        # print("min",np.min(data_tracklets))
        seq_tracklets.append(torch.from_numpy(data_tracklets).double())
    batch_tracklets=torch.stack(seq_tracklets,0)  ###torch.stack(list,0),list中每个元素为tensor中第0维度的每个元素
    return batch_tracklets

def test_data_iter(dataset,test_batch):
    print("prepare the test dataset")
    test_batch_sk=[]
    skeleton_all =[]
    for i in range(len(dataset)):
        for j in range(dataset[i].shape[0]):
            skeleton_all.append(dataset[i][j])
    for i in range(len(skeleton_all)):
        for j in range(test_batch):
            skeletons_cordinate=[]
            skeletons_ID =[]
            skeletons_state =[]

            skeleton_cordinate = skeleton_all[i][0] ###skeleton_all  numpy array
            skeleton_ID = skeleton_all[i][1]
            skeleton_state =skeleton_all[i][2]

            skeletons_cordinate.append(skeleton_cordinate)
            skeletons_ID.append(skeleton_ID)
            skeletons_state.append(skeleton_state)

        batch_sk = (skeletons_cordinate, skeletons_ID[0],skeletons_state[0])
        test_batch_sk.append(batch_sk)

    return test_batch_sk


def test(model,queryloader,galleryloader,use_gpu,ranks=[1,5,10,20]):
    model.eval()
    ########process query dataset ##########
    qf ,q_pids,q_states =[],[],[]
    for batch_idx,(input,pids,state) in enumerate(queryloader):
        n=len(input)
        input = data_to_tensor(input)
        if use_gpu:
            input =input.cuda()
        input =Variable(input)
        with torch.no_grad():
            y,features = model(input)
        #####process  features #######
        features = features.view(n, -1)
        features = torch.mean(features, 0)
        features = features.data.cpu()
        qf.append(features)
        q_pids.append(pids)
        q_states.append(state)   #####list.extend() put a new list into list
    qf = torch.stack(qf)

    ##########process gallery dataset ##########
    gf, g_pids, g_states = [], [], []
    for batch_idx, (input, pids, state) in enumerate(galleryloader):
        input = data_to_tensor(input)
        if use_gpu:
            input = input.cuda()
        input = Variable(input)
        with torch.no_grad():
            y, features = model(input)
        features = features.view(n, -1)
        features = torch.mean(features, 0)
        features = features.data.cpu()
        gf.append(features)
        g_pids.append(pids)
        g_states.append(state)  ##list.extend() put a new list into list
    gf = torch.stack(gf)

    ######process q_features and g_features ########
    #1.calculate the dis Martix
    #2.evalute
    print("caculate")
    m, n = qf.size(0), gf.size(0)
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, qf, gf.t())
    distmat = distmat.numpy()

    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_states, g_states)

    print("result-----------")
    print("mAP: {:.1%}".format(mAP))
    print("cmc curve")
    for r in ranks:
        print("Rank-{:<3}:{:.1%}".format(r,cmc[r-1]))
    print("--------------------")
    return cmc[0]

def main():
    ####GPU,log file config###
    args = parse_arguments()
    torch.manual_seed(args.seed)
    assert torch.cuda.is_available()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    use_gpu = torch.cuda.is_available()
    sys.stdout= Logger(os.path.join("./", 'log_test_lstm.txt'))

    ###dataset loader###
    dataset = SkDataset()
    query_loader = test_data_iter(dataset.query, args.test_batch)
    gallery_loader = test_data_iter(dataset.gallery, args.test_batch)

    ###model initialize###
    # model = LSTMGait(args.test_batch, args.hidden_size)
    model = RNN(18,2,24,128,128,124)
    model.load_state_dict(torch.load("./log/swj_checkpoint_ep200.pth.tar"))
    if use_gpu:
        model = model.cuda()

    ###test model###
    rank1 = test(model, query_loader, gallery_loader, use_gpu)


if __name__ == '__main__':
    ###main function ###
    try:
        main()
    except KeyboardInterrupt as e:
        print("STOP", e)
