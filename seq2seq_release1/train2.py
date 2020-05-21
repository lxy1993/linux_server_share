import os

import argparse
import sys
from data_prepare import SkDataset
from utils import load_dataset, Logger
import numpy as np
import random
import torch
from torch import optim
from torch import nn
from torch.autograd import Variable
from model import  LSTMGait, RNN
from losses import CrossEntropyLabelSmooth, TripletLoss
from utils import AverageMeter
from torch.optim import lr_scheduler
import pandas as pd
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter


def parse_arguments():
    p = argparse.ArgumentParser(description='Hyperparams')
    p.add_argument('-epochs', type=int, default=5,
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
                   help='initial learning rate')  ##学习速率为0.001时候会loss出现震荡
    p.add_argument('-grad_clip', type=float, default=10.0,
                   help='in case of gradient explosion')  ###预防梯度爆炸
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
    p.add_argument("--test_batch", default=1, type=int, help="has to be 1")
    p.add_argument("--margin", default=0.3, type=float, help="margin for the triplet loss")
    return p.parse_args()


def data_to_tensor(data):
    seq_tracklets = []
    for i in range(len(data)):
        for j in range(len(data[i]) - 1):
            if j == 0:
                data_tracklet = np.expand_dims(np.array(data[i][j]), 0)
                data_tracklet1 = np.expand_dims(np.array(data[i][j + 1]), 0)
                data_tracklets = np.concatenate((data_tracklet, data_tracklet1), 0)
            else:
                data_tracklet1 = np.expand_dims(np.array(data[i][j + 1]), 0)
                data_tracklets = np.concatenate((data_tracklets, data_tracklet1), 0)

        #####abnormal data detect and process#####
        PD = pd.DataFrame(data_tracklets)
        nan_num = PD.isnull().sum()
        if nan_num.sum() > 0:
            print("nan eixist########", nan_num.sum())
            data_tracklets = np.array(PD.fillna(0))
        # print("max",np.max(data_tracklets))
        # print("min",np.min(data_tracklets))
        seq_tracklets.append(torch.from_numpy(data_tracklets).double())
    batch_tracklets = torch.stack(seq_tracklets, 0)  ###torch.stack(list,0),list中每个元素为tensor中第0维度的每个元素
    return batch_tracklets


def data_iter(dataset, batch_size):
    skeleton_all = []
    batches_sk = []
    num_class = len(dataset)
    for i in range(num_class):
        for j in range(len(dataset[i])):
            skeleton_all.append(dataset[i][j])
    random.shuffle(skeleton_all)
    num_iter = len(skeleton_all) // batch_size
    for i in range(num_iter):
        skeletons_cordinate = []
        skeletons_ID = []
        for j in range(batch_size):
            skeleton_cordinate = skeleton_all[j + i * batch_size][0]
            skeleton_ID = skeleton_all[j + i * batch_size][1]
            skeletons_cordinate.append(skeleton_cordinate)
            skeletons_ID.append(skeleton_ID)
            batch_sk = [(tuple(skeletons_cordinate), tuple(skeletons_ID))]
        batches_sk.append(batch_sk)
    return batches_sk

###混淆矩阵绘制###
def show_confusion_matrix(confusion_mat,classes,set_name,epoch,out_dir):
    print("calculate the confusion matrix")
    #归一化
    confusion_mat_N=confusion_mat.copy().astype(float) #math operation (data dytpe convert float)
    for i in range(len(classes)):
        confusion_mat_N[i,:] = confusion_mat[i,:] / confusion_mat[i,:].sum()

    #获取颜色
    cmap = plt.cm.get_cmap("Oranges")
    plt.imshow(confusion_mat_N,cmap=cmap)
    plt.colorbar()

    #设置标题
    xlocations=np.array(range(len(classes)))
    plt.xticks(xlocations,list(classes),rotation=60)
    plt.yticks(xlocations, list(classes))
    plt.xlabel("predict label")
    plt.ylabel("True label")
    plt.title("confusion_matrix_"+set_name)

    #打印数字
    for i in range(confusion_mat_N.shape[0]):
        for j in range(confusion_mat_N.shape[1]):
            plt.text(x=i, y=j, s=float(confusion_mat[i,j]),va="center",ha="center",color="red",fontsize=10)  #设置输出字体属性
    #保存
    plt.savefig(os.path.join(out_dir,"confusion_matrix_"+set_name+str(epoch)+".jpg"))
    plt.close()

def train(model, dataloader, batch_size, optimizer, criterion_xent, criterion_htri):
    model.train()
    train_losses = []
    running_corrects = 0.0
    for batch_idx, data in enumerate(dataloader):
        print("batch_idx############:", batch_idx)
        input, label = data[0][0], data[0][1]
        label = torch.from_numpy(np.array(label, dtype=float)).to(torch.long)
        label = Variable(label.cuda())
        input = data_to_tensor(input)
        input = Variable(input.cuda())  ######这一块需要先将list 转换成tensor
        y, f = model(input)
        _, preds = torch.max(y.data, 1)

        optimizer.zero_grad()
        xent_loss = criterion_xent(y, label)
        htri_loss = criterion_htri(f, label)
        loss = xent_loss + htri_loss
        print("loss", loss)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.data.cpu())
        # losses.update(loss.data,batch_size)  ####这块比较模糊
        running_corrects += float(torch.sum(preds == label.data))

    running_corrects = running_corrects / (len(dataloader) * batch_size)
    return np.mean(train_losses), running_corrects


def train_val(epoch, model, classes, dataloaders, batch_size, optimizer, criterion_xent):
    train_losses = []
    val_losses = []
    acc = {}
    cls_num = 84
    for phase in ["train", "val"]:
        if phase == "train":
            model.train()
        else:
            model.eval()

        running_corrects = 0.0

        for i, data in enumerate(dataloaders[phase]):

            ###get the inputs ,label###
            input, label = data[0][0], data[0][1]
            label = torch.from_numpy(np.array(label, dtype=float)).to(torch.long)
            label = Variable(label.cuda())
            input = Variable(data_to_tensor(input).cuda())

            ###forward,backward,update weights###
            optimizer.zero_grad()
            if phase == "val":
                with torch.no_grad():
                    y, f = model(input)
            else:
                y, f = model(input)
                confusion_matrix = np.zeros((cls_num, cls_num))

            xent_loss = criterion_xent(y, label)
            # htri_loss = criterion_htri(f, label)
            if phase == "train":
                train_loss = xent_loss
                train_loss.backward()
                optimizer.step()
                train_losses.append(train_loss.data.cpu())
            else:
                val_loss = xent_loss
                val_losses.append(val_loss.data.cpu())

            _, preds = torch.max(y.data, 1)
            ###static the prediction info###
            print("phase",phase,"epoch",epoch)
            if phase == "val" and epoch ==5:
                print("val confusion matrix")
                label_cpu =label.data.cpu()
                preds_cpu =preds.data.cpu()
                for j in range(len(label_cpu)):
                    true_i=label_cpu.data[j].numpy()
                    pre_i=preds_cpu[j].numpy()
                    confusion_matrix[true_i,pre_i]+=1.0   #true_i 为行数，pre_i为列数

            running_corrects += float(torch.sum(preds == label.data))
        acc[phase] = running_corrects / (len(dataloaders[phase]) * batch_size)
        print("acc", acc[phase])
        if phase == "val" and epoch ==5:
            show_confusion_matrix(confusion_matrix, classes, phase,epoch, "./")
    return np.mean(train_losses), np.mean(val_losses), acc


def main():
    ###GPU,log file config###
    args = parse_arguments()
    torch.manual_seed(args.seed)
    print(torch.cuda.is_available())
    assert torch.cuda.is_available()
    class_name=[]
    for i in range(84):
        ID ="ID"+str(i+1)
        class_name.append(ID)
    print(class_name)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    use_gpu = torch.cuda.is_available()
    sys.stdout = Logger(os.path.join("./", 'log_train_val_lstm_5.17.txt'))
    # writer = SummaryWriter()
    ###########initilize dataset####
    dataset = SkDataset()
    train_loader = data_iter(dataset.train, args.batch_size)
    val_loader = data_iter(dataset.val, args.batch_size)
    train_val_loader = {"train": train_loader, "val": val_loader}

    # #####initialize the version 1.0 model#####
    # model = LSTMGait(args.batch_size, args.hidden_size)
    #
    # ###loss function and optimizer config###
    # optimizer = optim.Adam(model.parameters(), lr=args.lr,
    #                        weight_decay=args.weight_decay)  # optimizer config and L_2 normalization
    # if args.stepsize > 0:
    #     scheduler = lr_scheduler.StepLR(optimizer, step_size=args.stepsize,
    #                                     gamma=args.gamma)  # learning rate decay strategy
    # criterion_xent = CrossEntropyLabelSmooth(num_classes=124, use_gpu=use_gpu)  # loss function
    # criterion_htri = TripletLoss(margin=args.margin)
    # if use_gpu:
    #     model = nn.DataParallel(model).cuda()

    ###initialize the version 2.0 model ###
    input_size = 18  # 关节点数
    channel_size = 2
    step_size = 24  # time steps 24
    embed_size = 128  # 128
    hidden_size = 128  # 128
    output_size = 84  # classes  #n_class 2020
    model = RNN(input_size, channel_size, step_size, embed_size, hidden_size, output_size)
    print(model)
    lr = 0.2  # 0.2
    device = torch.device('cuda:0')
    model.to(device)
    loss_function = nn.CrossEntropyLoss().to(device)  # .to(device) #
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    if args.stepsize > 0:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=args.gamma)
    criterion_xent = CrossEntropyLabelSmooth(num_classes=84, use_gpu=use_gpu)
    criterion_htri =TripletLoss(margin=args.margin)

    ############train,val per epoch###########
    for epoch in range(args.epochs):
        print("==> Epoch {}/{}".format(epoch, args.epochs))
        train_loss, val_loss, acc = train_val(epoch+1,model,class_name, train_val_loader, args.batch_size, optimizer, criterion_xent)
        print("train", acc["train"], "val_acc", acc["val"])
        # writer.add_scalars("result/loss_5.14", {"train_loss": train_loss, "val_loss": val_loss}, epoch)
        # writer.add_scalars("result/acc_5.14", {"train_acc": acc["train"], "val_acc": acc["val"]}, epoch)
        if args.stepsize > 0: scheduler.step()

        #######save the model checkpoint#########
        if (epoch + 1) == args.epochs:
            print("save the checkpoint")
            # if use_gpu:
            #     state_dict =model.module.state_dict()
            # else:
            #     state_dict =model.state_dict()
            state_dict = model.state_dict()
            torch.save(state_dict, os.path.join(args.save_dir, "5.17_checkpoint_ep" + str(epoch + 1) + ".pth.tar"))


if __name__ == "__main__":
    # try:
    #     main()
    # except KeyboardInterrupt as e:
    #     print("[STOP]", e)
    classes=[]
    for i in range(20):
        ID=str(i)
        classes.append(ID)
    # confusion_matrix =np.ones((10,10))
    confusion_matrix = np.random.randint(2,88,size=(20,20))
    print(confusion_matrix)
    show_confusion_matrix(confusion_matrix,classes,"train","2","./")