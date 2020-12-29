import torch.optim as optim
from utensils import *
from DisGCN import *
import time
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', action='store_true', default=False, help="use cuda or not")
parser.add_argument('--dataset', action='store', default='Cora', help="use cuda or not")
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--epochs', action='store', default=200, type=int, help='number of epochs')
parser.add_argument('--lr', action='store', default=0.001, type=float, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dropout', action='store', default=0.1, type=float, help='dropout')
parser.add_argument('--beta', type=float, default=1, help='beta in DisConv')
parser.add_argument('--iterations', type=int, default=7, help='iterations of each DisConv')
parser.add_argument('--out_dim', type=int, default=7, help='class numbers')
args = parser.parse_args()
args.layer_num = 2
args.channels = [8, 4]
args.c_dims = [8, 4]
args.in_dims = []

if args.dataset == 'Cora':
    args.in_dims.append(1433)
    for i in range(args.layer_num-1):
        args.in_dims.append(args.channels[i] * args.c_dims[i])

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

features, adj, labels, idx_train, idx_val, idx_test = load("data","cora")

model = DisGCN(args.in_dims, args.channels, args.c_dims, args.iterations, args.beta, args.layer_num, args.dropout,
               args.out_dim)
opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()


def train_model(epoch):
    start_time = time.time()
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    opt.zero_grad()
    y = model(adj, features)
    loss = loss_fn(y[idx_train], labels[idx_train])
    train_acc = acc(y[idx_train], labels[idx_train])
    loss.backward()
    opt.step()

    val_acc = acc(y[idx_val], labels[idx_val])
    # print('in epoch: {}'.format(epoch))
    # print('loss is {}'.format(loss.item()))
    # print("train acc is {}".format(train_acc))
    # print("val acc is {}".format(val_acc))
    # print("training time is {}".format(time.time() - start_time))
    return train_acc, val_acc, loss.item(), time.time() - start_time


def val_model():
    model.eval()
    y = model(adj, features)
    test_acc = acc(y[idx_test], labels[idx_test])
    print("test acc is {}".format(test_acc))

x = []
loss_list = []
train_acc_list = []
val_acc_list = []
epoch_time_list = []
for epoch in range(args.epochs):
    train_acc, val_acc, loss, epoch_time = train_model(epoch)
    x.append(epoch)
    train_acc_list.append(train_acc)
    val_acc_list.append(val_acc)
    loss_list.append(loss)
    epoch_time_list.append(epoch_time)

plt.figure()
plt.subplot(311)
plt.plot(x, loss_list, c='b', label="loss")
plt.subplot(312)
plt.plot(x, train_acc_list, c='g', label="train acc")
plt.plot(x, val_acc_list, c='b', label="val acc")
plt.subplot(313)
plt.plot(x, epoch_time_list, c='g', label="epoch time")
plt.savefig("results/result_fig.png")
torch.save(model, "DisenGCN_model.pt")
val_model()

