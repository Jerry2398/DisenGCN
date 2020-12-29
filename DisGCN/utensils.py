import numpy as np
import torch
import torch.nn.functional as F
from sklearn import preprocessing
from sklearn.metrics import accuracy_score

def load(file_path,dataset):
    feature_labels_data = np.genfromtxt("{}/{}/cora.content".format(file_path,dataset),dtype=np.dtype(str))
    fetaures = np.array(feature_labels_data[:, 1:-1], dtype=np.float32)
    labels = np.array(feature_labels_data[:,-1], dtype=str).reshape(-1,1)
    encoder = preprocessing.OneHotEncoder(handle_unknown='ignore')
    labels = encoder.fit_transform(labels).toarray()
    label_number = len(labels[0])
    labels = np.argwhere(labels==1)[:,1]
    labels = labels.squeeze()

    paper_id = np.array(feature_labels_data[:,0], dtype=int)
    idx_map = {i:j for j,i in enumerate(paper_id)}
    edges = np.genfromtxt("{}/{}/cora.cites".format(file_path,dataset),dtype=int)
    source_id = edges[:,0]
    target_id = edges[:,1]
    source_index = [idx_map[i] for i in source_id]
    target_index = [idx_map[i] for i in target_id]

    graph = np.zeros([len(paper_id),len(paper_id)],dtype=int)
    graph[source_index,target_index] = 1
    graph[target_index,source_index] = 1

    idx_train = range(1500)
    idx_val = range(1500, 2000)
    idx_test = range(2000, 2500)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    adj_graph = torch.FloatTensor(graph)
    feature_tensor = torch.FloatTensor(fetaures)
    labels_tensor = torch.Tensor(labels).long()

    return feature_tensor, adj_graph, labels_tensor, idx_train, idx_val, idx_test


def acc(pre_targets, targets):
    pre_index = torch.argmax(pre_targets,dim=1)
    accuracy = torch.sum(pre_index==targets)
    return accuracy/len(targets)