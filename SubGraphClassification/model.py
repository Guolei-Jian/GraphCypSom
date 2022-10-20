from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType, ChiralType
import torch
from torch_geometric.data import Dataset, Data, DataLoader
import numpy as np
import os
import networkx as nx
import torch.nn as nn
from torch_geometric.nn import SAGEConv, global_mean_pool, global_add_pool, global_max_pool
from torch.utils.data import random_split
import pickle
import random
from torch_geometric.utils import from_networkx
from sklearn.metrics import roc_auc_score, accuracy_score
import nni
from nni.utils import merge_parameter
import argparse
import logging

logger = logging.getLogger('subgraphclassification')

class SubGraph(Dataset):

    def __init__(self, root, filename, test=False,transform=None, pre_transform=None, pre_filter=None):
        self.filename = filename
        self.test = test
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return self.filename

    @property
    def processed_file_names(self):
        self.raws = pickle.load(open(self.raw_paths[0], 'rb'))
        if self.test:
            return [f'data_test_{i}' for i in range(len(self.raws))]
        else:
            return [f'data_{i}.pt' for i in range(len(self.raws))]

    def download(self):
        pass

    def process(self):
        self.raws = pickle.load(open(self.raw_paths[0], 'rb'))
        for idx, mol in enumerate(self.raws):
            subgraph, label = mol
            # create data object
            data = from_networkx(subgraph)
            label = torch.tensor(label, dtype=torch.int64)
            data['target'] = label
            if self.test:
                torch.save(data, os.path.join(self.processed_dir, \
                f'data_test_{idx}.pt'))
            else:
                torch.save(data, os.path.join(self.processed_dir, \
                f'data_{idx}.pt'))
        
    def len(self):
        return len(self.raws)

    def get(self, idx):
        if self.test:
            data = torch.load(os.path.join(self.processed_dir, f'data_test_{idx}.pt'))
        else:
            data = torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))
        return data

train_dataset = SubGraph('./subgraphdataset/', 'train.pkl')
test_dataset = SubGraph('./subgraphdataset/', 'test.pkl', test=True)
training_set, validation_set  = random_split(train_dataset, [int(len(train_dataset) * 0.8), len(train_dataset) - int(len(train_dataset) * 0.8)], generator=torch.Generator().manual_seed(42))


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        num_classses = 2
        
        tmp = {
            0: global_mean_pool,
            1: global_add_pool,
            2: global_max_pool,
        }
        conv_hidden = args['conv_hidden']
        cls_hidden = args['cls_hidden']
        self.n_layers = args['n_layers']
        cls_drop = args['cls_drop']

        self.conv_layers = nn.ModuleList([])

        self.conv1 = SAGEConv(26, conv_hidden)

        for i in range(self.n_layers):
            self.conv_layers.append(
                SAGEConv(conv_hidden, conv_hidden)
            )

        self.linear1 = nn.Linear(conv_hidden, cls_hidden)
        self.linear2 = nn.Linear(cls_hidden, num_classses)
        self.relu = nn.ReLU()
        self.drop1 = nn.Dropout(p=cls_drop)
        self.readout = tmp[args['readout']]

    def forward(self, mol):

        res = self.conv1(mol.x, mol.edge_index)
        for i in range(self.n_layers):
            res = self.conv_layers[i](res, mol.edge_index)

        res = self.readout(res, mol.batch)
        res = self.linear1(res)
        res = self.relu(res)
        res = self.drop1(res)
        res = self.linear2(res)

        return res

def seed_torch(seed=42):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.

def train(args, model, device, training_set, optimizer, criterion, epoch):
    model.train()
    sf = nn.Softmax(dim=1)
    total_loss = 0
    all_pred = []
    all_pred_raw = []
    all_labels = []
    for sub_mol in training_set:
        sub_mol = sub_mol.to(device)
        sub_mol.x = sub_mol.x.to(torch.float32)
        target = sub_mol.target
        optimizer.zero_grad()
        output= model(sub_mol)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        # tracking
        all_pred.append(np.argmax(output.cpu().detach().numpy(), axis=1))
        all_pred_raw.append(sf(output)[:, 1].cpu().detach().numpy())
        all_labels.append(target.cpu().detach().numpy())
    
    all_pred = np.concatenate(all_pred).ravel()
    all_pred_raw = np.concatenate(all_pred_raw).ravel()
    all_labels = np.concatenate(all_labels).ravel()
    
    logger.info(f'Train Epoch: {epoch}, Ave Loss: {total_loss / len(training_set)} ACC: {accuracy_score(all_labels, all_pred)}  AUC: {roc_auc_score(all_labels, all_pred_raw)}')

def val(args, model, device, val_set, optimizer, criterion, epoch):
    model.eval()
    sf = nn.Softmax(dim=1)
    total_loss = 0
    all_pred = []
    all_pred_raw = []
    all_labels = []
    for sub_mol in val_set:
        sub_mol = sub_mol.to(device)
        sub_mol.x = sub_mol.x.to(torch.float32)
        target = sub_mol.target
        optimizer.zero_grad()
        output= model(sub_mol)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # tracking
        all_pred.append(np.argmax(output.cpu().detach().numpy(), axis=1))
        all_pred_raw.append(sf(output)[:, 1].cpu().detach().numpy())
        all_labels.append(target.cpu().detach().numpy())
    
    all_pred = np.concatenate(all_pred).ravel()
    all_pred_raw = np.concatenate(all_pred_raw).ravel()
    all_labels = np.concatenate(all_labels).ravel()
    logger.info(f'validation Epoch: {epoch}, Ave Loss: {total_loss / len(val_set)} ACC: {accuracy_score(all_labels, all_pred)}  AUC: {roc_auc_score(all_labels, all_pred_raw)}')
    return accuracy_score(all_labels, all_pred)

def test(model, device, test_set):
    model.eval()
    sf = nn.Softmax(dim=1)
    all_pred = []
    all_pred_raw = []
    all_labels = []
    subgraph_num = 0
    with torch.no_grad():
        for sub_mol in test_set:
            sub_mol = sub_mol.to(device)
            sub_mol.x = sub_mol.x.to(torch.float32)
            target = sub_mol.target
            output= model(sub_mol)
            # tracking
            all_pred.append(np.argmax(output.cpu().detach().numpy(), axis=1))
            all_pred_raw.append(sf(output)[:, 1].cpu().detach().numpy())
            all_labels.append(target.cpu().detach().numpy())
    
    all_pred = np.concatenate(all_pred).ravel()
    all_pred_raw = np.concatenate(all_pred_raw).ravel()
    all_labels = np.concatenate(all_labels).ravel()

    print(f'ACC: {accuracy_score(all_labels, all_pred)} AUC: {roc_auc_score(all_labels, all_pred_raw)}')
    return accuracy_score(all_labels, all_pred)

def main(args):
    batch_size = args['batch_size']
    train_loader = DataLoader(training_set, batch_size, shuffle=True)
    val_loader = DataLoader(validation_set, batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
    seed_torch(args['seed'])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Model(args).to(device)
    print(model)
    # weights = torch.tensor([1, args['pos_weight']], dtype=torch.float32).to(device)
    # loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args['lr'])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    max_acc = 0
    for epoch in range(1, args['epoch'] + 1):
        train(args, model, device, train_loader, optimizer, loss_fn, epoch)
        acc = val(args, model, device, val_loader, optimizer, loss_fn, epoch)
        nni.report_intermediate_result(acc)
        scheduler.step()
        if acc > max_acc:
            max_acc = acc
            print('Saving model (epoch = {:4d}, max_acc = {:.4f})'
                .format(epoch, max_acc))
            torch.save(model.state_dict(), args['save_path'])
    # final result
    model.load_state_dict(torch.load(args['save_path']))
    final_acc = test(model, device, test_loader)
    nni.report_final_result(final_acc)

def get_params():
    # Training settings
    parser = argparse.ArgumentParser(description='atombasedmodel')
    parser.add_argument("--conv_hidden", type=int, default=1024, metavar='CH',
                        help='conv hidden size (default: 1024)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--epoch', type=int, default=300, metavar='E',
                        help='number of epochs to train (default: 300)')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--n_layers', type=int, default=2, metavar='NL',
                        help='conv layer num (default: 2)')
    parser.add_argument('--cls_drop', type=float, default=0.3, metavar='D',
                        help='classification dropout (default: 0.3)')
    parser.add_argument("--readout", type=int, default=0, metavar='P',
                        help='select which readout function (default: 0)')
    parser.add_argument('--save_path', type=str, default='./model', metavar='SP',
                        help='save_path (default: ./model)')
    parser.add_argument('--cls_hidden', type=int, default=1024, metavar='H',
                        help='Linear hidden size defaule 1024')
    parser.add_argument('--batch_size', type=int, default=32, metavar='BN',
                        help='batch size (default: 32)')
    args, _ = parser.parse_known_args()
    return args

if __name__ == "__main__":
    try:
        tuner_params = nni.get_next_parameter()
        logger.debug(tuner_params)
        params = vars(merge_parameter(get_params(), tuner_params))
        logger.info(params)
        params['save_path'] = './model/model_' + nni.get_trial_id()
        main(params)
    except Exception as exception:
        logger.exception(exception)
        raise