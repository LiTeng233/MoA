import sys
sys.path.append('../input/iterative-stratification/iterative-stratification-master/')

import numpy as np
import random
import pandas as pd
import os
from matplotlib import pyplot as plt 
%matplotlib inline
import seaborn as sns
sns.set_style('ticks')
sns.set_context("poster")
sns.set_palette('colorblind')
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer
from sklearn.feature_selection import VarianceThreshold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.modules.loss import _WeightedLoss
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')
# os.listdir('../input/lish-moa')


plt.rcParams['figure.figsize'] = (20.0, 10.0)


device = ('cuda' if torch.cuda.is_available() else 'cpu')


params = {'device': device,
          'n_comp_g': 450, 
          'n_comp_c': 45, 
          'var_thresh': 0.67,
          'epochs': 25,
          'batch_size': 128,
          'lr': 1e-3,
          'weight_decay': 1e-5, 
          'n_folds': 5, 
          'early_stopping_steps': 10,
          'early_stop': False,
          'in_size': None,
          'out_size': None,
          'hidden_size': 1500}


train_features = pd.read_csv('../input/lish-moa/train_features.csv') # ../input/lish-moa/
train_targets = pd.read_csv('../input/lish-moa/train_targets_scored.csv') # ../input/lish-moa/

test_features = pd.read_csv('../input/lish-moa/test_features.csv') # ../input/lish-moa/
sample_submission = pd.read_csv('../input/lish-moa/sample_submission.csv') # ../input/lish-moa/


train_features.shape

test_features.shape

sample_submission.shape

g_features = [col for col in train_features.columns if col.startswith('g-')]
c_features = [col for col in train_features.columns if col.startswith('c-')]

g_c_features = g_features + c_features


transformer = QuantileTransformer(n_quantiles=100,random_state=0, output_distribution="normal")


trans_train_features = transformer.fit_transform(train_features[g_c_features])
trans_test_features = transformer.transform(test_features[g_c_features])

trans_train_df = pd.DataFrame(trans_train_features, columns = g_c_features)
trans_test_df = pd.DataFrame(trans_test_features, columns = g_c_features)

train_features = pd.concat([train_features.drop(columns=g_c_features), trans_train_df], axis=1)
test_features = pd.concat([test_features.drop(columns=g_c_features), trans_test_df], axis=1)


g_sample = random.sample(g_features, 3)
c_sample = random.sample(c_features, 3)


colors = ['navy', 'r', 'g']
for col, color in zip(g_sample, colors):
    plt.hist(test_features[col], bins=50, alpha=0.5, label=col)
    plt.axvline(np.median(test_features[col]), linewidth=3, color=color, label='median_{}'.format(col))
plt.xlim(-7, 7)
plt.legend();


colors = ['navy', 'r', 'g']
for col, color in zip(c_sample, colors):
    plt.hist(test_features[col], bins=50, alpha=0.5, label=col)
    plt.axvline(np.median(test_features[col]), linewidth=3, color=color, label='median_{}'.format(col))
plt.xlim(-7, 7)
plt.legend();


def transfrom_all_data(transformer, train, test, feature_list):
    
    data = pd.concat([train[feature_list], test[feature_list]], axis=0).reset_index(drop=True)
    n = train.shape[0]
    
    data_trans = transformer.fit_transform(data)
    train_trans = data_trans[:n, :]
    test_trans = data_trans[n:, :]
    return train_trans, test_trans


def make_pca_features(n_comp, train, test, feature_list, name, normalize=False, scaler=None):
    
    pca = PCA(n_comp)
    
    train_pca, test_pca = transfrom_all_data(pca, train, test, feature_list)
    
    if normalize and scaler is not None:
        train_pca = scaler.fit_transform(train_pca)
        test_pca = scaler.transform(test_pca)
    
    for i in range(n_comp):
        train['{0}_{1}'.format(name, i)] = train_pca[:, i]
        test['{0}_{1}'.format(name, i)] = test_pca[:, i]
        
    return train, test


def preprocess(data):
    data['cp_time'] = data['cp_time'].map({24:0, 48:1, 72:2})
    data['cp_dose'] = data['cp_dose'].map({'D1':0, 'D2':1})
    return data


train_features, test_features = make_pca_features(params['n_comp_g'], train_features, test_features, g_features, 'g_pca')

train_features, test_features = make_pca_features(params['n_comp_c'], train_features, test_features, c_features, 'c_pca')


var_thresh = VarianceThreshold(params['var_thresh'])
to_thresh = train_features.columns[4:]
cat_features = train_features.columns[:4]


train_thresh, test_thresh = transfrom_all_data(var_thresh, train_features, test_features, to_thresh)

train_features = pd.concat([train_features[cat_features], pd.DataFrame(train_thresh)], axis=1)
test_features = pd.concat([test_features[cat_features], pd.DataFrame(test_thresh)], axis=1)


train_features.shape

test_features.shape


train_mask = train_features['cp_type'] != 'ctl_vehicle'
train_sig_ids = train_features.loc[train_mask]['sig_id']
train = train_features.loc[train_mask].reset_index(drop=True)

test_mask = test_features['cp_type'] != 'ctl_vehicle'
test_sig_ids = test_features.loc[test_mask]['sig_id']
test = test_features.loc[test_mask].reset_index(drop=True)

train_target_sigids = train_targets[['sig_id']]
y_true  = train_targets.copy()

train_targets = train_targets[train_targets['sig_id'].isin(train_sig_ids)].reset_index(drop=True)
train_targets.drop(columns=['sig_id'], inplace=True)
train_targets.reset_index(drop=True, inplace=True)

train.shape

test.shape

y_true.shape

train_targets.shape

train_target_sigids

params['in_size'] = train.shape[1] - 2
params['out_size'] = train_targets.shape[1]

params['out_size']

params['in_size']

train.head()


#Cross validation split
mskf = MultilabelStratifiedKFold(n_splits=params['n_folds'])

folds = train.copy()

for f, (t_idx, v_idx) in enumerate(mskf.split(X=train, y=train_targets)):
    folds.loc[v_idx, 'kfold'] = int(f)

folds['kfold'] = folds['kfold'].astype(int)

folds.head()


#Modal
class TabularDataset:
    
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return(self.X.shape[0])
    
    def __getitem__(self, i):
        
        X_i = torch.tensor(self.X[i, :], dtype=torch.float)
        y_i = torch.tensor(self.y[i, :], dtype=torch.float)
        
        return X_i, y_i
    
    

class TabularDatasetTest:
    
    def __init__(self, X):
        self.X = X
    
    def __len__(self):
        return(self.X.shape[0])
    
    def __getitem__(self, i):
        
        X_i = torch.tensor(self.X[i, :], dtype=torch.float)        
        return X_i
    
    
def train_func(model, optimizer, scheduler, loss_func, dataloader, device):
    
    train_loss = 0
    
    model.train()  
    for inputs, labels in dataloader:        
        optimizer.zero_grad()
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        train_loss += loss.item()
        
    train_loss /= len(dataloader)
    
    return train_loss


def valid_func(model, loss_func, dataloader, device):
    
    model.eval()
    
    valid_loss = 0
    valid_preds = []
    
    for inputs, labels in dataloader:   
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = loss_func(outputs, labels)
        
        valid_loss += loss.item()
        valid_preds.append(outputs.sigmoid().detach().cpu().numpy())
        
    valid_loss /= len(dataloader)
    valid_preds = np.concatenate(valid_preds)
    
    return valid_loss, valid_preds


def inference_fn(model, dataloader, device):
    model.eval()
    preds = []
    
    for data in dataloader:
        inputs = data.to(device)

        with torch.no_grad():
            outputs = model(inputs)
        
        preds.append(outputs.sigmoid().detach().cpu().numpy())
        
    preds = np.concatenate(preds)
    
    return preds


class SmoothBCEwLogits(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth(targets:torch.Tensor, n_labels:int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = targets * (1.0 - smoothing) + 0.5 * smoothing
        return targets

    def forward(self, inputs, targets):
        targets = SmoothBCEwLogits._smooth(targets, inputs.size(-1),
            self.smoothing)
        loss = F.binary_cross_entropy_with_logits(inputs, targets,self.weight)

        if  self.reduction == 'sum':
            loss = loss.sum()
        elif  self.reduction == 'mean':
            loss = loss.mean()

        return loss
    
    
class Model(nn.Module):
    def __init__(self, num_features, num_targets, hidden_size):
        super(Model, self).__init__()
        self.batch_norm1 = nn.BatchNorm1d(num_features)
        self.dense1 = nn.utils.weight_norm(nn.Linear(num_features, hidden_size))
        
        self.batch_norm2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(0.25)
        self.dense2 = nn.utils.weight_norm(nn.Linear(hidden_size, hidden_size))
        
        self.batch_norm3 = nn.BatchNorm1d(hidden_size)
        self.dropout3 = nn.Dropout(0.25)
        self.dense3 = nn.utils.weight_norm(nn.Linear(hidden_size, num_targets))
    
    def forward(self, x):
        x = self.batch_norm1(x)
        x = F.leaky_relu(self.dense1(x))
        
        x = self.batch_norm2(x)
        x = self.dropout2(x)
        x = F.leaky_relu(self.dense2(x))
        
        x = self.batch_norm3(x)
        x = self.dropout3(x)
        x = self.dense3(x)
        
        return x
    
    
def run_training(fold, seed):
    
    seed_everything(seed)
    
    train = preprocess(folds.drop(columns = ['sig_id', 'cp_type']))
    
    train_mask = train['kfold'] != fold
    valid_idc = train.loc[~train_mask].index
    
    X_train = train.loc[train_mask].reset_index(drop=True)
    y_train = train_targets.loc[train_mask].reset_index(drop=True)

    
    X_val = train.loc[~train_mask].reset_index(drop=True)
    y_val = train_targets.loc[~train_mask].reset_index(drop=True)
    
    X_train.drop(columns=['kfold'], inplace=True)
    X_val.drop(columns=['kfold'], inplace=True)
    
    test_ = preprocess(test.drop(columns = ['sig_id', 'cp_type']))

    
    train_ds = TabularDataset(X_train.values, y_train.values)
    valid_ds = TabularDataset(X_val.values, y_val.values)
    test_ds = TabularDatasetTest(test_.values)
    
    train_dl = DataLoader(train_ds, batch_size=params['batch_size'], shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=params['batch_size'], shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=params['batch_size'], shuffle=False)
    
    
    model = Model(num_features=params['in_size'], num_targets=params['out_size'], 
                  hidden_size=params['hidden_size'] )
    
    model.to(params['device'])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=1e3, 
                                              max_lr=1e-2, epochs=params['epochs'], steps_per_epoch=len(train_dl))
    
    loss_fn = nn.BCEWithLogitsLoss()
    loss_tr = SmoothBCEwLogits(smoothing=0.001)
    
    early_stopping_steps = params['early_stopping_steps']
    early_step = 0
   
    oof = np.zeros((train.shape[0], params['out_size']))
    best_loss = np.inf
    
    for epoch in range(params['epochs']):
        
        train_loss = train_func(model, optimizer,scheduler, loss_tr, train_dl, params['device'])
        print(f"FOLD: {fold}, EPOCH: {epoch}, train_loss: {train_loss}")
        valid_loss, valid_preds = valid_func(model, loss_fn, valid_dl, params['device'])
        print(f"FOLD: {fold}, EPOCH: {epoch}, valid_loss: {valid_loss}")
        
        if valid_loss < best_loss:
            
            best_loss = valid_loss
            oof[valid_idc] = valid_preds
            torch.save(model.state_dict(), f"FOLD{fold}_seed{seed}.pth")
            #torch.save(model.state_dict(), f"FOLD{fold}_.pth")
        elif(params['early_stop'] == True):
            
            early_step += 1
            if (early_step >= early_stopping_steps):
                break
            
    
    #--------------------- PREDICTION---------------------

    
    model = Model(num_features=params['in_size'], num_targets=params['out_size'], 
                  hidden_size=params['hidden_size'] )
    model.load_state_dict(torch.load(f"FOLD{fold}_seed{seed}.pth"))
    #model.load_state_dict(torch.load(f"FOLD{fold}_.pth"))
    model.to(params['device'])
    
    
    predictions = np.zeros((test.shape[0], params['out_size']))
    predictions = inference_fn(model, test_dl, params['device'])
    
    return oof, predictions


def run_k_fold(n_folds, seed):
    oof = np.zeros((train.shape[0], params['out_size']))
    predictions = np.zeros((test.shape[0], params['out_size']))
    
    for fold in range(n_folds):
        oof_, pred_ = run_training(fold, seed)
        
        predictions += pred_ / n_folds
        oof += oof_
        
    return oof, predictions


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_everything(seed=42)


# Averaging on multiple SEEDS

seeds = [0, 1, 2, 3, 4, 5, 6]
oof = np.zeros((train.shape[0], params['out_size']))
predictions = np.zeros((test.shape[0], params['out_size']))

for seed in seeds:
    
    oof_, predictions_ = run_k_fold(params['n_folds'], seed)
    oof += oof_ / len(seeds)
    predictions += predictions_ / len(seeds)
    
    
valid_results = pd.concat([train_target_sigids[train_target_sigids['sig_id'].isin(train_sig_ids)].reset_index(drop=True), pd.DataFrame(oof)], axis=1)

test_results = pd.concat([test[['sig_id']], pd.DataFrame(predictions, columns = sample_submission.columns[1:])], axis=1)

valid_full = train_target_sigids.merge(valid_results, on='sig_id', how='left').fillna(0)

y_true = y_true.drop(columns=['sig_id']).values
y_pred = valid_full.drop(columns=['sig_id']).values

score = 0
for i in range(y_true.shape[1]):
    score_ = log_loss(y_true[:, i], y_pred[:, i])
    score += score_ / y_true.shape[1]
    
print("CV log_loss: ", score)    

sub = sample_submission[['sig_id']].merge(test_results, on='sig_id', how='left').fillna(0)
sub.to_csv('submission.csv', index=False)

