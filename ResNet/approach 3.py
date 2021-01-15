import os
import random
import time
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_selection import VarianceThreshold
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.metrics import fbeta_score
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
import tensorflow.keras.backend as K
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
from xgboost import XGBClassifier 


#import sys
#sys.path += ['/input/iterstrat', '/input/tabnet']
import sys
sys.path.append('../input/iterative-stratification/iterative-stratification-master/iterstrat')
import sys
sys.path.append('../input/tf-TabNet-master/tabnet')

from ml_stratifiers import MultilabelStratifiedKFold
from tabnet.stacked_tabnet import StackedTabNetClassifier


def set_random_seeds(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
set_random_seeds(43)


X_full = pd.read_csv('../input/lish-moa/train_features.csv')
Y_scored = pd.read_csv('../input/lish-moa/train_targets_scored.csv')
Y_nonscored = pd.read_csv('../input/lish-moa/train_targets_nonscored.csv')
drug = pd.read_csv('../input/lish-moa/train_drug.csv')

X_test_full = pd.read_csv('../input/lish-moa/test_features.csv')
sig_id = X_test_full['sig_id'].values.reshape((-1, 1))


cat_cols = ['cp_type', 'cp_time', 'cp_dose']
numerical_cols = [c for c in X_full.columns if c not in ['sig_id'] + cat_cols]
label_cols = [c for c in Y_scored.columns if c != 'sig_id']

num_features = X_full.shape[1] - 1
num_labels = Y_scored.shape[1] - 1
num_nonscored_labels = Y_nonscored.shape[1] - 1


def logloss(Y_true, Y_preds, label_smoothing=0, remove_unpredicted=False):
    if remove_unpredicted:
        predicted = (Y_preds.sum(axis=1) != 0)
        Y_true = Y_true.loc[predicted, :]
        Y_preds = Y_preds.loc[predicted, :]
    Y_true = Y_true.astype(np.float32)
    return tf.reduce_mean(keras.losses.binary_crossentropy(Y_true, Y_preds, label_smoothing=label_smoothing)).numpy()


def metrics(Y_true, Y_preds):
    pr = tf.keras.metrics.Precision()
    pr.update_state(Y_true, Y_preds)
    re = tf.keras.metrics.Recall()
    re.update_state(Y_true, Y_preds)

    pr = pr.result().numpy()
    re = re.result().numpy()
    f1 = 2 * (pr * re) / (pr + re)
    
    auc = tf.keras.metrics.AUC(curve='PR', multi_label=True)
    auc.update_state(Y_true, Y_preds)
    return pr, re, f1, auc.result().numpy()


def balance_class_weights(Y_true, class_weights=(0.7, 1.6)):
    n_labels = Y_true.shape[1]
    weights = np.empty([n_labels, 2])
    for i in range(n_labels):
        if class_weights == 'balanced':
            total = len(Y_true[:, i])
            pos = np.sum(Y_true[:, i])
            neg = total - pos
            weight_for_0 = (1 / neg)*(total)/2.0 
            weight_for_1 = (1 / pos)*(total)/2.0
            weights[i] = [weight_for_0, weight_for_1]
        else:
            weights[i] = np.array(class_weights)
    return weights


def weighted_binary_crossentropy(weights, label_smoothing=0):
    def weighted_loss(Y_true, Y_pred):
        bce = tf.keras.losses.BinaryCrossentropy(label_smoothing=label_smoothing)
        return K.mean(
            (weights[:, 0]**(1 - Y_true)) * (weights[:, 1]**(Y_true)) * bce(Y_true, Y_pred),
            axis=-1
        )
    return weighted_loss


def clear_session():
    curr_session = tf.compat.v1.get_default_session()
    if curr_session is not None:
        curr_session.close()
    K.clear_session()
    
    s = tf.compat.v1.InteractiveSession()
    tf.compat.v1.keras.backend.set_session(s)
    
    
def preprocess(df, targets=None):
    df = df.copy()
    df.loc[:, 'cp_type'] = df.loc[:, 'cp_type'].map({'trt_cp': 0, 'ctl_vehicle': 1})
    df.loc[:, 'cp_dose'] = df.loc[:, 'cp_dose'].map({'D1': 0, 'D2': 1})
    df = df.drop(columns='sig_id')
    
    if targets is not None:
        targets = targets.copy()
        targets = targets.drop(columns='sig_id')
        targets = targets[df['cp_type'] == 0].reset_index(drop=True)

    df = df[df['cp_type'] == 0].reset_index(drop=True)
    
    if targets is None:
        return df
    return df, targets


def complete_train_labels(Y, train_idx, test_idx):
    """ Make sure the train set has at least 1 positive class for every label """
    train_idx, test_idx = list(train_idx), list(test_idx)
    Y_train = Y.iloc[train_idx, :]
    
    train_pos_cnts = Y_train.sum(axis=0)
    missing_labels = train_pos_cnts[train_pos_cnts == 0].index
    removed_from_test = []
    
    for l in missing_labels:
        for i in test_idx:
            if Y[l].iloc[i] == 1:
                train_idx.append(i)
                removed_from_test.append(i)
                break
    test_idx = [i for i in test_idx if i not in removed_from_test]
    return train_idx, test_idx


X_train, Y_train = preprocess(X_full, Y_scored)
X_test = preprocess(X_test_full)
Y_train_nonscored = Y_nonscored.drop(columns='sig_id')[X_full['cp_type'] == 'trt_cp'].reset_index(drop=True)
initial_bias = -np.log(Y_train.mean(axis=0).values)
nonscored_initial_bias = -np.log(Y_train_nonscored.mean(axis=0).values + 1e-6)


def multilabel_split_by_drugs(scored, n_folds, seed):
    scored = scored.copy().merge(drug, on='sig_id', how='left') 

    # LOCATE DRUGS
    vc = scored.drug_id.value_counts()
    vc1 = vc.loc[vc <= 18].index
    vc2 = vc.loc[vc > 18].index

    # STRATIFY DRUGS 18X OR LESS
    dct1 = {}
    dct2 = {}
    skf = MultilabelStratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    tmp = scored.groupby('drug_id')[label_cols].mean().loc[vc1]
    for fold, (train_idx, test_idx) in enumerate(skf.split(tmp, tmp[label_cols])):
        dd = {k:fold for k in tmp.index[test_idx].values}
        dct1.update(dd)

    # STRATIFY DRUGS MORE THAN 18X
    skf = MultilabelStratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    tmp = scored.loc[scored.drug_id.isin(vc2)].reset_index(drop=True)
    for fold, (train_idx, test_idx) in enumerate(skf.split(tmp, tmp[label_cols])):
        dd = {k:fold for k in tmp.sig_id[test_idx].values}
        dct2.update(dd)
        
  # ASSIGN FOLDS
    scored['fold'] = scored.drug_id.map(dct1)
    scored.loc[scored.fold.isna(), 'fold'] = scored.loc[scored.fold.isna(), 'sig_id'].map(dct2)
    scored.fold = scored.fold.astype('int8')
    
    test_idx = [np.where(scored['fold'] == fold)[0].tolist() for fold in range(n_folds)]
    train_idx = [np.where(scored['fold'] != fold)[0].tolist() for fold in range(n_folds)]
    return zip(train_idx, test_idx)


def select_features(X):
    return X.iloc[:, [
        1,   2,   3,   4,   5,   6,   7,   9,  11,  14,  15,  16,  17,
        18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  29,  30,  31,
        32,  33,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,  46,
        47,  48,  49,  50,  51,  52,  53,  54,  55,  56,  58,  59,  60,
        61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,  72,  73,
        74,  75,  76,  78,  79,  80,  81,  82,  83,  84,  86,  87,  88,
        89,  90,  91,  92,  93,  94,  95,  96,  97,  98,  99, 100, 101,
       102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114,
       115, 116, 117, 118, 120, 121, 122, 123, 124, 125, 126, 127, 128,
       129, 130, 131, 132, 133, 136, 137, 138, 139, 140, 141, 142, 143,
       144, 145, 146, 147, 149, 150, 151, 152, 153, 154, 155, 156, 157,
       158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170,
       171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183,
       184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 197,
       198, 199, 200, 202, 203, 204, 205, 206, 208, 209, 210, 211, 212,
       213, 214, 215, 216, 217, 218, 219, 220, 221, 223, 224, 225, 226,
       227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239,
       240, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253,
       254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266,
       267, 268, 269, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280,
       281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 294,
       295, 296, 298, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309,
       310, 311, 312, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323,
       324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336,
       337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349,
       350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362,
       363, 364, 365, 366, 367, 368, 369, 370, 371, 374, 375, 376, 377,
       378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 390, 391,
       392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404,
       405, 406, 407, 408, 409, 411, 412, 413, 414, 415, 416, 417, 418,
       419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431,
       432, 434, 435, 436, 437, 438, 439, 440, 442, 443, 444, 445, 446,
       447, 448, 449, 450, 453, 454, 456, 457, 458, 459, 460, 461, 462,
       463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475,
       476, 477, 478, 479, 481, 482, 483, 484, 485, 486, 487, 488, 489,
       490, 491, 492, 493, 494, 495, 496, 498, 500, 501, 502, 503, 505,
       506, 507, 509, 510, 511, 512, 513, 514, 515, 518, 519, 520, 521,
       522, 523, 524, 525, 526, 527, 528, 530, 531, 532, 534, 535, 536,
       538, 539, 540, 541, 542, 543, 544, 545, 546, 547, 549, 550, 551,
       552, 554, 557, 559, 560, 561, 562, 565, 566, 567, 568, 569, 570,
       571, 572, 573, 574, 575, 577, 578, 580, 581, 582, 583, 584, 585,
       586, 587, 588, 589, 590, 591, 592, 593, 594, 595, 596, 597, 599,
       600, 601, 602, 606, 607, 608, 609, 611, 612, 613, 615, 616, 617,
       618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630,
       631, 632, 633, 634, 635, 636, 637, 638, 639, 641, 642, 643, 644,
       645, 646, 647, 648, 649, 650, 651, 652, 654, 655, 656, 658, 659,
       660, 661, 662, 663, 664, 665, 666, 667, 668, 669, 670, 671, 672,
       673, 674, 675, 676, 677, 678, 679, 680, 681, 682, 683, 684, 685,
       686, 687, 688, 689, 691, 692, 693, 694, 695, 696, 697, 699, 700,
       701, 702, 704, 705, 707, 708, 709, 710, 711, 713, 714, 716, 717,
       718, 720, 721, 723, 724, 725, 726, 727, 728, 729, 730, 731, 732,
       733, 734, 735, 737, 738, 739, 740, 742, 743, 744, 745, 746, 747,
       748, 749, 750, 751, 752, 753, 754, 755, 756, 757, 759, 760, 761,
       762, 763, 764, 765, 766, 767, 768, 769, 770, 771, 772, 773, 774,
       775, 776, 777, 779, 780, 781, 782, 783, 784, 785, 786, 787, 788,
       789, 790, 792, 793, 794, 795, 796, 797, 798, 800, 801, 802, 803,
       804, 805, 806, 808, 809, 811, 813, 814, 815, 816, 817, 818, 819,
       821, 822, 823, 825, 826, 827, 828, 829, 830, 831, 832, 834, 835,
       837, 838, 839, 840, 841, 842, 845, 846, 847, 848, 850, 851, 852,
       854, 855, 856, 858, 859, 860, 861, 862, 864, 866, 867, 868, 869,
       870, 871, 872, 873, 874
    ]].values


gene_cols = [c for c in X_train.columns if c.startswith('g-')]
cell_cols = [c for c in X_train.columns if c.startswith('c-')]


import joblib

qtransform = joblib.load('../input/qtransform/qtransform.joblib')

X_train_selected = select_features(X_train)
X_test_selected = select_features(X_test)
X_train_qtrans = np.concatenate([X_train_selected[:, :2], qtransform.transform(X_train_selected[:, 2:])], axis=1)
X_test_qtrans =  np.concatenate([X_test_selected[:, :2], qtransform.transform(X_test_selected[:, 2:])], axis=1)

pca_qtrans_2 = joblib.load('../input/qtransform/pca_qtrans.joblib')


std_scaler_2 = joblib.load('../input/qtransform/std_scaler_2.joblib')
gene_pca_2 = joblib.load('../input/qtransform/gene_pca_2.joblib')
cell_pca_2 = joblib.load('../input/qtransform/cell_pca_2.joblib')


from sklearn.cluster import KMeans

kmeans_g = joblib.load('../input/qtransform/kmeans_g.joblib')
kmeans_c = joblib.load('../input/qtransform/kmeans_c.joblib')

#ResNet
LOAD_PRETRAINED_RESNET = True
REUSE_SPLITS_RESNET = True

def preprocess_resnet(X):
    X_scaled = pd.DataFrame(std_scaler_2.transform(X[numerical_cols]), columns=numerical_cols)
    X_genes = gene_pca_2.transform(X_scaled[gene_cols])
    X_cells = cell_pca_2.transform(X_scaled[cell_cols])
    return select_features(X), np.concatenate([X_genes, X_cells], axis=1)


X_train_resnet_1, X_train_resnet_2 = preprocess_resnet(X_train)
X_test_resnet_1, X_test_resnet_2 = preprocess_resnet(X_test)


class MultiLabelResNet(BaseEstimator):

    def __init__(self, n_input_1, n_input_2, n_output_labels):
        self.n_input_1 = n_input_1
        self.n_input_2 = n_input_2
        self.n_output_labels = n_output_labels
        self.estimator_ = None
        
    def _create_model(self, initial_bias, loss_fn=None):
        output_bias = keras.initializers.Constant(initial_bias)
        
        inp_1 = keras.layers.Input(self.n_input_1, name='raw_inp')
        inp_2 = keras.layers.Input(self.n_input_2, name='pca_inp')
        gauss_2 = keras.layers.GaussianNoise(1e-4)(inp_2)

        head_1 = keras.models.Sequential([
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.2),
            tfa.layers.WeightNormalization(keras.layers.Dense(512, activation='elu')),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.5),
            tfa.layers.WeightNormalization(keras.layers.Dense(256, activation='elu'))
        ], name='head1') 

        seq_1 = head_1(inp_1)
        seq_1_inp_concat = keras.layers.Concatenate()([gauss_2, seq_1])

        head_2 = keras.models.Sequential([
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.4),
            tfa.layers.WeightNormalization(keras.layers.Dense(512, "relu")),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.5),
            tfa.layers.WeightNormalization(keras.layers.Dense(512, "elu")),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.5),
            tfa.layers.WeightNormalization(keras.layers.Dense(256, "relu")),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.1),
            tfa.layers.WeightNormalization(keras.layers.Dense(256, "elu"))
        ], name='head2')

        seq_2 = head_2(seq_1_inp_concat)
        seq_1_seq_2_avg = keras.layers.Average()([seq_1, seq_2]) 
        
        head_3 = keras.models.Sequential([
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.1),
            tfa.layers.WeightNormalization(keras.layers.Dense(256, activation='relu')),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.1),
            tfa.layers.WeightNormalization(keras.layers.Dense(206, activation='relu')),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.1),
            tfa.layers.WeightNormalization(keras.layers.Dense(self.n_output_labels, bias_initializer=output_bias, activation='sigmoid'))
        ], name='head3')

        output = head_3(seq_1_seq_2_avg)
        
        model = keras.models.Model(inputs=[inp_1, inp_2], outputs=output)
        model.compile(
            loss=keras.losses.BinaryCrossentropy(label_smoothing=2e-3) if loss_fn is None else loss_fn,
            optimizer=tfa.optimizers.AdamW(learning_rate=0.02, weight_decay=1e-5),
            metrics=['binary_crossentropy'],
        )
        return model

    def fit(self, X, Y, X_valid, Y_valid, repeat, fold, max_epochs=100, save_weights=True, initial_bias=initial_bias, transfer_model=None):
        Y = tf.cast(Y, tf.float32)
        Y_valid = tf.cast(Y_valid, tf.float32)

        class_weights = balance_class_weights(Y.numpy(), class_weights=(0.5, 2.0))
        loss_fn = weighted_binary_crossentropy(class_weights, label_smoothing=2e-3)
        self.estimator_ = self._create_model(initial_bias, loss_fn=loss_fn)
        if transfer_model is not None:
            self.copy_weights(transfer_model)

        learning_rate_cb = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            patience=3, verbose=1, factor=0.1, min_lr=1e-8
        )
        early_stop_cb = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            min_delta=0.0001, patience=15,
            restore_best_weights=False
        )
        checkpoint_cb = keras.callbacks.ModelCheckpoint(
            f'model_resnet/weights_{repeat}_{fold}.h5',
            monitor='val_binary_crossentropy',
            save_best_only=True,
            save_weights_only=True
        )
        callbacks = [early_stop_cb, learning_rate_cb, checkpoint_cb] if save_weights else [early_stop_cb, learning_rate_cb]

        history = self.estimator_.fit(
            X, Y,
            batch_size=128,
            epochs=max_epochs,
            validation_data=(X_valid, Y_valid),
            callbacks=callbacks,
            shuffle=True,
            verbose=0
        )
        return history
    
    def predict(self, X):
        return self.estimator_.predict(X)
    
    def score(self, X, Y_true):
        Y_preds = self.predict(X)
        return logloss(Y_true, Y_preds)
    
    def save_weights(self, repeat, fold):
        self.estimator_.save_weights('model_resnet/weights_{}_{}.h5'.format(repeat, fold))
        
    def load_weights(self, repeat, fold, pretrained=True):
        if self.estimator_ is None:
            self.estimator_ = self._create_model(initial_bias)
        if pretrained:
            base_path = '../input/model-resnet/'
        else:
            base_path = './'
        self.estimator_.load_weights(base_path + 'model_resnet/weights_{}_{}.h5'.format(repeat, fold))
 
    def copy_weights(self, from_model):
        for to_layer, from_layer in zip(self.estimator_.layers[:-1], from_model.estimator_.layers[:-1]):
            to_layer.set_weights(from_layer.get_weights())
        for to_layer, from_layer in zip(self.estimator_.layers[-1].layers[:-1], from_model.estimator_.layers[-1].layers[:-1]):
            to_layer.set_weights(from_layer.get_weights())
            
n_repeats = 7
n_folds = 5
kfold_seeds = [188, 526, 1045, 453, 200, 8684, 87352] if REUSE_SPLITS_RESNET else np.random.randint(42, 1337, n_repeats)
kfold_seeds = kfold_seeds[:n_repeats]
assert len(kfold_seeds) == n_repeats
print(kfold_seeds)


val_losses = []
histories = []
Y_train_preds_resnet = pd.DataFrame(np.zeros((X_train.shape[0], num_labels)), columns=label_cols)
Y_test_preds_resnet = pd.DataFrame(np.zeros((X_test_full.shape[0], num_labels)), columns=label_cols)

set_random_seeds(123)

for repeat, kf_seed in enumerate(kfold_seeds):
    kf = MultilabelStratifiedKFold(n_splits=n_folds, random_state=kf_seed, shuffle=True)

    for fold, (train_idx, test_idx) in enumerate(kf.split(X_train, Y_train)):
        print('** Repeat {}/{}. Fold {}/{}'.format(repeat + 1, n_repeats, fold + 1, n_folds))
        K.clear_session()
        train_idx, test_idx = complete_train_labels(Y_train, train_idx, test_idx)
        train_idx = tf.random.shuffle(train_idx)
        X_train_1_fold = tf.gather(X_train_resnet_1, train_idx, axis=0)
        X_train_2_fold = tf.gather(X_train_resnet_2, train_idx, axis=0)
        Y_train_fold = tf.gather(Y_train, train_idx, axis=0)
        X_valid_1_fold = tf.gather(X_train_resnet_1, test_idx, axis=0)
        X_valid_2_fold = tf.gather(X_train_resnet_2, test_idx, axis=0)
        Y_valid_fold = tf.gather(Y_train, test_idx, axis=0)
        Y_nonscored_train_fold = tf.gather(Y_train_nonscored, train_idx, axis=0)
        Y_nonscored_valid_fold = tf.gather(Y_train_nonscored, test_idx, axis=0)

        model_resnet = MultiLabelResNet(
            n_input_1=X_train_resnet_1.shape[1],
            n_input_2=X_train_resnet_2.shape[1],
            n_output_labels=Y_train.shape[1]
        )
        if LOAD_PRETRAINED_RESNET:
            model_resnet.load_weights(repeat, fold)
            val_loss = model_resnet.score([X_valid_1_fold, X_valid_2_fold], Y_valid_fold.numpy())
        else:
            print('Pretrain model')
            transfer_model = MultiLabelResNet(
                n_input_1=X_train_resnet_1.shape[1],
                n_input_2=X_train_resnet_2.shape[1],
                n_output_labels=Y_train_nonscored.shape[1]
            )
            transfer_model.fit(
                [X_train_1_fold, X_train_2_fold], Y_nonscored_train_fold,
                [X_valid_1_fold, X_valid_2_fold], Y_nonscored_valid_fold,
                repeat, fold,
                max_epochs=10,
                save_weights=False,
                initial_bias=nonscored_initial_bias
            )

            print('Train model')
            history = model_resnet.fit(
                [X_train_1_fold, X_train_2_fold], Y_train_fold,
                [X_valid_1_fold, X_valid_2_fold], Y_valid_fold,
                repeat, fold,
                transfer_model=transfer_model
            ) 
            histories.append(history)
            val_loss = min(history.history['val_binary_crossentropy'])
            model_resnet.load_weights(repeat, fold, pretrained=False)
            del transfer_model

        val_preds = model_resnet.predict([X_valid_1_fold, X_valid_2_fold])
        add_outputs(Y_train_preds_resnet, val_preds, idx_list=test_idx)

        test_preds = model_resnet.predict([X_test_resnet_1, X_test_resnet_2])
        add_outputs(Y_test_preds_resnet, test_preds, X=X_test_full)

        val_losses.append(val_loss)
        print('Val loss: {}'.format(val_loss))

        del model_resnet
    
    print('----------')
    print('Repeat avg val loss: {}'.format(np.mean(val_losses[repeat*n_folds:((repeat + 1)*n_folds)])))
    print('Repeat OOF val loss: {}'.format(logloss(Y_train, Y_train_preds_resnet / (repeat + 1), remove_unpredicted=True)))
    print('----------')

Y_train_preds_resnet /= n_repeats       
Y_test_preds_resnet.loc[X_test_full['cp_type'] == 'trt_cp', :] /= (n_folds * n_repeats)
print('==========')
print('Overall avg val loss: {}'.format(np.mean(val_losses)))
print('Overall OOF val loss: {}'.format(logloss(Y_train, Y_train_preds_resnet, remove_unpredicted=True)))


def plot_history(history):
    df = pd.DataFrame({
        'loss': history.history['binary_crossentropy'],
        'val_loss': history.history['val_binary_crossentropy'],
    })
    print(df['val_loss'].min())
    df = df.loc[1:, :]
    df.plot()
    plt.axvline(x=df['val_loss'].idxmin(), color='r')
    







