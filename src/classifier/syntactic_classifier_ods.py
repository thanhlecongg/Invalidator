'''
This implementation is inspired from https://github.com/TruX-DTF/DL4PatchCorrectness
'''
import os
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score
from sklearn import utils
from sklearn.metrics import roc_curve, auc, accuracy_score, recall_score
from sklearn.preprocessing import StandardScaler
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import joblib
from random import *
from sklearn.metrics.pairwise import *
from sklearn.metrics import confusion_matrix
_PROCESSED_DATA_DIR = "data/processed_data/"
_RAW_DATA_DIR = "data/raw_data/"

_TRAIN_DATA = _PROCESSED_DATA_DIR + "train_ods.pkl"
_TEST_DATA = _PROCESSED_DATA_DIR + "test_ods.pkl"
_SHUFFLE_IDXS = _PROCESSED_DATA_DIR + "shuffle_ids_ods.txt"
_MODEL_PATH = "model/model_ods.joblib"

seed = 0
n_val = 74 # 10% x total_training_data(737) = 74

np.random.seed(seed)

def evaluation_metrics(y_true, y_pred_prob, threshold=0.5, is_eval = False):
    fpr, tpr, thresholds = roc_curve(y_true=y_true, y_score=y_pred_prob, pos_label=1)
    auc_ = auc(fpr, tpr)
    y_pred = [1 if p >= threshold else 0 for p in y_pred_prob]
    
    # ODS cannot generate features for PatchHDRepair5 --> we assume ODS consider the patch as correct
    if is_eval:
        y_true.append(1)
        y_pred.append(0)
    
    print('real positive: {}, real negative: {}'.format(list(y_true).count(1),list(y_true).count(0)))
    print('positive: {}, negative: {}'.format(y_pred.count(1),y_pred.count(0)))
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    prc = precision_score(y_true=y_true, y_pred=y_pred)
    rc = recall_score(y_true=y_true, y_pred=y_pred)
    f1 = 2 * prc * rc / (prc + rc)
    print('Accuracy: %f -- Precision: %f -- Recall: %f -- F1: %f -- AUC: %f' % (acc, prc, rc, f1, auc_))
    
    return acc, prc, rc, f1, auc_

def train(train_data, labels, val_data, labels_val):
    scaler = StandardScaler()
    scaler.fit_transform(train_data)

    x_train, y_train = train_data, labels

    x_test, y_test = val_data, labels_val

    clf = LogisticRegression(solver='lbfgs', max_iter=10000).fit(X=x_train, y=y_train)   
    y_pred = clf.predict_proba(x_test)[:, 1]
    joblib.dump(clf, _MODEL_PATH)
    print('[Threshold Tuning]')
    for i in range(1, 200):
        y_pred_tn = [1 if p >= i/200.0 else 0 for p in y_pred]
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_tn).ravel()
        recall = tp/(tp+fn)
        precision = tp/(tp + fp)
        f1 = 2 * recall * precision/(recall + precision)
        print('[Threshold:{}]'.format(i/200))
        print('===> TP: %d -- TN: %d -- FP: %d -- FN: %d F1 %f' % (tp, tn, fp, fn, f1))
    
    acc, precision, recall, f1, auc = evaluation_metrics(y_true=y_test, y_pred_prob=y_pred)

    print('Accuracy: %f -- Precision: %f -- Recall: %f -- F1: %f -- AUC: %f' % (acc, precision, recall, f1, auc))

def do_train():
    with open(_TRAIN_DATA, 'rb') as input:
        data = pickle.load(input)

    label, bug_correct_diffs, bug_patch_diffs = data
    if os.path.exists(_SHUFFLE_IDXS):
        idxs = []
        with open(_SHUFFLE_IDXS, "r") as f:
            for i in f:
                if len(i) > 0:
                    idxs.append(int(i))
    else:
        idxs = [i for i in range(len(label)-223)]
        idxs = utils.shuffle(idxs, random_state= seed)
        with open(_SHUFFLE_IDXS, "w") as f:
            for i in idxs:
                f.write(f"{i}\n")
    
    correct_id = np.array([737-i-1 for i in range(221)])
    idxs = np.append(idxs, correct_id)
    train_label = np.array(label)[idxs[n_val:]]
    val_label = np.array(label)[idxs[:n_val]]
    train_data = np.hstack((bug_correct_diffs, bug_patch_diffs))[idxs[n_val:]]
    val_data = np.hstack((bug_correct_diffs, bug_patch_diffs))[idxs[:n_val]]

    train(train_data=train_data, labels=train_label, val_data=val_data, labels_val=val_label)

def do_eval(threshold = 0.955):
    #Loading data
    with open(_TRAIN_DATA, 'rb') as input:
        data = pickle.load(input)

    label, bug_correct_diffs, bug_patch_diffs = data

    idxs = []
    with open(_SHUFFLE_IDXS, "r") as f:
            for i in f:
                if len(i) > 0:
                    idxs.append(int(i))
    correct_id = np.array([737-i-1 for i in range(221)])
    idxs = np.append(idxs, correct_id)
    train_data = np.hstack((bug_correct_diffs, bug_patch_diffs))[idxs[n_val:]]
    
    with open(_TEST_DATA, 'rb') as input:
        data = pickle.load(input)
    
    label_t, bug_correct_diffs_t, bug_patch_diffs_t, origin_patch = data
    test_data = np.hstack((bug_correct_diffs_t, bug_patch_diffs_t))

    scaler = StandardScaler()
    scaler.fit_transform(train_data)

    x_test, y_test = test_data, label_t
    
    clf = joblib.load(_MODEL_PATH)
    y_pred = clf.predict_proba(x_test)[:, 1]
    
    acc, precision, recall, f1, auc = evaluation_metrics(y_true=y_test, y_pred_prob=y_pred, threshold=threshold, is_eval=True)
    
if __name__ == '__main__':
    print("=" * 100)
    print("Training...")
    print("=" * 100)
    do_train()
    print("=" * 100)
    print("Evaluating...")
    print("=" * 100)
    do_eval(0.955)