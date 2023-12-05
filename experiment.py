import argparse
import pickle
from tkinter import E
from src.classifier.syntactic_classifier import get_features, n_val, evaluation_metrics
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.metrics import confusion_matrix
import os 
from src.utils import Logger, read_idx2id, read_info_patch, read_invariant
from src.classifier.semantic_classifier import get_patch_intersect, get_patch_difference, overfitting_2, overfitting_1
import time

_PROCESSED_DATA_DIR = "data/processed_data/"
_RAW_DATA = "data/raw_data/"
_TRAIN_DATA = _PROCESSED_DATA_DIR + "train.pkl"
_TEST_DATA = _PROCESSED_DATA_DIR + "test.pkl"
_SHUFFLE_IDXS = _PROCESSED_DATA_DIR + "shuffle_ids.txt"
_MODEL_PATH = "model/model.joblib"

_INVARIANTS = _PROCESSED_DATA_DIR + "invariants/"
_PATCH_INFO = _RAW_DATA + "patch_info/139_patches.json"
_INFO_PATH = _RAW_DATA + "patch_info/PS_INFO/{}.json"

logger = Logger("log", "eval")

def get_syntactic_preds():
    #Loading data
    with open(_TRAIN_DATA, 'rb') as input:
        data = pickle.load(input)

    label, buggy, patched, gt = data
    buggy = np.array(buggy)
    patched = np.array(patched)
    idxs = []
    with open(_SHUFFLE_IDXS, "r") as f:
            for i in f:
                if len(i) > 0:
                    idxs.append(int(i))
    correct_id = np.array([746-i-1 for i in range(223)])
    idxs = np.append(idxs, correct_id)
    train_data = get_features(buggy, patched, gt)[idxs[n_val:]]
    
    with open(_TEST_DATA, 'rb') as input:
        data = pickle.load(input)
    label_t, buggy_t, patched_t, gt_t, origin_patch = data
    buggy_t = np.array(buggy_t)
    patched_t = np.array(patched_t)
    gt_t = np.array(gt_t)
    test_data = get_features(buggy_t, patched_t, gt_t)
    
    #Prediction
    scaler = StandardScaler()
    scaler.fit_transform(train_data)

    x_test, y_test = test_data, label_t

    clf = joblib.load(_MODEL_PATH)
    y_pred = clf.predict_proba(x_test)[:, 1]

    acc, precision, recall, f1, auc = evaluation_metrics(y_true=y_test, y_pred_prob=y_pred)

    logger.log('Accuracy: %f -- Precision: %f -- Recall: %f -- F1: %f -- AUC: %f' % (acc, precision, recall, f1, auc))

    predictions = {}
    for i in range(len(origin_patch)):
        predictions[origin_patch[i].split("_")[1]] = y_pred[i]
    return predictions

def syntax_check(origin_patch, syntax_preds, threshold):
    return syntax_preds[origin_patch] > threshold

def semantic_check(project, bug_id, patch):
    patch_invs_P, patch_invs_F, bug_invs_P, bug_invs_F, dev_invs_P, dev_invs_F = read_invariant(_INVARIANTS, project, bug_id, patch, use_z3=True)

    correct_spec = get_patch_intersect(bug_invs_P, dev_invs_P)
    error_beha = get_patch_difference(dev_invs_F, bug_invs_F)
        
    is_overfit2 = overfitting_2(patch_invs_P, correct_spec)

    is_overfit1 = overfitting_1(patch_invs_F, error_beha)
    
    if is_overfit1 or is_overfit2:
        return True
    return False


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--c", type=int, default=2, help="0: semantic, 1: syntactic, 2:combine")     
    parser.add_argument("--T", type=float, default=0.975, help="classification threshold for syntactic classifier")# 0.93 for w/o gt, 0.975 for with gt   
    return parser.parse_args()

def main():
    st_time = time.time()
    args = get_args()
    syntax_preds = get_syntactic_preds()
        
    list_patches = os.listdir(_INVARIANTS + "patches")
    idx2id = read_idx2id(_PATCH_INFO)

    true_overfit = 0
    false_overfit = 0
    total_correct = 0
    total_incorrect = 0
    list_correct = []
    list_incorrect = []
    miss_correctness = []
    rs4project_tp = {"Chart": 0, "Time": 0, "Lang": 0, "Math": 0}
    rs4project_fp = {"Chart": 0, "Time": 0, "Lang": 0, "Math": 0}
    total_correct_perproject = {"Chart": 0, "Time": 0, "Lang": 0, "Math": 0}
    total_incorrect_perproject = {"Chart": 0, "Time": 0, "Lang": 0, "Math": 0}
    
    for patch in list_patches:
        if patch == ".DS_Store" or patch == "incorrect":
            continue
        origin_patch = idx2id[int(patch)]
        project, bug_id, correctness, tool = read_info_patch(_INFO_PATH, origin_patch)
        logger.log(f"[{origin_patch}]: {project}-{bug_id}-{tool} ==> Label({correctness})")
        
        is_overfitting = False
        if args.c == 0:
            is_overfitting = semantic_check(project, bug_id, patch)
        
        if args.c == 1:
            is_overfitting = syntax_check(origin_patch, syntax_preds, args.T)

        if args.c == 2:
            is_overfitting = semantic_check(project, bug_id, patch)
            if not is_overfitting:
                is_overfitting = syntax_check(origin_patch, syntax_preds, args.T)
            
        
        if correctness == "Incorrect":
                total_incorrect += 1
                total_incorrect_perproject[project] += 1
        elif correctness == "Correct" or correctness == "Same":
                total_correct += 1
                total_correct_perproject[project] += 1
        else:
                miss_correctness.append(correctness)
        
        if is_overfitting:
            print("Patch: ", origin_patch, " Correctness: ", correctness)
            if correctness == "Incorrect":
                rs4project_tp[project] += 1
                list_incorrect.append(origin_patch)
                true_overfit += 1
            elif correctness == "Correct" or correctness == "Same":
                rs4project_fp[project] += 1
                list_correct.append(origin_patch)
                false_overfit += 1
    print("====================================")
    print("True overfitting: {}/{}".format(true_overfit, total_incorrect))
    print(list_incorrect)
    print("False overfitting: {}/{}".format(false_overfit, total_correct))
    print(list_correct)

    print("===========METRICS==============")
    print("Recall: {}/{} = {}".format(true_overfit, total_incorrect, true_overfit/total_incorrect))
    print("Precision: {}/{} = {}".format(false_overfit, true_overfit + false_overfit, true_overfit/(true_overfit + false_overfit)))
    print("Accuracy: {} ".format((true_overfit + total_correct - false_overfit)/(total_correct + total_incorrect)))
    precision, recall = true_overfit/(true_overfit + false_overfit), true_overfit/total_incorrect
    print("F1: {} ".format((2*precision*recall/(precision+recall))))
    print(list_correct)
    # print("Total time: {}".format(time.time()-st_time))

if __name__ == "__main__":
    main()
