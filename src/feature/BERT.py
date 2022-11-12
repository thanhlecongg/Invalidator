'''
This implementation is inspired from https://github.com/TruX-DTF/DL4PatchCorrectness
'''

import os
import re
from transformers import BertTokenizer, BertModel
import torch
import pickle
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_embedding = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True) # Whether the model returns all hidden-states
device = torch.device("cpu")
bert_embedding.to(device) 
bert_embedding.eval()
_RAW_DATA_DIR = "data/raw_data/"
_PROCESSED_DATA_DIR = "data/processed_data/"
os.makedirs(_PROCESSED_DATA_DIR, exist_ok= True)
projects_ASE20 = ["CapGen", "Jaid", "SOFix", "SequenceR", "SketchFix"]
DATA_139 = ['Patch1','Patch2','Patch4','Patch5','Patch6','Patch7','Patch8','Patch9','Patch10','Patch11','Patch12','Patch13','Patch14','Patch15','Patch16','Patch17','Patch18','Patch19','Patch20','Patch21','Patch22','Patch23','Patch24','Patch25','Patch26','Patch27','Patch28','Patch29','Patch30','Patch31','Patch32','Patch33','Patch34','Patch36','Patch37','Patch38','Patch44','Patch45','Patch46','Patch47','Patch48','Patch49','Patch51','Patch53','Patch54','Patch55','Patch58','Patch59','Patch62','Patch63','Patch64','Patch65','Patch66','Patch67','Patch68','Patch69','Patch72','Patch73','Patch74','Patch75','Patch76','Patch77','Patch78','Patch79','Patch80','Patch81','Patch82','Patch83','Patch84','Patch88','Patch89','Patch90','Patch91','Patch92','Patch93','Patch150','Patch151','Patch152','Patch153','Patch154','Patch155','Patch157','Patch158','Patch159','Patch160','Patch161','Patch162','Patch163','Patch165','Patch166','Patch167','Patch168','Patch169','Patch170','Patch171','Patch172','Patch173','Patch174','Patch175','Patch176','Patch177','Patch180','Patch181','Patch182','Patch183','Patch184','Patch185','Patch186','Patch187','Patch188','Patch189','Patch191','Patch192','Patch193','Patch194','Patch195','Patch196','Patch197','Patch198','Patch199','Patch201','Patch202','Patch203','Patch204','Patch205','Patch206','Patch207','Patch208','Patch209','Patch210','PatchHDRepair1','PatchHDRepair3','PatchHDRepair4','PatchHDRepair5','PatchHDRepair6','PatchHDRepair7','PatchHDRepair8','PatchHDRepair9','PatchHDRepair10']
_ASE21_INFO_PATH= _RAW_DATA_DIR + "ASE20_Patches/patches.json"
_ICSE18_INFO_PATH = _RAW_DATA_DIR + "ICSE18_Patches/INFO/{}.json"
_PATCH_ICSE18_PATH = _RAW_DATA_DIR + "ICSE18_Patches/{}"
_PATCH_ICSE20_PATH = _RAW_DATA_DIR + "ASE20_Patches/Patches_ICSE/{}/{}/{}/{}.patch"
_PATCH_ASE20_PATH = _RAW_DATA_DIR + "ASE20_Patches/Patches_others/{}/{}/{}/{}.patch"
_DEFECTS4J_CORRECT_PATCH = _RAW_DATA_DIR + "defects4j-developer/"
_TRAIN_DATA = _PROCESSED_DATA_DIR + "train_bert.pkl"
_TEST_DATA = _PROCESSED_DATA_DIR + "test_bert.pkl"


def read_info(dataset, path = None):
    data = {}
    if dataset == "ICSE18":
        list_patches = []
        for i in range(210):
            list_patches.append("Patch"+ str(i+1))
        for i in range(10):
            list_patches.append("PatchHDRepair"+ str(i+1))
        for patch in list_patches:
            with open(_ICSE18_INFO_PATH.format(patch), "r") as f:
                tmp = eval(f.read())
                data[tmp["ID"]] = tmp       
    if dataset == "ASE21":    
        with open(path, "r") as f:
            tmp = eval(f.read())
            for patch in tmp:
                data[str(patch["id"])] = patch

    return data

def read_patch(path, type):
    with open(path, "r") as f:
        code = ''
        p = r"([^\w_])"
        flag = True
        for line in f:
            line = line.strip()
            if '*/' in line:
                flag = True
                continue
            if not flag:
                continue
            if line != '':
                if line.startswith('@@') or line.startswith('diff') or line.startswith('index'):
                    continue
                if line.startswith('Index') or line.startswith('==='):
                        continue
                elif '/*' in line:
                    flag = False
                    continue
                elif type == "buggy":
                    if line.startswith('---') or line.startswith('PATCH_DIFF_ORIG=---'):
                        continue
                    elif line.startswith('-'):
                        if line[1:].strip().startswith('//'):
                            continue
                        line = re.split(pattern=p, string=line[1:].strip())
                        line = [x.strip() for x in line]
                        while '' in line:
                            line.remove('') 
                        line = ' '.join(line)                    
                        code += line.strip() + ' '
                    elif line.startswith('+'):
                        pass
                    else:
                        line = re.split(pattern=p, string=line.strip())
                        line = [x.strip() for x in line]
                        while '' in line:
                            line.remove('')
                        line = ' '.join(line)
                        code += line.strip() + ' '
                
                elif type == 'fixed':
                    if line.startswith('+++'):
                         continue
                    elif line.startswith('+'):
                         if line[1:].strip().startswith('//'):
                             continue
                         line = re.split(pattern=p, string=line[1:].strip())
                         line = [x.strip() for x in line]
                         while '' in line:
                             line.remove('')
                         line = ' '.join(line)
                         code += line.strip() + ' '
                    elif line.startswith('-'):
                         pass
                    else:
                        line = re.split(pattern=p, string=line.strip())
                        line = [x.strip() for x in line]
                        while '' in line:
                             line.remove('')
                        line = ' '.join(line)
                        code += line.strip() + ' '
    if len(code) > 512:
        code = code[:512]
    return code

def embedding(code):
    tokens  = tokenizer.tokenize(code)
    code_ids = tokenizer.convert_tokens_to_ids(tokens)
    segments_ids = [1] * len(code_ids)
    tokens_tensor = torch.tensor([code_ids])
    segments_tensors = torch.tensor([segments_ids])
    tokens_tensor = tokens_tensor.to(device)
    segments_tensors = segments_tensors.to(device)
    outputs = bert_embedding(tokens_tensor, segments_tensors)
    return outputs.last_hidden_state[:, 0, :]

def prepare_test_data():
    label_array, buggy_array, patched_array, gt_array, origin_patch = list(), list(), list(), list(), list()
    count = 0
    deduplicated_data = []
    with open(_TEST_DATA,'wb') as f:
        patches_ICSE = read_info("ICSE18")
        for id in DATA_139: 
            patch_info = patches_ICSE[id]
            label = 0
            project_path = os.path.join(_DEFECTS4J_CORRECT_PATCH, patch_info["project"])
            bug_id = patch_info["bug_id"]
            gt_path = os.path.join(project_path,'patches')
            gt_path = os.path.join(gt_path,'{}.src.patch'.format(bug_id))
            path = _PATCH_ICSE18_PATH.format(patch_info["ID"])     
            if patch_info["correctness"] == "Correct":
                label = 0
            else:
                label = 1
            bug_code = read_patch(path, "buggy")
            fixed_code = read_patch(path, "fixed")
            deduplicated_data.append(fixed_code)
            gt_code = read_patch(gt_path, "buggy")
            buggy_array.append(embedding(bug_code).detach().cpu().numpy()[0])
            patched_array.append(embedding(fixed_code).detach().cpu().numpy()[0])
            gt_array.append(embedding(gt_code).detach().cpu().numpy()[0])
            origin_patch.append("ICSE18_" + str(id))
            label_array.append(label)

        data = label_array, buggy_array, patched_array, gt_array, origin_patch
        pickle.dump(data, f) 
    return deduplicated_data

def prepare_train_data(deduplicated_data):
    data = []
    patches_ASE = read_info("ASE21", _ASE21_INFO_PATH) 
    label_array, buggy_array, patched_array, gt_array = list(), list(), list(), list()
    with open(_TRAIN_DATA,'wb') as f:
        for idx , patch_info in patches_ASE.items():
            label = 0
            if patch_info["correctness"] != "Error" and patch_info["project"] in {"Chart", "Time", "Lang", "Math"}:
                    project_path = os.path.join(_DEFECTS4J_CORRECT_PATCH, patch_info["project"])
                    bug_id = patch_info["bug_id"].split("-")[2]
                    gt_path = os.path.join(project_path,'patches')
                    gt_path = os.path.join(gt_path,'{}.src.patch'.format(bug_id))
                    if patch_info["tool"] in projects_ASE20:
                        path = _PATCH_ASE20_PATH.format(patch_info["correctness"], patch_info["tool"], patch_info["project"], patch_info["bug_id"])
                    else:
                        path = _PATCH_ICSE20_PATH.format(patch_info["correctness"], patch_info["tool"], patch_info["project"], patch_info["bug_id"])
                    
                    if patch_info["correctness"] in ["Ddifferent", "Dcorrect", "Dsame"]:
                        label = 0
                    else:
                        label = 1
                    bug_code = read_patch(path, "buggy")
                    fixed_code = read_patch(path, "fixed")
                    
                    if fixed_code in deduplicated_data:
                        continue

                    deduplicated_data.append(fixed_code)
                    gt_code = read_patch(gt_path, "buggy")
                    buggy_array.append(embedding(bug_code).detach().cpu().numpy()[0])
                    patched_array.append(embedding(fixed_code).detach().cpu().numpy()[0])
                    gt_array.append(embedding(gt_code).detach().cpu().numpy()[0])
                    label_array.append(label)
        print(len(label_array))
        count = 0
        for bug in ["Math", "Chart", "Time", "Lang"]:
            label = 0
            bug_path = os.path.join(_DEFECTS4J_CORRECT_PATCH, bug)
            correct_patches = os.path.join(bug_path,'patches')
            for patch in os.listdir(correct_patches):
                path = correct_patches + "/" + patch
                if not patch.endswith('src.patch'):
                    continue
                try:
                    bug_code = read_patch(path, "fixed")
                    fixed_code = read_patch(path, "buggy")
                except:
                    continue
                
                buggy_array.append(embedding(bug_code).detach().cpu().numpy()[0])
                patched_array.append(embedding(fixed_code).detach().cpu().numpy()[0])
                gt_array.append(embedding(fixed_code).detach().cpu().numpy()[0])
                count += 1
                label_array.append(label)
        data = label_array,buggy_array,patched_array, gt_array
        pickle.dump(data, f)
    return data

if __name__ == "__main__":
    deduplicated_data = prepare_test_data()
    prepare_train_data(deduplicated_data)
