import argparse
from src.classifier.semantic_classifier import get_patch_intersect, get_patch_difference, overfitting_2, overfitting_1
from src.utils import read_invariant_all_info
from src.utils import Logger

logger = Logger("log", "run")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--c", type=int, default=0, help="0: semantic, 1: syntactic, 2:combine")     
    parser.add_argument("--T", type=float, default=0.975, help="classification threshold for syntactic classifier")# 0.93 for w/o gt, 0.975 for with gt   
    parser.add_argument("--p_inv_pass_path", type=str, default="data/processed_data/invariants/patches/112/result_passing.txt", help="path to patch invariants pass")
    parser.add_argument("--p_inv_fail_path", type=str, default="data/processed_data/invariants/patches/112/result_failing.txt", help="path to patch invariants fail")
    parser.add_argument("--b_inv_pass_path", type=str, default="data/processed_data/invariants/b/Math/88/result_passing.txt", help="path to bug invariants pass")
    parser.add_argument("--b_inv_fail_path", type=str, default="data/processed_data/invariants/b/Math/88/result_failing.txt", help="path to bug invariants fail")
    parser.add_argument("--f_inv_pass_path", type=str, default="data/processed_data/invariants/f/Math/88/result_passing.txt", help="path to dev invariants pass")
    parser.add_argument("--f_inv_fail_path", type=str, default="data/processed_data/invariants/f/Math/88/result_failing.txt", help="path to dev invariants fail")
    parser.add_argument("--use-z3", action="store_true", help="use z3 to check equivalent otherwise use string equivalent checking")
    return parser.parse_args()

def semantic_check(args):
    patch_invs_P, patch_invs_F, bug_invs_P, bug_invs_F, dev_invs_P, dev_invs_F = read_invariant_all_info(args.p_inv_pass_path, args.p_inv_fail_path, args.b_inv_pass_path, args.b_inv_fail_path, args.f_inv_pass_path, args.f_inv_fail_path, args.use_z3)

    correct_spec = get_patch_intersect(bug_invs_P, dev_invs_P)
    error_beha = get_patch_difference(dev_invs_F, bug_invs_F)
        
    is_overfit2 = overfitting_2(patch_invs_P, correct_spec)

    is_overfit1 = overfitting_1(patch_invs_F, error_beha)
    print(is_overfit1)
    print(is_overfit2)
    if is_overfit1 or is_overfit2:
        return True
    return False

def main():
    args = get_args()
    is_overfitting = False
    is_overfitting = semantic_check(args=args)
    
    if args.c >= 1:
        return NotImplementedError("We currently only support semantic checking")
    
    if is_overfitting:
        print("This patch is overfitting !")
    else:
        print("This patch is not overfitting !")

if __name__ == "__main__":
    main()