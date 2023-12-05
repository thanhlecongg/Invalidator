import logging
import os
from z3 import Bool, Real, Implies, Not, And, unsat, Solver

class Logger(object):
    def __init__(self, log_dir, log_name):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        log_path = os.path.join(log_dir, f"{log_name}.txt")
        print(log_path)
        if os.path.exists(log_path):
            os.remove(log_path)
        logging.basicConfig(filename= log_path, level=logging.INFO)

    def log(self, content):
        logging.info(content)
        print(content)
        
def read_idx2id(path):
    idx2id = {}
    with open(path, "r") as f:
            data = eval(f.read())
            for item in data:
                idx2id[item["id"]] = item["patch_file"]
    return idx2id

def read_info_patch(_INFO_PATH, patch):
    path = _INFO_PATH.format(patch)
    try:
        with open(path, "r") as f:
            data = eval(f.read())
            return data["project"], data["bug_id"], data["correctness"], data[
                "tool"]
    except FileNotFoundError:
        print(path)
        print(
            "Invalid patch !!! \nPlease use available patch: \n==> Patch1, ..., Patch210; \n==> HDRepair1, ..., HDRepair10"
        )  

def read_invariant_all_info(p_inv_pass_path, p_inv_fail_path, b_inv_pass_path, b_inv_fail_path, f_inv_pass_path, f_inv_fail_path, use_z3):
    p_inv_pass = read_invariant_with_path(p_inv_pass_path, use_z3)
    p_inv_fail = read_invariant_with_path(p_inv_fail_path, use_z3)
    b_inv_pass = read_invariant_with_path(b_inv_pass_path, use_z3)
    b_inv_fail = read_invariant_with_path(b_inv_fail_path, use_z3)
    f_inv_pass = read_invariant_with_path(f_inv_pass_path, use_z3)
    f_inv_fail = read_invariant_with_path(f_inv_fail_path, use_z3)
   
    return p_inv_pass, p_inv_fail, b_inv_pass, b_inv_fail, f_inv_pass, f_inv_fail
    
def read_invariant(_INVARIANT_PATH, project, bug_id, patch, use_z3):
    _INVARIANT_PATCH_PASS = _INVARIANT_PATH + "patches/{}/result_passing.txt"
    _INVARIANT_BUG_PASS   = _INVARIANT_PATH + "b/{}/{}/result_passing.txt"
    _INVARIANT_FIX_PASS   = _INVARIANT_PATH + "f/{}/{}/result_passing.txt"
    _INVARIANT_PATCH_FAIL = _INVARIANT_PATH + "patches/{}/result_failing.txt"
    _INVARIANT_BUG_FAIL   = _INVARIANT_PATH + "b/{}/{}/result_failing.txt"
    _INVARIANT_FIX_FAIL   = _INVARIANT_PATH + "f/{}/{}/result_failing.txt"
    p_inv_pass = read_invariant_with_path(
        _INVARIANT_PATCH_PASS.format(patch), use_z3)
    p_inv_fail = read_invariant_with_path(
        _INVARIANT_PATCH_FAIL.format(patch), use_z3)
    b_inv_pass = read_invariant_with_path(
        _INVARIANT_BUG_PASS.format(project, bug_id), use_z3)
    b_inv_fail = read_invariant_with_path(
        _INVARIANT_BUG_FAIL.format(project, bug_id), use_z3)
    f_inv_pass = read_invariant_with_path(
        _INVARIANT_FIX_PASS.format(project, bug_id), use_z3)
    f_inv_fail = read_invariant_with_path(
        _INVARIANT_FIX_FAIL.format(project, bug_id), use_z3)
    return p_inv_pass, p_inv_fail, b_inv_pass, b_inv_fail, f_inv_pass, f_inv_fail

def read_invariant_with_path(path, use_z3):
    data = {}
    if os.path.exists(path):
        with open(path, "r") as f:
            current_key = 0
            last_key = 0
            exit_count = 0
            is_start = False
            for line in f:
                if line[0:3] == "===":
                    is_start = True
                    current_key = f.readline().strip('\n')
                    if "EXIT" in current_key:
                        if last_key != current_key:
                            exit_count = 0
                        current_key = get_method_name_with_exit(current_key)
                        last_key = current_key
                        if exit_count == 0:
                            data[current_key] = []
                        exit_count += 1
                    else:
                        last_key = current_key
                        data[current_key] = []
                        exit_count = 0
                else:
                    if is_start:
                        data[current_key].append(line.strip())
    if use_z3:
        return format_dict(data)
    else:
        return data

def get_method_name_with_exit(string):
    index = 0
    for i in range(len(string)):
        if string[i] == ":":
            index = i
    return string[:index + 5]

def is_equiv(X, Y):
    s = Solver()
    s.add(Not(X == Y))
    r = s.check()
    if r == unsat:
        return True
    return False

def check_set_equiv(A, B):
    return sorted(A) == sorted(B)

def format_dict(inv_dict):
    formatted_dict = {}
    for key, value in inv_dict.items():
        formatted_dict[key] = []
        for inv in value:
            formatted_dict[key].append(format_invariant(inv))
    return formatted_dict

def format_invariant(invariant):
        '''Get type of invariant'''
        if " ==> " in invariant:
            return ImplyInvariant(invariant)
        elif " <==> " in invariant:
            return EquivInvariant(invariant)
        if " == " in invariant:
            return EqualInvariant(invariant)
        elif " != " in invariant:
            return NotEqualInvariant(invariant)
        elif " >= " in invariant or " <= " in invariant:
            return LessGreaterEqualInvariant(invariant)
        elif " > " in invariant or " < " in invariant:
            return LessGreaterInvariant(invariant)
        elif "one of" in invariant:
            return OneOfInvariant(invariant)
        elif "subset of" in invariant:
            return SubsetOfInvariant(invariant)
        elif "has values:" in invariant:
            return CompleteOneOfInvariant(invariant)
        elif "sorted by" in invariant or "elements are equal":
            return EltwiseComparisionInvariant(invariant)
        elif " in " in invariant:
            return MemberInvariant(invariant)
        elif "no duplicates" in invariant:
            return NoDuplicatesInvariant(invariant)
        else:
            raise("unsupported invariant: {}".format(invariant))
              
class Invariant(object):
    def __init__(self, original_invariant):
        self.origin = original_invariant
    def __str__(self) -> str:
        return self.origin

class EqualInvariant(Invariant):
    def __init__(self, original_invariant):
        super().__init__(original_invariant)
        
        splitted = original_invariant.split("==")
        
        try:
            assert len(splitted) == 2
        except AssertionError:
            print(original_invariant)
            exit()
        
        l = Real(splitted[0].strip())
        try:
            if 'Infinity' not in splitted[1].strip():
                r = float(splitted[1].strip())
            else:
                r = Real(splitted[1].strip())
        except ValueError:
            r = Real(splitted[1].strip())
        
        self.vars = [splitted[0].strip(), splitted[1].strip()]
        self.expr = And(l == r)
        
    
    def __str__(self) -> str:
         return "[EqualInvariant: " + super().__str__() + "]"
    
    def __eq__(self, other):
        if (isinstance(other, EqualInvariant)):
            if check_set_equiv(self.vars, other.vars):
                return is_equiv(self.expr, other.expr)
        return False
    
class NotEqualInvariant(Invariant):
    def __init__(self, original_invariant):
        super().__init__(original_invariant)
        
        splitted = original_invariant.split("!=")
        try:
            assert len(splitted) == 2
        except AssertionError:
            print(original_invariant)
            exit()
        
        l = Real(splitted[0].strip())
        try:
            r = float(splitted[1].strip())
        except ValueError:
            r = Real(splitted[1].strip())
        
        self.vars = [splitted[0].strip(), splitted[1].strip()]
        self.expr = And(l != r)
        
    def __str__(self) -> str:
         return "[NotEqualInvariant: " + super().__str__() + "]"
     
    def __eq__(self, other):
        if (isinstance(other, NotEqualInvariant)):
            if check_set_equiv(self.vars, other.vars):
                return is_equiv(self.expr, other.expr)
        return False
    
class LessGreaterEqualInvariant(Invariant):
    def __init__(self, original_invariant):
        super().__init__(original_invariant)
        if ">=" in original_invariant:
            self.op = ">="
        else:
            self.op = "<="
        
        splitted = original_invariant.split(self.op)
        try:
            assert len(splitted) == 2
        except AssertionError:
            print(original_invariant)
            exit()
        l = Real(splitted[0].strip())
        try:
            r = float(splitted[1].strip())
        except ValueError:
            r = Real(splitted[1].strip())
        
        self.vars = [splitted[0].strip(), splitted[1].strip()]
        if self.op == ">=":
            self.expr = And(l >= r)
        else:
            self.expr = And(l <= r)
        
    def __str__(self) -> str:
         return "[LessGreaterEqualInvariant: " + super().__str__() + "]"
     
    def __eq__(self, other):
        if (isinstance(other, LessGreaterEqualInvariant)):
            if check_set_equiv(self.vars, other.vars):
                return is_equiv(self.expr, other.expr)
        return False
    
class LessGreaterInvariant(Invariant):
    def __init__(self, original_invariant):
        super().__init__(original_invariant)
        if ">" in original_invariant:
            self.op = ">"
        else:
            self.op = "<"
        
        splitted = original_invariant.split(self.op)
        try:
            assert len(splitted) == 2
        except AssertionError:
            print(original_invariant)
            exit()
        l = Real(splitted[0].strip())
        try:
            r = float(splitted[1].strip())
        except ValueError:
            r = Real(splitted[1].strip())
        
        self.vars = [splitted[0].strip(), splitted[1].strip()]
        if self.op == ">":
            self.expr = And(l >= r)
        else:
            self.expr = And(l <= r)
            
    def __str__(self) -> str:
         return "[LessGreaterInvariant: " + super().__str__() + "]"
    
    def __eq__(self, other):
        if (isinstance(other, LessGreaterInvariant)):
            if check_set_equiv(self.vars, other.vars):
                return is_equiv(self.expr, other.expr)
        return False
    
class ImplyInvariant(Invariant):
    def __init__(self, original_invariant):
        super().__init__(original_invariant)
        
        splitted = original_invariant.split("==>")
        try:
            assert len(splitted) == 2
        except AssertionError:
            print(original_invariant)
            exit()
        
        l = splitted[0].strip()
        r = splitted[1].strip()
        
        cmp_ops = [" != ", " == ", " >= ", " <= ", " < ", " > "]
        if any(op in l for op in cmp_ops):
            l = format_invariant(l).expr
        else:
            l = Bool(l)
        
        if any(op in r for op in cmp_ops):
            r = format_invariant(r).expr
        else:
            r = Bool(r)
            
        self.vars = [splitted[0].strip(), splitted[1].strip()]
        self.expr = Implies(l, r)
        
    def __str__(self) -> str:
         return "[ImplyInvariant: " + super().__str__() + "]"
    
    def __eq__(self, other):
        if (isinstance(other, ImplyInvariant)):
            if check_set_equiv(self.vars, other.vars):
                return is_equiv(self.expr, other.expr)
        return False
    
class EquivInvariant(Invariant):
    def __init__(self, original_invariant):
        super().__init__(original_invariant)
        
        splitted = original_invariant.split("<==>")
        
        try:
            assert len(splitted) == 2
        except AssertionError:
            print(original_invariant)
            exit()
        
        l = splitted[0].strip()
        r = splitted[1].strip()
        
        cmp_ops = [" != ", " == ", " >= ", " <= ", " < ", " > "]
        if any(op in l for op in cmp_ops):
            l = format_invariant(l).expr
        else:
            l = Bool(l)
        
        if any(op in r for op in cmp_ops):
            r = format_invariant(r).expr
        else:
            r = Bool(r)
        
        self.vars = [splitted[0].strip(), splitted[1].strip()]
        self.expr = (l == r)
        
    def __str__(self) -> str:
         return "[EquivInvariant: " + super().__str__() + "]"
     
    def __eq__(self, other):
        if (isinstance(other, EquivInvariant)):
            if check_set_equiv(self.vars, other.vars):
                return is_equiv(self.expr, other.expr)
        return False
    
class OneOfInvariant(Invariant):
    def __init__(self, original_invariant):
        super().__init__(original_invariant)
    def __str__(self) -> str:
         return "[OneOfInvariant: " + super().__str__() + "]"
    def __eq__(self, other):
        if (isinstance(other, OneOfInvariant)):
            return self.origin == other.origin
        return False
    
class CompleteOneOfInvariant(Invariant):
    def __init__(self, original_invariant):
        super().__init__(original_invariant)
    def __str__(self) -> str:
         return "[CompleteOneOfInvariant: " + super().__str__() + "]"
    def __eq__(self, other):
        if (isinstance(other, CompleteOneOfInvariant)):
            return self.origin == other.origin
        return False
    
class SubsetOfInvariant(Invariant):
    def __init__(self, original_invariant):
        super().__init__(original_invariant)
    def __str__(self) -> str:
         return "[SubsetOfInvariant: " + super().__str__() + "]"
    def __eq__(self, other):
        if (isinstance(other, SubsetOfInvariant)):
            return self.origin == other.origin
        return False
    
class EltwiseComparisionInvariant(Invariant):
    def __init__(self, original_invariant):
        super().__init__(original_invariant)
    def __str__(self) -> str:
         return "[EltwiseComparisionInvariant: " + super().__str__() + "]"
    def __eq__(self, other):
        if (isinstance(other, EltwiseComparisionInvariant)):
            return self.origin == other.origin
        return False

class MemberInvariant(Invariant):
    def __init__(self, original_invariant):
        super().__init__(original_invariant)
    def __str__(self) -> str:
         return "[MemberInvariant: " + super().__str__() + "]"
    def __eq__(self, other):
        if (isinstance(other, MemberInvariant)):
            return self.origin == other.origin
        return False
    
class NoDuplicatesInvariant(Invariant):
    def __init__(self, original_invariant):
        super().__init__(original_invariant)
    def __str__(self) -> str:
         return "[NoDuplicatesInvariant: " + super().__str__() + "]"
    def __eq__(self, other):
        if (isinstance(other, NoDuplicatesInvariant)):
            return self.origin == other.origin
        return False

def test_EqualInvariant():
    inv1 = format_invariant("x == y")
    inv2 = format_invariant("y == x")
    assert inv1 == inv2
    
def test_NotEqualInvariant():
    inv1 = format_invariant("x != y")
    inv2 = format_invariant("y != x")
    assert inv1 == inv2
    
def test_LessGreaterEqualInvariant():
    inv1 = format_invariant("x >= y")
    inv2 = format_invariant("y <= x")
    assert inv1 == inv2

def test_LessGreaterInvariant():
    inv1 = format_invariant("x > y")
    inv2 = format_invariant("y < x")
    assert inv1 == inv2

def test_ImplyInvariant():
    inv1 = format_invariant("x ==> y")
    inv2 = format_invariant("x ==> y")
    assert inv1 == inv2
    
def test_EquivInvariant():
    inv1 = format_invariant("x <==> y")
    inv2 = format_invariant("y <==> x")
    assert inv1 == inv2

def test_in_array():
    inv = format_invariant("x <==> y")
    inv1 = format_invariant("x == y")
    inv2 = format_invariant("y <==> x")
    inv3 = format_invariant("z <= t")
    inv4 = format_invariant("g => h")
    print(inv in [inv1, inv2, inv3, inv4])

