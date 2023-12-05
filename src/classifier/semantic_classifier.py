from tqdm import tqdm

def get_patch_difference(baselineprog, prog):
    print("Error behaviours ...")
    diff = {}
    keys = list(prog.keys())
    for i in tqdm(range(len(keys))):
        program_point = keys[i]
        if program_point in baselineprog:
            prog_inv = prog[program_point]
            baseline_inv = baselineprog[program_point]
            diff[program_point] = []
            for inv in prog_inv:
                if inv not in baseline_inv:
                    diff[program_point].append(inv)
    return diff

def get_patch_intersect(baselineprog, prog):
    print("Correct specifications ...")
    intersection = {}
    keys = list(baselineprog.keys())
    for i in tqdm(range(len(keys))):
        program_point = keys[i]
        if program_point in prog:
            baseline_inv = baselineprog[program_point]
            patch_inv = prog[program_point]
            intersection[program_point] = []
            for inv_b in baseline_inv:
                if inv_b in patch_inv:
                    intersection[program_point].append(inv_b)
    return intersection

def overfitting_1(patch_inv, error_beha):
    print("MAINTAIN ERROR:")
    score = 0
    for program_point in patch_inv:
        if program_point in error_beha:
            for inv in error_beha[program_point]:
               if inv in patch_inv[program_point]:
                   print(program_point + ": \t")
                   print(inv)
                   if inv != "Exiting Daikon.":
                       score += 1
    if score > 0:
        return True
    return False

def overfitting_2(patch_inv, correct_spec):
    print("VIOLATE CORRECT:")
    score = 0
    for program_point in patch_inv:
        if program_point in correct_spec:
            for inv in correct_spec[program_point]:
               if inv not in patch_inv[program_point]:
                   if inv != "Exiting Daikon.":
                       print(program_point + ": \t")
                       print(inv)
                       score += 1

    if score > 0:
        return True
    return False