import os
import json
import sys
import time
import random
import subprocess
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

folder = "BCGP"
name = "bcgp"

log_dir = os.path.join("Log")
json_dir = os.path.join("Json")
results_dir = os.path.join("Results", folder)
os.makedirs(log_dir, exist_ok=True)
os.makedirs(json_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

def run_command(func, op, gen, pop, run, log_file):
    try:
        command = f'python3 main.py {run} {func} {op} {pop} {gen}'
        start_time = time.time()
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            with open(log_file, 'a') as f:
                f.write(f"Run {run}: Command failed with return code {result.returncode}\n")
            return None, None, None

        lines = result.stdout.splitlines()
        mae = None
        op_val = None
        gen = None
        for line in lines:
            if line.startswith("Final MAE"):
                mae = float(line.split(":")[1].strip())
            elif line.startswith("Final OP"):
                op_val = int(line.split(":")[1].strip())
            elif line.startswith("Total Gen"):
                gen = int(line.split(":")[1].strip())

        end_time = time.time()
        with open(log_file, 'a') as f:
            status = "Succeed" if mae is not None and mae < 1e-5 else "Fail"
            f.write(f"Run {run}: MAE={mae:.8f}, OP={op_val}, Gen={gen}, Status={status}, Time={end_time - start_time:.2f}s\n")

        return mae, op_val, gen
    except Exception as e:
        with open(log_file, 'a') as f:
            f.write(f"Run {run}: Exception occurred - {e}\n")
        return None, None, None

def run_pop_repeat(func, max_op, gen, pop, runtime):
    best_ops = []
    actual_ops = []
    actual_maes = []
    fail_maes = []
    cost_gens = []

    log_file = f"./Results/{folder}/log_pop{pop}_{name}_f{func}.txt"
    with open(log_file, 'w') as f:
        f.write(f'Problem: f{func}\n')
        f.write(f'Population size: {pop}\n')

    futures = {}
    with ThreadPoolExecutor(max_workers=1) as executor:
        for run_number in range(runtime):
            seed = random.randint(0, 99999)
            futures[executor.submit(run_command, func, max_op, gen, pop, seed, log_file)] = run_number

        for future in as_completed(futures):
            mae, op_val, gen = future.result()
            if gen is not None:
                cost_gens.append(gen)
            if mae is not None:
                actual_maes.append(mae)
                actual_ops.append(op_val if mae < 1e-5 else np.nan)
                if mae < 1e-5:
                    best_ops.append(op_val)
                else:
                    fail_maes.append(mae)

    mae_scores = []
    for i in range(len(actual_maes)):
        if actual_maes[i] < 1e-5:
            mae_scores.append(0.0)
        else:
            mae_scores.append(actual_maes[i])

    avg_op_all = np.nanmean(best_ops) if len(best_ops) > 0 else np.nan
    std_op_all = np.nanstd(best_ops, ddof=0) if len(best_ops) > 1 else np.nan
    sstd_op_all = np.nanstd(best_ops, ddof=1) if len(best_ops) > 1 else np.nan
    avg_mae_all = np.mean(actual_maes) if actual_maes else 0
    std_mae_all = np.std(actual_maes, ddof=1) if len(actual_maes) > 1 else 0
    avg_mae_score = np.mean(mae_scores) if mae_scores else 0
    std_mae_score = np.std(mae_scores, ddof=0) if len(mae_scores) > 1 else np.nan

    with open(log_file, 'a') as f:
        f.write(f"Recorded Values: {best_ops}\n")
        f.write(f"Avg OP_num when Succ: {avg_op_all}\n")
        f.write(f"Stdev OP_num when Succ: {std_op_all}\n")
        f.write(f"Sample Stdev OP_num when Succ: {sstd_op_all}\n")
        f.write(f"Num of Succ: {len(best_ops)} out of {10}\n")
        f.write(f"Recorded Values: {actual_maes}\n") # MAE
        f.write(f"Avg MAE of all: {avg_mae_all}\n")
        f.write(f"Stdev MAE of all: {std_mae_all}\n")
        f.write(f"Avg MAE Score: {avg_mae_score}\n")
        f.write(f"Stdev MAE Score: {std_mae_score}\n")

    return actual_ops, best_ops, cost_gens, actual_maes, avg_mae_score, std_mae_score

def sweep():
    if len(sys.argv) != 5:
        print("Usage: python mysweep_test.py <func_num> <generation_limit> <op_limit> <num_repeat>")
        return

    func_num = int(sys.argv[1])
    gen_limit = int(sys.argv[2])
    op_limit = int(sys.argv[3])
    repeat = int(sys.argv[4])
    pops = [500, 400, 300, 200, 100, 50]

    print(f"Running sweep for f{func_num}, max_op={op_limit}")

    start_time = time.time()
    all_results = {}
    json_results = {}
    json_results = {
        "func_id": func_num
    }

    log_file = f"./Log/log_f{func_num}.txt"
    with open(log_file, 'w') as f:
        f.write(f'Function: f{func_num}\n')
        f.write(f'Op Limits: {op_limit}\n')
        f.write(f'Max Generation: {gen_limit}\n')
        f.write(f'Initial Depth Limits: ({2}, {6})\n')
        f.write(f'Tournament Size: {2}\n')
        f.write(f'Function Sets: add, sub, mul, div, sin, cos, exp\n')
        f.write(f'Penalty: {0.001}\n')
        f.write(f'Crossover Rate: {1.0}\n')
        f.write(f'Subtree Mutation Rate: {0.0}\n')
        f.write(f'Hoist Mutation Rate: {0.0}\n')
        f.write(f'Point Mutation Rate: {0.0}\n')
        f.write(f'Reproduction: {0.1}\n')
        f.write(f'Constant: None\n')
        f.write(f'Epoch: {10}\n')
        f.write(f'MLP Layers: (3, 128, 2)\n')
        f.write(f'MLP Model: Fixed Embeddings\n')
        f.write(f'MLP Activation Function: ReLU\n')
        f.write(f'Operator Encoding: cutpoint\n')

    best_pop = 0
    best_mae_score_mean = 1e6
    best_mae_score_std = 0
    best_maes = []
    for pop in pops:
        print(f"--- Population {pop} ---")
        final_ops, success_ops, final_gens, actual_maes, avg_mae_score, std_mae_score = run_pop_repeat(func_num, op_limit, gen_limit, pop, repeat)

        json_results[f"pop_{pop}"] = {
            "avg_mae_score": avg_mae_score, 
            "stddev_mae_score": std_mae_score, 
            "maes": actual_maes
        }

        if avg_mae_score < best_mae_score_mean:
            best_pop = pop
            best_mae_score_mean = avg_mae_score
            best_mae_score_std = std_mae_score
            best_maes = actual_maes
        
        final_avg_op = np.mean(final_ops)
        final_std_op = np.std(final_ops, ddof=1)
        success_avg_op = np.mean(success_ops) if success_ops else 0
        success_std_op = np.std(success_ops, ddof=1) if success_ops else 0
        final_avg_gen = np.mean(final_gens)
        final_std_gen = np.std(final_gens, ddof=1)

        all_results[f"{pop}"] = {
            "Avg OP": final_avg_op,
            "Std OP": final_std_op,
            "Avg Gen": final_avg_gen,
            "Std Gen": final_std_gen
        }
        all_results[f"{pop} success"] = {
            "Avg OP": success_avg_op,
            "Std OP": success_std_op,
            "Success Rate": len(success_ops) / repeat
        }

    with open(log_file, 'a') as f:
        f.write(f'Best Population: {best_pop}\n')
        f.write(f'Avg MAE Score: {best_mae_score_mean}\n')
        f.write(f'Stdev MAE Score: {best_mae_score_std}\n')
        f.write(f'MAEs: ')
        for i in range(len(best_maes)):
            f.write(f'{best_maes[i]} ')

    json_file = f"./Json/log_f{func_num}.json"
    with open(json_file, "w") as jsonout:
        json.dump(json_results, jsonout, indent=2)

    print(f"Total time cost: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    sweep()

'''
python sweep.py func_num gen_limit op_limit repeat
python sweep.py 1 1000 70000000 10
'''