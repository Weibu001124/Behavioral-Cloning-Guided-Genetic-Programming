import sys
import math
import random
import math
import time
import resource
import datetime
import numpy as np
from gplearn.genetic import SymbolicRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

def main():
    start = time.time()
    run = int(sys.argv[1])
    func = int(sys.argv[2])
    op = int(sys.argv[3])
    sweep_pop = int(sys.argv[4])
    gen = int(sys.argv[5])

    print(f'Problem: f{func}')
    print(f'OP_quota: {op}')
    print(f'sweep_pop: {sweep_pop}')
    print(f'generations: {gen}')

    func_to_files = {
        1: ("./Benchmark/order6_train.txt", "./Benchmark/order6_test.txt"),
    }

    if func in func_to_files:
        training_data, testing_data = func_to_files[func]
    else:
        print(f"Function {func} not found.")
        return

    f1 = open(training_data)
    X_train = []
    y_train = []
    for line in f1.readlines():
        # print(line)
        # line = line.split(" ")
        # X_train.append([float(line[0])])
        # y_train.append([float(line[1])])
        nums = list(map(float, line.strip().split()))
        X_train.append(nums[:-1])
        y_train.append(nums[-1])
    f1.close()
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    y_train = y_train.ravel()
    # print(f'X_train: {X_train}')
    # print(f'y_train: {y_train}')

    f2 = open(testing_data)
    X_test = []
    y_test = []
    for line in f2.readlines():
        # print(line)
        # line = line.split(" ")
        # X_test.append([float(line[0])])
        # y_test.append([float(line[1])])
        nums = list(map(float, line.strip().split()))
        X_test.append(nums[:-1])
        y_test.append(nums[-1])
    f2.close()
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    # print(f'X_test: {X_test}')
    # print(f'y_test: {y_test}')
        
    print(f'Running...')
    est_gp = SymbolicRegressor(
            population_size=sweep_pop, 
            generations=gen, # 1000,
            op=op,
            tournament_size=2,
            stopping_criteria=1e-5, # 0 
            const_range=None,
            init_depth=(2, 6), # default
            init_method='half and half', # grow, full, half and half
            function_set=('add', 'sub', 'mul', 'div', 'sin', 'cos', 'exp'), # default # 'add', 'sub', 'mul', 'div'
            p_crossover=1.0, 
            p_subtree_mutation=0.0,
            p_hoist_mutation=0.0, 
            p_point_mutation=0.0,
            p_point_replace=0.0,
            max_samples=1.0,
            verbose=1,
            parsimony_coefficient=0.001, # 0.001
            n_jobs=1, 
            # random_state=run, 
            random_state=random.seed(datetime.datetime.now().timestamp()+run)
    )                  
    est_gp.fit(X_train, y_train, None)
    end = time.time()

    print("Gen Test:")
    print(f'Best program: {est_gp._program}')   
    print('Final MAE:', mean_absolute_error(y_test, est_gp.predict(X_test)))
    print(f'Final OP: {int(est_gp.final_op_count)}')
    print(f'Total Gen: {est_gp.costgen}')
    print(f'Execution time: {end - start} secs')

if __name__ == '__main__':
    main()

'''
run this
python main.py run function op population generation
python3 main.py 2025 1 70000000 500 1000
'''