'''
第6题情况比较特殊，直接做速度无法接受，要评估时少训几轮，求优时再训练完全
'''
import os
import time
import numpy as np
from Reader import ReadU
from utils import RandomCNOTs
from ArchitectureSearch import RandomSearch
from paddle_model import BackwardParams

INPUT_TXT = 'Questions/Question_6_Unitary.txt'
ANSWER_TXT = '../Answer/Question_6_Answer.txt'

LAYER_COUNT=15

def main(in_txt=INPUT_TXT, out_txt=ANSWER_TXT):
    U = ReadU(in_txt)
    quantum_count = 8
    np.random.seed(2021)
    cnot_creater = lambda:RandomCNOTs(quantum_count, layer_count=LAYER_COUNT)
    solver_faster = lambda cnot:BackwardParams(U, quantum_count, cnot_layers=cnot, learning_rate=0.1, iterations=500, verbose=1)
    solver_better = lambda cnot:BackwardParams(U, quantum_count, cnot_layers=cnot, learning_rate=0.01, iterations=500, verbose=10)
    best_score=0
    start_time = time.time()
    for epoch in range(1):
        cnot = cnot_creater()
        sc, model_str = solver_faster(cnot)
        if best_score<sc:
            best_score = sc
            best_cnot = cnot
        print('time=%gs, epoch_%d, sc=%g, best_score=%g'%(time.time()-start_time, epoch, sc, best_score))
    # best_score, model_str = solver_better(best_cnot)
    print('In question 6: best_score = %g'%(best_score))
    with open(out_txt, 'w') as f:
        f.write(model_str)

if __name__ == '__main__':
    if not os.path.exists('../Answer'):
        os.makedirs('../Answer')
    main()
