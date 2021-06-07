import os
import argparse

from Reader import ReadU, ReadSystem
from ops import Fidelity
from utils import CostCompute

def get_parser():
    # 生成argparse对象
    parser = argparse.ArgumentParser(description="Help:")
    
    # 添加需要的参数
    parser.add_argument('--save', type=str, default='True')
 
    # 返回parser对象
    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    print(args)
    in_path = 'Questions/Question_%d_Unitary.txt'
    out_path = '../Answer/Question_%d_Answer.txt'
    score_sum = 1
    for case, quantum_count, F_scale, cost_scale in zip([2,3,4,5,6], [2,3,3,4,8], [2,3,11,34,70], [0,0,1/400,1/400,1/4000]):
        in_file_path = in_path%case
        out_file_path = out_path%case

        if not os.path.exists(out_file_path):
            continue
        U = ReadU(in_file_path)
        M = ReadSystem(out_file_path, quantum_count)
        cost = CostCompute(M.string)
        F = Fidelity(U, M.matrix)
        sc = F_scale*(F-cost*cost_scale) if F>0.75 else 0
        score_sum += sc
        print('In question %d:'%case)
        print("Fidelity: ", F)
        print('Cost:', cost)
        print('Score:', sc)
        # if case == 4:
        #     print(M.string)
        print()
    print('Total score:', score_sum)
    if args.save.lower() == 'true':
        if not os.path.exists('../submits'):
            os.makedirs('../submits')
        os.system('cd ../ && rm -rf *.zip && zip -r Answer.zip Answer/*.txt')
        os.system('cd ../ && zip -r src_%.2f.zip work/ Answer.zip && mv src_%.2f.zip submits/'%(score_sum, score_sum))
        print('save to: ~/submits/src_%.2f.zip'%(score_sum))
