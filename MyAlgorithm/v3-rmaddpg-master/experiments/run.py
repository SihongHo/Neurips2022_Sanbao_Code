import sys
import os
import copy
import json
import time
import argparse

from grpc import server

# nohup python run.py --exp-index 0 --server-name "FeiNewML"> 2022_04_30_run.out 2> 2022_04_30_run.err &
# nohup python run.py --exp-index 1 --lr-adv 5e-2 --server-name "FeiNewML"> 2022_05_01_large_run.out 2> 2022_05_01_large_run.err &
# nohup python run.py --exp-index 2 --lr-adv 1e-3 --server-name "FeiML"> 2022_05_01_small_run.out 2> 2022_05_01_small_run.err &
def get_command(scenario_index = 8, exp_index = 0, cuda_index = 2, alg = "ma", server_name = "FeiML", lr_adv = "2e-2"):
    scenario_list = ["simple","simple_reference", "simple_speaker_listener", "simple_spread",
                        "simple_adversary", "simple_crypto", "simple_push",
                        "simple_tag", "simple_world_comm"]
    num_adv_list = [0, 0, 0, 0, 1, 1, 1, 3, 4]
    exp_name_list = ["e0" + str(i) for i in range(1,10,1)] + ["e" + str(i) for i in range(10,21,1)]
    exp_name = alg + "_s" + str(scenario_index) + "_" + str(exp_name_list[exp_index])
    exp_path = " > results/s" + str(scenario_index) + "/" + exp_name + ".out 2> results/s" \
                    + str(scenario_index) + "/" + exp_name + ".err &"

    opt1 = dict()
    opt1['server-name'] = server_name
    opt1['scenario'] = scenario_list[scenario_index]
    opt1['num-adversaries'] = num_adv_list[scenario_index]
    opt1['save-dir'] = "models/s" + str(scenario_index) + "/" + exp_name + "/"
    opt1['lr-adv'] = lr_adv
    opt2 = dict()
    opt2['exp-name'] = exp_name + exp_path

    def generate_command(opt1, opt2, cuda_index):
        cmd = 'CUDA_VISIBLE_DEVICES=' + str(cuda_index) + ' nohup python train.py'
        for opt, val in opt1.items():
            cmd += ' --' + opt + ' ' + str(val)
        for opt, val in opt2.items():
            cmd += ' --' + opt + ' ' + str(val)
        return cmd
    
    return generate_command(opt1, opt2, cuda_index)

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning Experiments Excution")
    parser.add_argument("--server-name", type=str, default="FeiML", help="FeiML, FeiNewML, Miao_Exxact")
    parser.add_argument("--exp-index", type=int, default=0, help="FeiML, FeiNewML, Miao_Exxact = 1 2 3")
    parser.add_argument("--lr-adv", type=str, default=2e-2, help="learning rate of adversary")
    return parser.parse_args()

def run(scenario_index, exp_index, server_name, lr_adv):
    opt = get_command(scenario_index = scenario_index, exp_index = exp_index, cuda_index = 1, alg = "ma", server_name = server_name, lr_adv = lr_adv)
    conda_command = "conda activate hsh_maddpg"
    # print(conda_command)
    print(opt)
    # os.system(conda_command)
    os.system(opt)
    print("------------------------sleep------------------------")
    time.sleep(3600) # sleep for 1h

if __name__ == '__main__':
    arglist = parse_args()
    run(scenario_index = 1, server_name = arglist.server_name, exp_index = arglist.exp_index, lr_adv = arglist.lr_adv)
    run(scenario_index = 2, server_name = arglist.server_name, exp_index = arglist.exp_index, lr_adv = arglist.lr_adv)
    run(scenario_index = 3, server_name = arglist.server_name, exp_index = arglist.exp_index, lr_adv = arglist.lr_adv)
    run(scenario_index = 4, server_name = arglist.server_name, exp_index = arglist.exp_index, lr_adv = arglist.lr_adv)
    run(scenario_index = 5, server_name = arglist.server_name, exp_index = arglist.exp_index, lr_adv = arglist.lr_adv)
    run(scenario_index = 6, server_name = arglist.server_name, exp_index = arglist.exp_index, lr_adv = arglist.lr_adv)
    run(scenario_index = 7, server_name = arglist.server_name, exp_index = arglist.exp_index, lr_adv = arglist.lr_adv)
    run(scenario_index = 8, server_name = arglist.server_name, exp_index = arglist.exp_index, lr_adv = arglist.lr_adv)

# CUDA_VISIBLE_DEVICES=1 nohup python train.py --save-dir models/s1/ma_s1_e02/ --server-name FeiML --scenario simple_reference --num-adversaries 0 --exp-name ma_s1_e02 > results/s1/ma_s1_e02.out 2> results/s1/ma_s1_e02.err &
# ------------------------sleep------------------------
# CUDA_VISIBLE_DEVICES=1 nohup python train.py --save-dir models/s2/ma_s2_e02/ --server-name FeiML --scenario simple_speaker_listener --num-adversaries 0 --exp-name ma_s2_e02 > results/s2/ma_s2_e02.out 2> results/s2/ma_s2_e02.err &
# ------------------------sleep------------------------
# CUDA_VISIBLE_DEVICES=1 nohup python train.py --save-dir models/s3/ma_s3_e02/ --server-name FeiML --scenario simple_spread --num-adversaries 0 --exp-name ma_s3_e02 > results/s3/ma_s3_e02.out 2> results/s3/ma_s3_e02.err &
# ------------------------sleep------------------------
# CUDA_VISIBLE_DEVICES=1 nohup python train.py --save-dir models/s4/ma_s4_e02/ --server-name FeiML --scenario simple_adversary --num-adversaries 1 --exp-name ma_s4_e02 > results/s4/ma_s4_e02.out 2> results/s4/ma_s4_e02.err &
# ------------------------sleep------------------------
# CUDA_VISIBLE_DEVICES=1 nohup python train.py --save-dir models/s5/ma_s5_e02/ --server-name FeiML --scenario simple_crypto --num-adversaries 1 --exp-name ma_s5_e02 > results/s5/ma_s5_e02.out 2> results/s5/ma_s5_e02.err &
# ------------------------sleep------------------------
# CUDA_VISIBLE_DEVICES=1 nohup python train.py --save-dir models/s6/ma_s6_e02/ --server-name FeiML --scenario simple_push --num-adversaries 1 --exp-name ma_s6_e02 > results/s6/ma_s6_e02.out 2> results/s6/ma_s6_e02.err &
# ------------------------sleep------------------------
# CUDA_VISIBLE_DEVICES=1 nohup python train.py --save-dir models/s7/ma_s7_e02/ --server-name FeiML --scenario simple_tag --num-adversaries 3 --exp-name ma_s7_e02 > results/s7/ma_s7_e02.out 2> results/s7/ma_s7_e02.err &
# ------------------------sleep------------------------
# CUDA_VISIBLE_DEVICES=1 nohup python train.py --save-dir models/s8/ma_s8_e02/ --server-name FeiML --scenario simple_world_comm --num-adversaries 4 --exp-name ma_s8_e02 > results/s8/ma_s8_e02.out 2> results/s8/ma_s8_e02.err &
# ------------------------sleep------------------------

# CUDA_VISIBLE_DEVICES=1 nohup python train.py --lr-adv 5e-2 --server-name FeiNewML --num-adversaries 0 --save-dir models/s1/ma_s1_e02/ --scenario simple_reference --exp-name ma_s1_e02 > results/s1/ma_s1_e02.out 2> results/s1/ma_s1_e02.err &
# ------------------------sleep------------------------
# CUDA_VISIBLE_DEVICES=1 nohup python train.py --lr-adv 5e-2 --server-name FeiNewML --num-adversaries 0 --save-dir models/s2/ma_s2_e02/ --scenario simple_speaker_listener --exp-name ma_s2_e02 > results/s2/ma_s2_e02.out 2> results/s2/ma_s2_e02.err &
# ------------------------sleep------------------------
# CUDA_VISIBLE_DEVICES=1 nohup python train.py --lr-adv 5e-2 --server-name FeiNewML --num-adversaries 0 --save-dir models/s3/ma_s3_e02/ --scenario simple_spread --exp-name ma_s3_e02 > results/s3/ma_s3_e02.out 2> results/s3/ma_s3_e02.err &
# ------------------------sleep------------------------
# CUDA_VISIBLE_DEVICES=1 nohup python train.py --lr-adv 5e-2 --server-name FeiNewML --num-adversaries 1 --save-dir models/s4/ma_s4_e02/ --scenario simple_adversary --exp-name ma_s4_e02 > results/s4/ma_s4_e02.out 2> results/s4/ma_s4_e02.err &
# ------------------------sleep------------------------
# CUDA_VISIBLE_DEVICES=1 nohup python train.py --lr-adv 5e-2 --server-name FeiNewML --num-adversaries 1 --save-dir models/s5/ma_s5_e02/ --scenario simple_crypto --exp-name ma_s5_e02 > results/s5/ma_s5_e02.out 2> results/s5/ma_s5_e02.err &
# ------------------------sleep------------------------
# CUDA_VISIBLE_DEVICES=1 nohup python train.py --lr-adv 5e-2 --server-name FeiNewML --num-adversaries 1 --save-dir models/s6/ma_s6_e02/ --scenario simple_push --exp-name ma_s6_e02 > results/s6/ma_s6_e02.out 2> results/s6/ma_s6_e02.err &
# ------------------------sleep------------------------
# CUDA_VISIBLE_DEVICES=1 nohup python train.py --lr-adv 5e-2 --server-name FeiNewML --num-adversaries 3 --save-dir models/s7/ma_s7_e02/ --scenario simple_tag --exp-name ma_s7_e02 > results/s7/ma_s7_e02.out 2> results/s7/ma_s7_e02.err &
# ------------------------sleep------------------------
# CUDA_VISIBLE_DEVICES=1 nohup python train.py --lr-adv 5e-2 --server-name FeiNewML --num-adversaries 4 --save-dir models/s8/ma_s8_e02/ --scenario simple_world_comm --exp-name ma_s8_e02 > results/s8/ma_s8_e02.out 2> results/s8/ma_s8_e02.err &
# ------------------------sleep------------------------

# CUDA_VISIBLE_DEVICES=1 nohup python train.py --server-name FeiNewML --num-adversaries 0 --scenario simple_reference --save-dir models/s1/ma_s1_e03/ --lr-adv 1e-3 --exp-name ma_s1_e03 > results/s1/ma_s1_e03.out 2> results/s1/ma_s1_e03.err &
# ------------------------sleep------------------------
# CUDA_VISIBLE_DEVICES=1 nohup python train.py --server-name FeiNewML --num-adversaries 0 --scenario simple_speaker_listener --save-dir models/s2/ma_s2_e03/ --lr-adv 1e-3 --exp-name ma_s2_e03 > results/s2/ma_s2_e03.out 2> results/s2/ma_s2_e03.err &
# ------------------------sleep------------------------
# CUDA_VISIBLE_DEVICES=1 nohup python train.py --server-name FeiNewML --num-adversaries 0 --scenario simple_spread --save-dir models/s3/ma_s3_e03/ --lr-adv 1e-3 --exp-name ma_s3_e03 > results/s3/ma_s3_e03.out 2> results/s3/ma_s3_e03.err &
# ------------------------sleep------------------------
# CUDA_VISIBLE_DEVICES=1 nohup python train.py --server-name FeiNewML --num-adversaries 1 --scenario simple_adversary --save-dir models/s4/ma_s4_e03/ --lr-adv 1e-3 --exp-name ma_s4_e03 > results/s4/ma_s4_e03.out 2> results/s4/ma_s4_e03.err &
# ------------------------sleep------------------------
# CUDA_VISIBLE_DEVICES=1 nohup python train.py --server-name FeiNewML --num-adversaries 1 --scenario simple_crypto --save-dir models/s5/ma_s5_e03/ --lr-adv 1e-3 --exp-name ma_s5_e03 > results/s5/ma_s5_e03.out 2> results/s5/ma_s5_e03.err &
# ------------------------sleep------------------------
# CUDA_VISIBLE_DEVICES=1 nohup python train.py --server-name FeiNewML --num-adversaries 1 --scenario simple_push --save-dir models/s6/ma_s6_e03/ --lr-adv 1e-3 --exp-name ma_s6_e03 > results/s6/ma_s6_e03.out 2> results/s6/ma_s6_e03.err &
# ------------------------sleep------------------------
# CUDA_VISIBLE_DEVICES=1 nohup python train.py --server-name FeiNewML --num-adversaries 3 --scenario simple_tag --save-dir models/s7/ma_s7_e03/ --lr-adv 1e-3 --exp-name ma_s7_e03 > results/s7/ma_s7_e03.out 2> results/s7/ma_s7_e03.err &
# ------------------------sleep------------------------
# CUDA_VISIBLE_DEVICES=1 nohup python train.py --server-name FeiNewML --num-adversaries 4 --scenario simple_world_comm --save-dir models/s8/ma_s8_e03/ --lr-adv 1e-3 --exp-name ma_s8_e03 > results/s8/ma_s8_e03.out 2> results/s8/ma_s8_e03.err &
# ------------------------sleep------------------------