#!/bin/bash
#SBATCH -J yuchen                   # 作业名为 test
#SBATCH -o result.out               # 屏幕上的输出文件重定向到 test.out
#SBATCH -p Student                    # 作业提交的分区为 cpu
#SBATCH -N 1                      # 作业申请 1 个节点
#SBATCH --ntasks-per-node=1       # 单节点启动的进程数为 1
#SBATCH --cpus-per-task=4         # 单任务使用的 CPU 核心数为 4
#SBATCH -t 5:00:00                # 任务运行的最长时间为 1 小时
#SBATCH --gres=gpu:1              # 单个节点使用 1 块 GPU 卡
#SBATCH --mem=10G              # 单个节点使用 1 块 GPU 卡

python train.py