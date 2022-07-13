#!/bin/bash

cd /cluster/home/${USER}/HINet_mydataloader

# IS_DISTRIBUTED = 1 \
bsub -W 4:00 -n 12 \
-R "rusage[ngpus_excl_p=1,mem=4000,scratch=2000]" \
-R "select[gpu_mtotal0>=10000]"  < ./sh_files/bb__sub_son.sh

# delete all experiments, results, tb_logger before submit !!
# change name in yml every time before submit !!

