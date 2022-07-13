#!/bin/bash

cd /cluster/home/${USER}/EventDeblurProject/HINet_mydataloader

# titan
# bsub -W 48:00 -n 24 \
# -R "rusage[ngpus_excl_p=4,mem=4000,scratch=2000]" \
# -R "select[gpu_mtotal0>=20000]"  < ./sh_files/bb__sub_son.sh

# 2080ti
# bsub -W 72:00 -n 24 \
# -R "rusage[ngpus_excl_p=4,mem=4000,scratch=2000]" \
# -R "select[gpu_mtotal0>=10000]"  < ./sh_files/bb__sub_son.sh

# 1 titan
bsub -W 120:00 -n 16 \
-R "rusage[ngpus_excl_p=1,mem=4000,scratch=2000]" \
-R "select[gpu_mtotal0>=20000]"  < ./sh_files/bb__sub_son.sh

# delete all experiments, results, tb_logger before submit !!
# change name in yml every time before submit !!

