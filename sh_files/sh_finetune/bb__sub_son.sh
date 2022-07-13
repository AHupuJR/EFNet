#!/bin/bash
#B SUB -n 24                                    # 24 cpu cores
#B SUB -W 120:00                                 # 48-hour hard-limit run-time (bmod -W 24:00 JOBID for more time)
#B SUB -R "rusage[ngpus_excl_p=3,mem=3000,scratch=2000]"     # gpu number, memroy per core; scratch=50000 for tmp space
#B SUB -R "select[gpu_mtotal0>=10000]"          # gpu memory (i.e., type)
#B SUB -J analysis1                            # job name
#B SUB -o analysis1.out                        # output file
#B SUB -e analysis1.err                        # error file (merged with output file by default)
#BSUB -B                                      # email begin
#BSUB -N                                       # email end
#BSUB -u youremail@ethz.ch

RAND_PORT=$(( ((RANDOM<<15)|RANDOM) % 45000 + 20000 ))
TOTAL_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
TOTAL_CUDA_VISIBLE_DEVICES=$((${#TOTAL_CUDA_VISIBLE_DEVICES}/2+1))
echo "running job=$LSB_BATCH_JID by $USER@$HOSTNAME(GPU: $CUDA_VISIBLE_DEVICES of $TOTAL_CUDA_VISIBLE_DEVICES) in $LS_SUBCWD"
echo Starting on: `date`


echo "copying codes"
rsync -aq ./ ${TMPDIR}
cd $TMPDIR

# echo "copying dataset"
# mkdir ${TMPDIR}/datasets
# rsync -aqr /cluster/work/cvl/leisun/GOPRO_fullsize_h5_bin6_ver3/ ${TMPDIR}/datasets

# rsync -aqr /cluster/work/cvl/leisun/GOPRO_fullsize_h5_bin6_sbt/ ${TMPDIR}/datasets
# rsync -aqr /cluster/work/cvl/leisun/GOPRO_fullsize_h5_bin1_sbt/ ${TMPDIR}/datasets
# rsync -aqr /cluster/work/cvl/leisun/GOPRO_upsample_h5_bin6_ver3_c02/ ${TMPDIR}/datasets
# rsync -aqr /cluster/work/cvl/leisun/GOPRO_upsample_h5_bin6_ver3_allC/ ${TMPDIR}/datasets # All C Dataset
# rsync -aqr /cluster/work/cvl/leisun/GOPRO_fullsize_h5_bin6_ver3_nocenter_noisy_c02/ ${TMPDIR}/datasets #  Noisy dataset
# rsync -aqr /cluster/work/cvl/leisun/GOPRO_upsample_h5_bin6_ver3_c08/ ${TMPDIR}/datasets # c=0.8 noisy dataset


#module load pigz
# ln -s /cluster/work/cvl/leisun/log/experiments ${TMPDIR} #!
# ln -s /cluster/work/cvl/leisun/log/results ${TMPDIR}
# ln -s /cluster/work/cvl/leisun/log/tb_logger ${TMPDIR}

#module load pigz
ln -s /cluster/work/cvl/leisun/log/experiments ./ #!
ln -s /cluster/work/cvl/leisun/log/results $./
ln -s /cluster/work/cvl/leisun/log/tb_logger ./




### 3, run experiments in ${TMPDIR}
#cd ${TMPDIR}/codes

source /cluster/apps/local/env2lmod.sh && module load gcc/6.3.0 python_gpu/3.8.5
source /cluster/project/cvl/admin/cvl_settings


# if (( $IS_DISTRIBUTED > 0 )); then

source /cluster/home/leisun/base/bin/activate
export PYTHONPATH="${PYTHONPATH}: ./"
python setup.py develop --no_cuda_ext

python ./basicsr/train.py -opt options/train/HQBlur/SRN_Event.yml