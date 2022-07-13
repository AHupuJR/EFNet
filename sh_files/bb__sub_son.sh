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


### copy large dataset to tmpdir (on computation node, requires scratch=40000) for further accerleration
### 0, preprocessing: compress dataset and save it on my own disk (use -n 32 to speed up)
### module load pigz
### cd /cluster/work/cvl/jinliang/dataset
### tar -I pigz -cvf /cluster/work/cvl/jinliang/dataset/Set5.tar.gz Set5
### tar -I pigz -cvf /cluster/work/cvl/jinliang/dataset/DIV2K+Flickr2K_decoded.tar.gz DIV2K+Flickr2K_decoded
### 1, directly decompress dataset to tmpdir (even faster than copy & decompress)

cd /cluster/home/${USER}/EventDeblurProject/HINet_mydataloader
echo "copying codes"
rsync -aq ./ ${TMPDIR}
cd $TMPDIR

echo "copying dataset"
mkdir ${TMPDIR}/datasets

# GoPro with events
# rsync -aqr /cluster/work/cvl/leisun/GOPRO_fullsize_h5_bin6_ver3/ ${TMPDIR}/datasets

module load pigz
# unzip /cluster/work/cvl/leisun/Datasets/Aberration-DIV2K.zip -d ${TMPDIR}/datasets
unzip /cluster/work/cvl/leisun/Datasets/DIVPano_TCI_v2.zip -d ${TMPDIR}/datasets

# rsync -aqr /cluster/work/cvl/leisun/GOPRO_fullsize_h5_bin6_sbt/ ${TMPDIR}/datasets
# rsync -aqr /cluster/work/cvl/leisun/GOPRO_fullsize_h5_bin1_sbt/ ${TMPDIR}/datasets
# rsync -aqr /cluster/work/cvl/leisun/GOPRO_upsample_h5_bin6_ver3_c02/ ${TMPDIR}/datasets
# rsync -aqr /cluster/work/cvl/leisun/GOPRO_upsample_h5_bin6_ver3_allC/ ${TMPDIR}/datasets # All C Dataset
# rsync -aqr /cluster/work/cvl/leisun/GOPRO_fullsize_h5_bin6_ver3_nocenter_noisy_c02/ ${TMPDIR}/datasets #  Noisy dataset
# rsync -aqr /cluster/work/cvl/leisun/GOPRO_upsample_h5_bin6_ver3_c08/ ${TMPDIR}/datasets # c=0.8 noisy dataset


#module load pigz
#tar -I pigz -xf /cluster/work/cvl/jinliang/dataset/Set5.tar.gz -C ${TMPDIR}/datasets
#tar -I pigz -xf /cluster/work/cvl/jinliang/dataset/DIV2K+Flickr2K_decoded.tar.gz -C ${TMPDIR}/datasets
### 2, copy code and symbolic links to tmpdir
#rsync -aq ./ ${TMPDIR}/codes
#ln -s /cluster/work/cvl/jinliang/log/srflow_experiments ${TMPDIR}/experiments
#ln -s /cluster/work/cvl/jinliang/log/srflow_results ${TMPDIR}/results
#ln -s /cluster/work/cvl/jinliang/log/srflow_tb_logger ${TMPDIR}/tb_logger
# ln -s /cluster/work/cvl/leisun/log/evhinet_experiments ${TMPDIR}/experiments
# ln -s /cluster/work/cvl/leisun/log/evhinet_results ${TMPDIR}/results
# ln -s /cluster/work/cvl/leisun/log/evhinet_tb_logger ${TMPDIR}/tb_logger
rm -r ./experiments
rm -r ./results
rm -r ./tb_logger

ln -s /cluster/work/cvl/leisun/log/experiments ${TMPDIR} #!
ln -s /cluster/work/cvl/leisun/log/results ${TMPDIR}
ln -s /cluster/work/cvl/leisun/log/tb_logger ${TMPDIR}

### 3, run experiments in ${TMPDIR}
#cd ${TMPDIR}/codes

source /cluster/apps/local/env2lmod.sh && module load gcc/6.3.0 python_gpu/3.8.5
source /cluster/project/cvl/admin/cvl_settings


# if (( $IS_DISTRIBUTED > 0 )); then

source /cluster/home/leisun/base/bin/activate
export PYTHONPATH="${PYTHONPATH}: ./"
python setup.py develop --no_cuda_ext

echo "Using distributed nn.DistributedDataParallel."
# python -m torch.distributed.launch --nproc_per_node=$TOTAL_CUDA_VISIBLE_DEVICES --master_port=$RAND_PORT ./basicsr/train.py -opt options/train/GoPro/EVHINet.yml --launcher pytorch
# python -m torch.distributed.launch --nproc_per_node=$TOTAL_CUDA_VISIBLE_DEVICES --master_port=$RAND_PORT ./basicsr/train.py -opt options/train/GoPro/FullHINet.yml --launcher pytorch
# python -m torch.distributed.launch --nproc_per_node=$TOTAL_CUDA_VISIBLE_DEVICES --master_port=$RAND_PORT ./basicsr/train.py -opt options/train/GoPro/SRN_Event.yml --launcher pytorch
# python -m torch.distributed.launch --nproc_per_node=$TOTAL_CUDA_VISIBLE_DEVICES --master_port=$RAND_PORT ./basicsr/train.py -opt options/train/GoPro/Single_HINet.yml --launcher pytorch
# python -m torch.distributed.launch --nproc_per_node=$TOTAL_CUDA_VISIBLE_DEVICES --master_port=$RAND_PORT ./basicsr/train.py -opt options/train/GoPro/HINet.yml --launcher pytorch
# python -m torch.distributed.launch --nproc_per_node=$TOTAL_CUDA_VISIBLE_DEVICES --master_port=$RAND_PORT ./basicsr/train.py -opt options/train/GoPro/SBTNoEMGC.yml --launcher pytorch
# python -m torch.distributed.launch --nproc_per_node=$TOTAL_CUDA_VISIBLE_DEVICES --master_port=$RAND_PORT ./basicsr/train.py -opt options/train/GoPro/Single_EVHINet.yml --launcher pytorch
python -m torch.distributed.launch --nproc_per_node=$TOTAL_CUDA_VISIBLE_DEVICES --master_port=$RAND_PORT ./basicsr/train.py -opt options/train/GoPro/EV_TransformerHINet.yml --launcher pytorch


# python -m torch.distributed.launch --nproc_per_node=$TOTAL_CUDA_VISIBLE_DEVICES \
#     --master_port=$RAND_PORT ./basicsr/train.py -opt options/train/AS_PAL/SRN_new.yml --launcher pytorch



# else
    # echo "Using nn.DataParallel."
    # python -u basicsr/train.py --gpu_ids $CUDA_VISIBLE_DEVICES -opt options/train/GoPro/HINet.yml

# fi

#rsync -auq ${TMPDIR}/ $LS_SUBCWD # copy new files back to my own disk, eg, results

# python basicsr/train.py -opt options/train/GoPro/EV_TransformerHINet.yml
