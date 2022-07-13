source /cluster/home/leisun/base/bin/activate
export PYTHONPATH="${PYTHONPATH}: ./"
python setup.py develop --no_cuda_ext

python ./basicsr/train.py -opt options/train/GoPro/EV_TransformerHINet.yml