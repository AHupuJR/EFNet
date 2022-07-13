source /cluster/home/leisun/base/bin/activate
export PYTHONPATH="${PYTHONPATH}: ./"
python setup.py develop --no_cuda_ext
python ./basicsr/test.py -opt options/test/GoPro/Test-Full_EVHINet.yml