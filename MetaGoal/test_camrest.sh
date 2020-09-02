#!/bin/bash

# set gpu id to use
#export CUDA_VISIBLE_DEVICES=2

# set python path according to your actual environment
pythonpath_test='python3'
pythonpath_eval='python3 -m'


test_model=S
output_dir=./outputs-multiwoz-${test_model}
data_name=multiwoz
data_dir=./data/multiwoz
#${pythonpath_eval} tools.eval --data_name=${data_name} --data_dir=${data_dir} --eval_dir=${output_dir}/${ckpt}

for i in {20..25}
do
  #a=10
  #let i*=a
  ckpt=state_epoch_${i}.model
  mkdir -p ${output_dir}/${ckpt}
  ${pythonpath_test} ./main.py --test=True --data_dir=${data_dir} --data_name=${data_name} --ckpt=${ckpt} --save_file=${output_dir}/${ckpt}/output.txt
  ${pythonpath_eval} tools.eval --data_name=${data_name} --data_dir=${data_dir} --eval_dir=${output_dir}/${ckpt}
done
