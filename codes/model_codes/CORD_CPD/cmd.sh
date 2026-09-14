#!/usr/bin/env bash
GPU=${1:-0}
mode=${2:-train}
exp_name=${3:-my_exp}
num_atoms=${4:-15}

echo "Mode: ${mode}"
echo "GPU: ${GPU}"
echo "Experiment: ${exp_name}"
echo "Number of atoms: ${num_atoms}"

#if [ ${mode} == 'train' ]
#then
#    data_path=data/cp_change
#elif [ ${mode} == 'test' ]
#then
#    data_path=data
#fi

#for real data
if [ ${mode} == 'train' ]
then
    data_path=data/cp_gdelt
elif [ ${mode} == 'test' ]
then
    data_path=data
fi

# Parameter for RNN TEL, GNN SEL
#CUDA_VISIBLE_DEVICES=${GPU} python.exe main.py --mode $mode --data_path $data_path \
#    --spatial-encoding-layer gnn --temporal-encoding-layer rnn --epochs 10 \
#    --exp_dir exp/${exp_name}_rnn_gnn

# Parameter for TRANS TEL, GNN SEL
#CUDA_VISIBLE_DEVICES=${GPU} python main.py --mode $mode --data_path $data_path \
#    --spatial-encoding-layer gnn --temporal-encoding-layer transformer \
#    --encoder-hidden 64 --decoder mlp --batch-size 32  \
#    --exp_dir exp/${exp_name}_trans_gnn


# Parameter for RNN TEL, TRANS SEL
#CUDA_VISIBLE_DEVICES=${GPU} python main.py --mode $mode --data_path $data_path \
#    --spatial-encoding-layer trans --temporal-encoding-layer rnn \
#    --encoder-hidden 64 --decoder mlp --batch-size 32  \
#    --exp_dir exp/${exp_name}_rnn_trans

# For real data example
CUDA_VISIBLE_DEVICES=${GPU} python.exe main.py --mode ${mode} --data_path data/cp_gdelt --data_type='real' \
    --spatial-encoding-layer gnn --temporal-encoding-layer rnn --dims 21 --epochs 100 \
    --exp_dir exp/${exp_name} --batch-size 16 --prediction-steps 1 --num-atoms ${num_atoms} --timesteps 489

# For obvious data example
#CUDA_VISIBLE_DEVICES=${GPU} python.exe main.py --mode ${mode} --data_path data/cp_obvious --data_type='real' \
#    --spatial-encoding-layer gnn --temporal-encoding-layer rnn --dims 6 --epochs 50 \
#    --exp_dir exp/${exp_name}_rnn_gnn_obv_data --batch-size 16 --prediction-steps 1 --num-atoms 7 --timesteps 20