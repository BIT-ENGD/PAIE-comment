if [ $# == 0 ] 
then
    SEED=42
    LR=2e-5
else
    SEED=$1
    LR=$2
fi

work_path=exps/ace05_softprompt/$SEED/$LR
mkdir -p $work_path

srun -p dsta --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --kill-on-bad-exit=1 -w SG-IDC1-10-51-2-70 python engine.py \
    --model_type paie \
    --dataset_type ace_eeqa \
    --model_name_or_path facebook/bart-base \
    --role_path ./data/dset_meta/description_ace.csv \
    --prompt_path ./data/prompts/prompts_ace_continuous.csv \
    --seed $SEED \
    --output_dir $work_path  \
    --learning_rate $LR \
    --batch_size 32 \
    --eval_steps 500  \
    --max_steps 10000 \
    --max_enc_seq_length 180 \
    --max_dec_seq_length 20 \
    --max_prompt_seq_length 50 \
    --bipartite