#!/bin/bash
#SBATCH --job-name=eval_checkpoints
#SBATCH --account=hitz-exclusive
#SBATCH --partition=hitz-exclusive
#SBATCH --qos=hitz-exclusive
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=64GB
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/agarciam/tfm/outputs/logs/val_eval_checkpoints_6.out
#SBATCH --error=/scratch/agarciam/tfm/outputs/logs/val_eval_checkpoints_6.err

module load Miniforge3/24.11.3-2
export PATH="/home/agarciam/.conda/envs/llama-omni/bin:$PATH"

cd /scratch/agarciam/tfm
export PYTHONPATH="$PWD:$PYTHONPATH"

mkdir -p val_predictions

for OUTPUT_DIR in /hitz_data/agarciam/saves_adam_098/saves_full_ft_6; do
    for checkpoint in ${OUTPUT_DIR}/checkpoint-*; do
        step=$(basename "$checkpoint" | sed 's/checkpoint-//')
        echo "=================================================="
        echo "Evaluating ${OUTPUT_DIR} checkpoint at step $step"
        echo "=================================================="
        python -u evaluation_ser/evaluation.py \
            --model-path "$checkpoint" \
            --split val
        
        mv val_predictions_ft.json "val_predictions/${OUTPUT_DIR}_step${step}.json"
    done
done

echo "All checkpoints evaluated."