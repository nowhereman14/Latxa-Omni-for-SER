#!/bin/bash
#SBATCH --job-name=eval_checkpoints
#SBATCH --account=hitz-exclusive
#SBATCH --partition=hitz-exclusive
#SBATCH --qos=hitz-exclusive
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=64GB
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/agarciam/val_eval_checkpoints_%j.out
#SBATCH --error=/scratch/agarciam/val_eval_checkpoints_%j.err

module load Miniforge3/24.11.3-2
export PATH="/home/agarciam/.conda/envs/llama-omni/bin:$PATH"

cd /scratch/agarciam/tfm/models/Latxa-Omni-Emotion
export PYTHONPATH="$PWD:$PYTHONPATH"

mkdir -p val_predictions

for OUTPUT_DIR in saves_full_ft_ser; do
    for checkpoint in ${OUTPUT_DIR}/checkpoint-*; do
        step=$(basename "$checkpoint" | sed 's/checkpoint-//')
        echo "=================================================="
        echo "Evaluating ${OUTPUT_DIR} checkpoint at step $step"
        echo "=================================================="
        python -u evaluation.py \
            --model-path "$checkpoint" \
            --split val
        
        mv val_predictions_ft.json "val_predictions/${OUTPUT_DIR}_step${step}.json"
    done
done

echo "All checkpoints evaluated."