#!/bin/bash
#SBATCH --job-name=eval_checkpoints
#SBATCH --account=hitz-exclusive
#SBATCH --partition=hitz-exclusive
#SBATCH --qos=hitz-exclusive
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=64GB
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/agarciam/eval_checkpoints_%j.out
#SBATCH --error=/scratch/agarciam/eval_checkpoints_%j.err

module load Miniforge3/24.11.3-2
export PATH="/home/agarciam/.conda/envs/llama-omni/bin:$PATH"

cd /scratch/agarciam/tfm/models/Latxa-Omni-Emotion

OUTPUT_DIR="saves_r8_adapter"

for checkpoint in ${OUTPUT_DIR}/checkpoint-*; do
    step=$(basename "$checkpoint" | sed 's/checkpoint-//')
    echo "=================================================="
    echo "Evaluating checkpoint at step $step"
    echo "=================================================="
    python -u evaluation.py \
        --model-path "$checkpoint" \
        --split val
    
    # Renombrar el archivo de resultados para no sobrescribir entre checkpoints
    mv val_predictions.json "val_predictions_step${step}.json"
done

echo "All checkpoints evaluated."