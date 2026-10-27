import json
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

def parse_training_log(path):
    steps, losses = [], []
    eval_steps, eval_losses = [], []
    step_counter = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("{'loss'") and not line.startswith("{'eval_loss'"):
                continue
            try:
                entry = json.loads(line.replace("'", '"'))
            except json.JSONDecodeError:
                continue
            if 'loss' in entry and 'eval_loss' not in entry:
                step_counter += 1
                steps.append(step_counter)
                losses.append(entry['loss'])
            elif 'eval_loss' in entry:
                eval_steps.append(step_counter)
                eval_losses.append(entry['eval_loss'])
    return steps, losses, eval_steps, eval_losses

def parse_checkpoint_evaluations(log_path):
    with open(log_path, "r", encoding="utf-8") as f:
        content = f.read()
    blocks = re.split(r"={10,}\nEvaluating (\S+) checkpoint at step (\d+)\n={10,}", content)
    results = []
    for i in range(1, len(blocks), 3):
        fold = blocks[i]
        step = int(blocks[i+1])
        block_content = blocks[i+2]
        acc_match = re.search(r"accuracy\s+([\d.]+)\s+\d+", block_content)
        if acc_match:
            results.append({"fold": fold, "step": step, "accuracy": float(acc_match.group(1))})
    return results

# --- Configuración: ajusta estas rutas ---
TRAINING_LOG = "/scratch/agarciam/instruct_fold_6.out"  # el .out del ENTRENAMIENTO de este fold
CHECKPOINT_EVAL_LOG = "/scratch/agarciam/val_eval_checkpoints_6.out"
FOLD_NAME = "/hitz_data/agarciam/saves_adam_098/saves_full_ft_6"  # el nombre exacto usado en el log de checkpoints
OUTPUT_IMAGE = "fold4_combined_curve_new.png"

# --- Cargar datos ---
train_steps, train_loss, eval_loss_steps, eval_loss = parse_training_log(TRAINING_LOG)

all_checkpoint_results = parse_checkpoint_evaluations(CHECKPOINT_EVAL_LOG)
fold_results = [r for r in all_checkpoint_results if r["fold"] == FOLD_NAME]
fold_results.sort(key=lambda x: x["step"])
acc_steps = [r["step"] for r in fold_results]
acc_values = [r["accuracy"] for r in fold_results]

# --- Graficar ---
fig, ax1 = plt.subplots(figsize=(10, 6))

loss_smooth = pd.Series(train_loss).rolling(window=20, min_periods=1).mean()
ax1.plot(train_steps, loss_smooth, label='Train Loss', color='red', alpha=0.5)
ax1.plot(eval_loss_steps, eval_loss, label='Val Loss', color='blue', marker='o')
ax1.set_xlabel('Steps')
ax1.set_ylabel('Loss')

ax2 = ax1.twinx()
ax2.plot(acc_steps, acc_values, label='Val Accuracy', color='green', marker='s', linewidth=2)
ax2.set_ylabel('Accuracy', color='green')
ax2.tick_params(axis='y', labelcolor='green')
ax2.set_ylim(0, 1)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2)

plt.title(f'Fold 4 — Loss and Accuracy over Training')
plt.grid(True)
plt.savefig(OUTPUT_IMAGE)
print(f"Graphic saved: {OUTPUT_IMAGE}")
print(f"Best checkpoint by accuracy: step {acc_steps[acc_values.index(max(acc_values))]} (acc={max(acc_values):.3f})")
train_steps, train_loss, eval_loss_steps, eval_loss = parse_training_log("/scratch/agarciam/instruct_fold_6.out")
print("Train steps:", len(train_steps))
print("Train loss (primeros 5):", train_loss[:5])
print("Eval steps:", len(eval_loss_steps))
print("Eval loss:", eval_loss[:5])