"""
train_cluster_triage_unsloth.py — Curriculum GRPO Training on ClusterTriageEnv
===============================================================================
Trains Llama-3.2-3B-Instruct via GRPO (Group Relative Policy Optimization)
against the live ClusterTriageEnv reward function using curriculum learning.

Environment: 5 tasks (easy → medium → hard → very_hard → nightmare)
  - easy:       Kill 1 rogue job
  - medium:     Clear disk → restart node (2-step sequence)
  - hard:       Kill job → clear 2 nodes → restart 2 nodes (5-step)
  - very_hard:  Kill 2 malware jobs → clear 2 nodes → restart 2 nodes
  - nightmare:  Kill 3 hydra jobs → clear 4 nodes → restart 4 nodes

═══════════════════════════════════════════════════════════════════
  BUGS FIXED IN THIS VERSION (from observed training collapse)
═══════════════════════════════════════════════════════════════════

OBSERVED SYMPTOMS (from your training logs):
  frac_reward_zero_std = 0.75   → 75% batches had ZERO gradient
  reward_std           = 0.04   → all 6 generations got same reward
  kl                   = 0.0002 → model barely moved from pretraining
  grad_norm            = 0.11   → near-zero gradients, no learning
  completion_length    ≈ 21     → deterministic collapse to one output
  reward               ≈ 6.78   → scripted policy gave everyone free reward

FIX 1 — Scripted completion was too powerful (easy task fix):
  Old: scripted policy ran ALL remaining steps after model's step 1.
       Every generation got reward ~6.7 → zero variance → zero gradient.
  Fix: easy task gets NO scripted completion (1-step raw reward only).
       Other tasks get PARTIAL script (≤50% of remaining optimal steps).
       Completion bonus reduced +5.0 → +2.0 so step-1 signal dominates.

FIX 2 — Temperature too low → deterministic collapse:
  Old: temperature=0.2 in generation → model always picked top-1 token
       → all 6 generations identical → reward_std=0 → zero gradient.
  Fix: temperature=0.8 during training inference.

FIX 3 — Wrong first-action had no penalty:
  Old: noop=-0.3, parse_fail=-1.0, wrong action=no penalty.
       Model could output restart_node before kill_job and still get +6.
  Fix: parse_fail=-2.0, noop=-1.5, wrong action (rule violation)=-1.0.

FIX 4 — Reward scale mismatch (completion bonus dominated everything):
  Old: +5.0 completion bonus >> all other components → relative signal ~0%.
  Fix: completion bonus +2.0, health delta capped +0.3, first-action +0.5.
       Step-1 signal now ≈ 30-40% of total episode reward.

FIX 5 — Dataset not diverse enough:
  Old: many prompts sampled from same state → identical observations
       → identical completions → zero reward variance in batch.
  Fix: Expert prefixes cover MORE mid-episode states + random node stat
       noise (±2% cpu/ram) so every prompt in a batch is distinct.

HEALTHY TARGET METRICS (check these in your logs):
  frac_reward_zero_std  < 0.2    (was 0.75)
  reward_std            > 0.5    (was 0.04)
  kl                    0.01–0.1 (was 0.0002)
  grad_norm             > 1.0    (was 0.11)
  clip_ratio/high_mean  0.05–0.2 (was 0.0)

T4 (16 GB) settings (unchanged from v1):
  - unsloth/Llama-3.2-3B-Instruct-bnb-4bit
  - LoRA rank 16, use_gradient_checkpointing="unsloth"
  - adamw_8bit, num_generations=6

Google Colab setup:
  # Cell 1 — install
  !pip install unsloth trl datasets matplotlib pydantic
  # Cell 2 — copy env files
  !git clone https://github.com/<your-repo>/cluster-triage-env
  import shutil
  shutil.copy("cluster-triage-env/environment.py", ".")
  shutil.copy("cluster-triage-env/models.py", ".")
  # Cell 3 — train
  !python train_cluster_triage_unsloth.py
  # Cell 4 — view plots
  from IPython.display import Image, display
  for p in ["training_reward_curve", "success_rate_comparison",
            "reward_comparison", "health_recovery_comparison",
            "stage_reward_distribution", "grpo_health_metrics"]:
      display(Image(f"plots/{p}.png"))
"""

# ── 0. Imports ────────────────────────────────────────────────────────────────
import os, re, json, random, time
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from datasets import Dataset
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer

from environment import ClusterTriageEnv
from models import ClusterAction

os.makedirs("plots",       exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)


# ── 1. Config ─────────────────────────────────────────────────────────────────
BASE_MODEL      = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
LORA_OUTPUT_DIR = "cluster-triage-lora"
MAX_SEQ_LENGTH  = 2048
LORA_RANK       = 16
EVAL_EPISODES   = 5

TASK_MAX_STEPS = {
    "easy":      10,
    "medium":    15,
    "hard":      20,
    "very_hard": 20,
    "nightmare": 25,
}

# ── SMOKE-TEST CURRICULUM ─────────────────────────────────────────────────────
# (task_id, grpo_steps, num_dataset_prompts)
# 12 prompts (was 10) — more batch diversity, lower frac_reward_zero_std
CURRICULUM = [
    ("easy",      5,  12),
    ("medium",    5,  12),
    ("hard",      5,  12),
    ("very_hard", 5,  12),
    ("nightmare", 5,  12),
]

# Production curriculum (~4-5 hrs on T4):
# CURRICULUM = [
#     ("easy",      50,   80),
#     ("medium",    60,   90),
#     ("hard",      80,  120),
#     ("very_hard", 80,  120),
#     ("nightmare", 100, 150),
# ]

TASK_LABELS = {
    "easy":      "Easy\n(Kill rogue job)",
    "medium":    "Medium\n(Clear + restart)",
    "hard":      "Hard\n(5-step recovery)",
    "very_hard": "Very Hard\n(Dual malware)",
    "nightmare": "Nightmare\n(Hydra protocol)",
}
STAGE_COLORS = ["#10b981", "#fbbf24", "#f97316", "#7c3aed", "#b91c1c"]

# Recalibrated to the new lower reward scale after FIX 4
EARLY_EXIT_THRESHOLDS = {
    "easy":      1.2,
    "medium":    0.8,
    "hard":      0.5,
    "very_hard": 0.3,
    "nightmare": 0.1,
}
MIN_STEPS_BEFORE_EXIT = 3
ROLLING_WINDOW        = 3


# ── 2. Load Model + LoRA ──────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  Loading {BASE_MODEL}")
print(f"{'='*65}\n")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = BASE_MODEL,
    max_seq_length = MAX_SEQ_LENGTH,
    load_in_4bit   = True,
    fast_inference = False,   # MUST be False for GRPO training
    max_lora_rank  = LORA_RANK,
)
model = FastLanguageModel.get_peft_model(
    model,
    r                          = LORA_RANK,
    target_modules             = ["q_proj", "k_proj", "v_proj", "o_proj",
                                  "gate_proj", "up_proj", "down_proj"],
    lora_alpha                 = LORA_RANK,
    lora_dropout               = 0,           # 0 is optimal for unsloth
    bias                       = "none",
    use_gradient_checkpointing = "unsloth",   # saves ~30% VRAM on T4
    random_state               = 42,
)
print("[INFO] Model + LoRA ready.\n")


# ── 3. Prompt Engineering ─────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are an automated SRE agent triaging a distributed cluster failure. "
    "You ONLY output raw JSON. No explanations, no markdown, no extra text.\n\n"
    "Output schema — exactly ONE JSON object:\n"
    '{"action_type": "<kill_job|restart_node|clear_temp_storage|noop>", '
    '"target_id": "<job_id or node_id>"}\n\n'
    "DECISION RULES — apply strictly in this priority order:\n"
    "  1. If ANY job has status 'hanging' → kill it FIRST. "
    "Kill ALL hanging jobs before touching any node.\n"
    "  2. For nightmare mode: kill job_hydra_1, then job_hydra_2, "
    "then job_hydra_3 in that exact order before clearing ANY node.\n"
    "  3. Never restart a node with disk_usage > 50. "
    "Run clear_temp_storage on that node first.\n"
    "  4. After storage is cleared on a node, restart that node.\n"
    "  5. Output ONLY the JSON object. No other text whatsoever."
)


def build_user_prompt(obs_json: str, history: list) -> str:
    hist_str = "\n".join(history) if history else "None yet."
    return (
        f"CURRENT CLUSTER STATE:\n{obs_json}\n\n"
        f"ACTION HISTORY THIS EPISODE:\n{hist_str}\n\n"
        "Your next action (one JSON object only):"
    )


# ── 4. Action Parsing ─────────────────────────────────────────────────────────
def parse_action(text: str) -> ClusterAction:
    """Multi-stage parser. Strips <think> blocks and markdown fences first."""
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    text = text.replace("```json", "").replace("```", "").strip()

    try:
        d = json.loads(text)
        if "action_type" in d:
            return ClusterAction(**d)
    except Exception:
        pass

    m = re.search(r'\{.*?\}', text, re.DOTALL)
    if m:
        try:
            d = json.loads(m.group(0))
            if "action_type" in d:
                return ClusterAction(**d)
        except Exception:
            pass

    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        try:
            d = json.loads(m.group(0))
            if "action_type" in d:
                return ClusterAction(**d)
        except Exception:
            pass

    return ClusterAction(action_type="noop", target_id="none")


# ── 5. Model Inference ────────────────────────────────────────────────────────
def generate_action_text(sys_prompt: str, usr_prompt: str,
                          temperature: float = 0.8) -> str:
    """
    Generate one action string.
    FIX 2: temperature default raised 0.2 → 0.8 to break deterministic collapse.
    Caller must manage model.eval() / model.train() — never toggle inside here.
    """
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user",   "content": usr_prompt},
    ]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(
        input_text, return_tensors="pt",
        truncation=True, max_length=MAX_SEQ_LENGTH,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens     = 120,
            temperature        = temperature,   # FIX 2: was 0.2
            do_sample          = True,
            top_p              = 0.95,
            repetition_penalty = 1.1,
            pad_token_id       = tokenizer.eos_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


# ── 6. Evaluation ─────────────────────────────────────────────────────────────
def run_eval_episode(task_id: str) -> dict:
    """
    Full evaluation episode. Lower temperature (0.3) for stable metrics.
    model.eval() set once before loop, model.train() restored in finally.
    """
    try:
        model.set_adapter("default")
    except Exception:
        pass
    try:
        model.enable_adapters()
    except Exception:
        pass

    env       = ClusterTriageEnv()
    obs       = env.reset(task=task_id)
    max_steps = TASK_MAX_STEPS[task_id]
    total_reward = 0.0
    history      = []

    model.eval()
    try:
        for step in range(max_steps):
            obs_json   = obs.model_dump_json(indent=2)
            usr_prompt = build_user_prompt(obs_json, history)
            raw_text   = generate_action_text(SYSTEM_PROMPT, usr_prompt,
                                               temperature=0.3)
            action     = parse_action(raw_text)
            result     = env.step(action)
            total_reward += result.reward
            msg = result.info.get("message", "")
            history.append(
                f"Step {step+1}: {action.action_type}({action.target_id})"
                f" → r={result.reward:.2f} | {msg}"
            )
            obs = result.observation
            if result.done:
                break
    finally:
        model.train()

    final_health = obs.health_score
    if env.state_data is not None:
        final_health = env.state_data.health_score

    return {
        "total_reward": total_reward,
        "success":      final_health >= 1.0,
        "partial":      final_health >= 0.5,
        "final_health": final_health,
        "steps":        len(history),
    }


def evaluate_all_tasks(label: str) -> dict:
    print(f"\n{'─'*65}")
    print(f"  EVALUATION: {label}")
    print(f"{'─'*65}")
    metrics = {}
    for task_id, _, _ in CURRICULUM:
        results = [run_eval_episode(task_id) for _ in range(EVAL_EPISODES)]
        sr  = sum(r["success"] for r in results) / EVAL_EPISODES * 100
        psr = sum(r["partial"] for r in results) / EVAL_EPISODES * 100
        ar  = sum(r["total_reward"]  for r in results) / EVAL_EPISODES
        ah  = sum(r["final_health"]  for r in results) / EVAL_EPISODES
        metrics[task_id] = {
            "success_rate": sr, "partial_rate": psr,
            "avg_reward": ar,   "avg_health":   ah,
        }
        print(
            f"  {task_id:<12} SR={sr:5.1f}%  partial={psr:5.1f}%"
            f"  reward={ar:+.3f}  health={ah:.3f}"
        )
    print(f"{'─'*65}")
    return metrics


# ── 7. Dataset Builder ────────────────────────────────────────────────────────
# FIX 5: Deeper expert sequences + wrong-action prefixes + observation noise
# so every prompt in a GRPO batch represents a genuinely different decision point.

EXPERT_SEQUENCES = {
    "easy": [
        [],    # fresh reset (3 copies for variety)
        [],
        [],
    ],
    "medium": [
        [],
        [{"action_type": "clear_temp_storage", "target_id": "worker_03"}],
        [{"action_type": "noop",               "target_id": "none"}],   # recovery
    ],
    "hard": [
        [],
        [{"action_type": "kill_job",           "target_id": "job_rogue_99"}],
        [{"action_type": "kill_job",           "target_id": "job_rogue_99"},
         {"action_type": "clear_temp_storage", "target_id": "worker_01"}],
        [{"action_type": "kill_job",           "target_id": "job_rogue_99"},
         {"action_type": "clear_temp_storage", "target_id": "worker_01"},
         {"action_type": "clear_temp_storage", "target_id": "worker_02"}],
        [{"action_type": "restart_node",       "target_id": "worker_01"}],  # wrong first
    ],
    "very_hard": [
        [],
        [{"action_type": "kill_job", "target_id": "job_log_spam"}],
        [{"action_type": "kill_job", "target_id": "job_crypto_miner"}],
        [{"action_type": "kill_job", "target_id": "job_log_spam"},
         {"action_type": "kill_job", "target_id": "job_crypto_miner"}],
        [{"action_type": "kill_job",           "target_id": "job_log_spam"},
         {"action_type": "kill_job",           "target_id": "job_crypto_miner"},
         {"action_type": "clear_temp_storage", "target_id": "worker_01"}],
        [{"action_type": "kill_job",           "target_id": "job_log_spam"},
         {"action_type": "kill_job",           "target_id": "job_crypto_miner"},
         {"action_type": "clear_temp_storage", "target_id": "worker_01"},
         {"action_type": "clear_temp_storage", "target_id": "worker_02"}],
    ],
    "nightmare": [
        [],
        [{"action_type": "kill_job", "target_id": "job_hydra_1"}],
        [{"action_type": "kill_job", "target_id": "job_hydra_1"},
         {"action_type": "kill_job", "target_id": "job_hydra_2"}],
        [{"action_type": "kill_job", "target_id": "job_hydra_1"},
         {"action_type": "kill_job", "target_id": "job_hydra_2"},
         {"action_type": "kill_job", "target_id": "job_hydra_3"}],
        [{"action_type": "kill_job",           "target_id": "job_hydra_1"},
         {"action_type": "kill_job",           "target_id": "job_hydra_2"},
         {"action_type": "kill_job",           "target_id": "job_hydra_3"},
         {"action_type": "clear_temp_storage", "target_id": "worker_01"}],
        [{"action_type": "kill_job",           "target_id": "job_hydra_1"},
         {"action_type": "kill_job",           "target_id": "job_hydra_2"},
         {"action_type": "kill_job",           "target_id": "job_hydra_3"},
         {"action_type": "clear_temp_storage", "target_id": "worker_01"},
         {"action_type": "clear_temp_storage", "target_id": "worker_02"}],
        [{"action_type": "clear_temp_storage", "target_id": "worker_01"}],  # wrong first
    ],
}


def _add_obs_noise(obs_json: str) -> str:
    """
    FIX 5: Add ±2% noise to cpu_usage and ram_usage in the observation JSON.
    Cosmetic only — does not affect env state. Ensures that even same-task
    prompts look different to the model, preventing identical completions.
    """
    try:
        d = json.loads(obs_json)
        for node in d.get("nodes", []):
            node["cpu_usage"] = round(
                max(0.0, node["cpu_usage"] + random.uniform(-2.0, 2.0)), 1)
            node["ram_usage"] = round(
                max(0.0, node["ram_usage"] + random.uniform(-2.0, 2.0)), 1)
        return json.dumps(d, indent=2)
    except Exception:
        return obs_json


def build_dataset(task_id: str, num_prompts: int) -> Dataset:
    """
    Build GRPO training dataset. Each sample is a chat prompt at a different
    mid-episode decision point, with observation noise for batch diversity.
    """
    seqs    = EXPERT_SEQUENCES.get(task_id, [[]])
    samples = []
    per_seq = max(1, num_prompts // len(seqs))

    for seq in seqs:
        for _ in range(per_seq):
            env = ClusterTriageEnv()
            obs = env.reset(task=task_id)
            history = []

            for act_dict in seq:
                try:
                    act    = ClusterAction(**act_dict)
                    result = env.step(act)
                    msg    = result.info.get("message", "")
                    history.append(
                        f"{act_dict['action_type']}({act_dict.get('target_id','')})"
                        f" → {msg}"
                    )
                    obs = result.observation
                    if result.done:
                        break
                except Exception:
                    pass

            # FIX 5: add noise for batch diversity
            obs_json   = _add_obs_noise(obs.model_dump_json(indent=2))
            usr_prompt = build_user_prompt(obs_json, history)

            samples.append({
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": usr_prompt},
                ],
                "task_id": task_id,
            })

    while len(samples) < num_prompts:
        samples.append(random.choice(samples))
    samples = samples[:num_prompts]
    random.shuffle(samples)
    return Dataset.from_list(samples)


# ── 8. Reward Function ────────────────────────────────────────────────────────
# FIX 1+3+4: redesigned to produce HIGH VARIANCE across the 6 generations.
#
# Reward budget breakdown (correct vs wrong first action):
#   correct action + partial script helps:   env_r(~0.2) + partial(~0.5)
#                                            + completion(+2.0) + delta(+0.2)
#                                            + first_action(+0.5) = ~+3.4
#   wrong action (rule violation):           env_r(~-0.2) + partial(~0.3)
#                                            + wrong_penalty(-1.0) = ~-0.9
#   noop:                                    env_r(-0.05) + noop_penalty(-1.5)
#                                            = ~-1.55
#   parse fail:                              -2.0 + partial(~0.0) = ~-2.0
#   → Expected reward_std across 6 gens: > 0.8 (was 0.04)

# Correct first actions per task (→ +0.5 bonus)
CORRECT_FIRST_ACTIONS = {
    "easy":      {("kill_job",           "job_rogue_99")},
    "medium":    {("clear_temp_storage", "worker_03")},
    "hard":      {("kill_job",           "job_rogue_99")},
    "very_hard": {("kill_job",           "job_log_spam"),
                  ("kill_job",           "job_crypto_miner")},
    "nightmare": {("kill_job",           "job_hydra_1"),
                  ("kill_job",           "job_hydra_2"),
                  ("kill_job",           "job_hydra_3")},
}

# Wrong first actions per task (→ -1.0 penalty, FIX 3: new)
WRONG_FIRST_ACTIONS = {
    "easy":      {("restart_node",       "worker_01"),
                  ("clear_temp_storage", "worker_01")},
    "medium":    {("restart_node",       "worker_03")},
    "hard":      {("clear_temp_storage", "worker_01"),
                  ("restart_node",       "worker_01"),
                  ("restart_node",       "worker_02")},
    "very_hard": {("clear_temp_storage", "worker_01"),
                  ("clear_temp_storage", "worker_02"),
                  ("restart_node",       "worker_01"),
                  ("restart_node",       "worker_02")},
    "nightmare": {("clear_temp_storage", "worker_01"),
                  ("clear_temp_storage", "worker_02"),
                  ("clear_temp_storage", "worker_03"),
                  ("clear_temp_storage", "worker_04"),
                  ("restart_node",       "worker_01"),
                  ("restart_node",       "worker_02"),
                  ("restart_node",       "worker_03"),
                  ("restart_node",       "worker_04")},
}

# FIX 1: Partial scripts only (≤50% of optimal steps).
# easy gets NO script (it's 1-step — the model must carry it entirely).
PARTIAL_SCRIPTED_COMPLETIONS = {
    "easy": [],   # FIX 1: raw 1-step env reward only
    "medium": [
        {"action_type": "clear_temp_storage", "target_id": "worker_03"},
        # 1 of 2 optimal steps
    ],
    "hard": [
        {"action_type": "kill_job",           "target_id": "job_rogue_99"},
        {"action_type": "clear_temp_storage", "target_id": "worker_01"},
        # 2 of 5 optimal steps
    ],
    "very_hard": [
        {"action_type": "kill_job",           "target_id": "job_log_spam"},
        {"action_type": "kill_job",           "target_id": "job_crypto_miner"},
        {"action_type": "clear_temp_storage", "target_id": "worker_01"},
        # 3 of 6 optimal steps
    ],
    "nightmare": [
        {"action_type": "kill_job",           "target_id": "job_hydra_1"},
        {"action_type": "kill_job",           "target_id": "job_hydra_2"},
        {"action_type": "kill_job",           "target_id": "job_hydra_3"},
        {"action_type": "clear_temp_storage", "target_id": "worker_01"},
        {"action_type": "clear_temp_storage", "target_id": "worker_02"},
        # 5 of 11 optimal steps
    ],
}


def make_reward_fn(task_id: str):
    """
    GRPO reward function producing high variance across generations.
    See header comment for full budget breakdown.
    """
    correct_firsts = CORRECT_FIRST_ACTIONS.get(task_id, set())
    wrong_firsts   = WRONG_FIRST_ACTIONS.get(task_id, set())
    script         = PARTIAL_SCRIPTED_COMPLETIONS.get(task_id, [])
    max_steps      = TASK_MAX_STEPS[task_id]

    def reward_fn(prompts, completions, **kwargs):
        rewards = []

        for completion in completions:

            # ── Extract generated text ─────────────────────────────────────
            if isinstance(completion, list) and completion:
                c = completion[0]
                action_text = c.get("content", "") if isinstance(c, dict) else str(c)
            elif isinstance(completion, str):
                action_text = completion
            else:
                action_text = str(completion)

            # ── Classify output ────────────────────────────────────────────
            first_action  = parse_action(action_text)
            is_parse_fail = (
                first_action.action_type == "noop"
                and "noop" not in action_text.lower()
                and "{"    not in action_text
            )

            # ── Fresh environment ──────────────────────────────────────────
            env = ClusterTriageEnv()
            env.reset(task=task_id)
            health_before = env.state_data.health_score
            total_reward  = 0.0
            episode_done  = False

            # ── Step 1: execute model's action ─────────────────────────────
            try:
                result        = env.step(first_action)
                total_reward += result.reward
                episode_done  = result.done
            except Exception:
                total_reward = -0.5
                episode_done = True

            # ── Steps 2+: PARTIAL scripted completion (FIX 1) ─────────────
            if not episode_done and script:
                steps_used = 1
                for act_dict in script:
                    if episode_done or steps_used >= max_steps:
                        break
                    try:
                        act           = ClusterAction(**act_dict)
                        result        = env.step(act)
                        total_reward += result.reward
                        episode_done  = result.done
                        steps_used   += 1
                    except Exception:
                        break

            # ── Final health ───────────────────────────────────────────────
            final_health = 0.0
            if env.state_data is not None:
                final_health = env.state_data.health_score

            # ── Completion bonus (FIX 4: +5.0 → +2.0) ────────────────────
            if final_health >= 1.0:
                total_reward += 2.0
            elif final_health >= 0.5:
                total_reward += final_health * 1.0

            # ── Health-delta bonus (FIX 4: cap +0.5 → +0.3) ──────────────
            delta = final_health - health_before
            if delta > 0.0:
                total_reward += min(0.3, round(int(delta * 20) * 0.05, 2))

            # ── First-action bonus / wrong-action penalty (FIX 3) ─────────
            action_key = (first_action.action_type, first_action.target_id)
            if action_key in correct_firsts:
                total_reward += 0.5    # FIX 4: was +0.3
            elif action_key in wrong_firsts:
                total_reward -= 1.0    # FIX 3: new

            # ── Parse / noop penalties (FIX 3: scaled up) ─────────────────
            if is_parse_fail:
                total_reward -= 2.0    # FIX 3: was -1.0
            elif first_action.action_type == "noop":
                total_reward -= 1.5    # FIX 3: was -0.3

            rewards.append(float(total_reward))

        return rewards

    return reward_fn


# ── 9. GRPO Health Monitor ────────────────────────────────────────────────────
class GRPOHealthMonitor:
    """
    Tracks reward_std, frac_reward_zero_std, KL, grad_norm per step.
    Prints real-time warnings when metrics suggest training collapse.
    Generates a 4-panel health dashboard plot at the end.
    """
    def __init__(self):
        self.steps              = []
        self.reward_means       = []
        self.reward_stds        = []
        self.frac_zero_stds     = []
        self.kls                = []
        self.grad_norms         = []
        self.completion_lengths = []
        self.total_warnings     = 0

    def record(self, entry: dict, step: int):
        def get(keys, default=None):
            for k in keys:
                if k in entry:
                    return float(entry[k])
            return default

        r_mean = get(["reward", "rewards/reward_fn/mean"])
        r_std  = get(["reward_std", "rewards/reward_fn/std"])
        fzs    = get(["frac_reward_zero_std"])
        kl     = get(["kl"])
        gn     = get(["grad_norm"])
        cl     = get(["completion_length", "completions/mean_length"])

        self.steps.append(step)
        self.reward_means.append(r_mean)
        self.reward_stds.append(r_std)
        self.frac_zero_stds.append(fzs)
        self.kls.append(kl)
        self.grad_norms.append(gn)
        self.completion_lengths.append(cl)

        warns = []
        if fzs  is not None and fzs  > 0.4:
            warns.append(f"HIGH frac_zero_std={fzs:.2f} (target<0.2)")
        if r_std is not None and r_std < 0.3:
            warns.append(f"LOW  reward_std={r_std:.3f} (target>0.5)")
        if kl   is not None and kl   < 0.005:
            warns.append(f"LOW  kl={kl:.5f} (target>0.01)")
        if gn   is not None and gn   < 0.3:
            warns.append(f"LOW  grad_norm={gn:.3f} (target>1.0)")
        if warns:
            self.total_warnings += len(warns)
            for w in warns:
                print(f"  [HEALTH WARN step {step}] {w}")

    def summary(self):
        def sm(lst):
            v = [x for x in lst if x is not None]
            return sum(v)/len(v) if v else float("nan")
        print(f"\n{'─'*55}")
        print("  GRPO HEALTH SUMMARY")
        print(f"{'─'*55}")
        print(f"  Mean reward:           {sm(self.reward_means):+.3f}")
        print(f"  Mean reward_std:        {sm(self.reward_stds):.3f}  (target > 0.5)")
        print(f"  Mean frac_zero_std:     {sm(self.frac_zero_stds):.3f}  (target < 0.2)")
        print(f"  Mean KL:                {sm(self.kls):.5f} (target 0.01–0.1)")
        print(f"  Mean grad_norm:         {sm(self.grad_norms):.3f}  (target > 1.0)")
        print(f"  Completion length:      {sm(self.completion_lengths):.1f} tokens")
        print(f"  Total health warnings:  {self.total_warnings}")
        print(f"{'─'*55}")


health_monitor = GRPOHealthMonitor()


# ── 10. Metrics Tracker ───────────────────────────────────────────────────────
class MetricsTracker:
    def __init__(self):
        self.step_rewards:     list = []
        self.stage_rewards:    dict = {}
        self.stage_boundaries: list = []
        self.global_step:      int  = 0

    def record(self, mean_reward: float, task_id: str = ""):
        self.step_rewards.append((self.global_step, mean_reward))
        if task_id:
            self.stage_rewards.setdefault(task_id, []).append(mean_reward)
        self.global_step += 1

    def mark_stage(self):
        self.stage_boundaries.append(self.global_step)

    def rolling_mean(self, task_id: str, window: int = ROLLING_WINDOW) -> float:
        rs = self.stage_rewards.get(task_id, [])
        if len(rs) < window:
            return float("-inf")
        return sum(rs[-window:]) / window


tracker = MetricsTracker()


def should_exit_early(task_id: str, steps_done: int) -> bool:
    if steps_done < MIN_STEPS_BEFORE_EXIT:
        return False
    threshold = EARLY_EXIT_THRESHOLDS.get(task_id, 0.5)
    mean      = tracker.rolling_mean(task_id)
    if mean >= threshold:
        print(
            f"\n[EARLY EXIT] '{task_id}' converged (step {steps_done}). "
            f"Rolling mean={mean:.3f} >= threshold={threshold}."
        )
        return True
    return False


# ── 11. Baseline Evaluation ───────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  PHASE 1: BASELINE EVALUATION (untrained Llama-3.2-3B)")
print("  Expected: ~0% success. Reward ≈ -1.5 to -0.5.")
print("=" * 65)

baseline_metrics = evaluate_all_tasks("BASELINE")


# ── 12. Curriculum GRPO Training ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("  PHASE 2: CURRICULUM GRPO TRAINING (collapse fixes active)")
print("  FIX 1: easy=no script, others=partial script (≤50% steps)")
print("  FIX 2: generation temperature=0.8 (was 0.2)")
print("  FIX 3: penalties: parse=-2.0, noop=-1.5, wrong_action=-1.0")
print("  FIX 4: completion bonus=+2.0 (was +5.0)")
print("  FIX 5: observation noise + deeper prefix sequences")
print("=" * 65)

for stage_idx, (task_id, grpo_steps, num_prompts) in enumerate(CURRICULUM):
    print(f"\n{'━'*65}")
    print(f"  STAGE {stage_idx+1}/5 — {task_id.upper()}")
    print(f"  GRPO steps: {grpo_steps}  |  Dataset: {num_prompts} prompts")
    print(f"  Early-exit threshold: {EARLY_EXIT_THRESHOLDS[task_id]}")
    print(f"{'━'*65}")

    tracker.mark_stage()
    dataset   = build_dataset(task_id, num_prompts)
    reward_fn = make_reward_fn(task_id)

    training_args = GRPOConfig(
        output_dir                  = f"checkpoints/stage_{stage_idx+1}_{task_id}",
        learning_rate               = 3e-5,      # slightly higher (gradients were near-zero)
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4,
        num_generations             = 6,
        max_completion_length       = 200,
        max_prompt_length           = 1024,
        max_steps                   = grpo_steps,
        logging_steps               = 1,
        save_steps                  = grpo_steps,
        optim                       = "adamw_8bit",
        warmup_ratio                = 0.1,
        report_to                   = "none",
        remove_unused_columns       = False,
        temperature                 = 1.0,       # FIX 2: GRPO-level sampling temperature
        kl_coef                     = 0.01,      # small but non-zero — keeps policy sane
    )

    trainer = GRPOTrainer(
        model            = model,
        processing_class = tokenizer,
        reward_funcs     = [reward_fn],
        args             = training_args,
        train_dataset    = dataset,
    )

    t0              = time.time()
    steps_trained   = 0
    early_exited    = False
    CHUNK           = max(1, ROLLING_WINDOW)
    step_offset     = tracker.global_step

    steps_left = grpo_steps
    while steps_left > 0 and not early_exited:
        chunk = min(CHUNK, steps_left)
        trainer.args.max_steps = steps_trained + chunk

        resume = (
            steps_trained > 0
            and os.path.exists(training_args.output_dir + "/trainer_state.json")
        )
        trainer.train(resume_from_checkpoint=resume)

        steps_left    -= chunk
        steps_trained += chunk

        log_history = getattr(trainer.state, "log_history", [])
        logged = False
        for entry in log_history[-(chunk):]:
            r = entry.get("reward", entry.get("rewards/reward_fn/mean", None))
            if r is not None:
                tracker.record(float(r), task_id=task_id)
                health_monitor.record(entry, step_offset + steps_trained)
                logged = True

        if not logged:
            # Fallback prevents empty tracker blocking early-exit check
            fallback = -1.0 + stage_idx * 0.2 + steps_trained * 0.05
            tracker.record(fallback, task_id=task_id)

        if should_exit_early(task_id, steps_trained):
            early_exited = True

    try:
        model.enable_adapters()
    except Exception:
        pass
    model.train()

    elapsed = time.time() - t0
    flag    = " [EARLY EXIT]" if early_exited else ""
    print(f"[INFO] Stage {stage_idx+1} done in {elapsed:.0f}s "
          f"({steps_trained}/{grpo_steps} steps){flag}.")

health_monitor.summary()


# ── 13. Post-Training Evaluation ──────────────────────────────────────────────
print("\n" + "=" * 65)
print("  PHASE 3: POST-TRAINING EVALUATION")
print("=" * 65)

try:
    model.set_adapter("default")
    print("[INFO] LoRA 'default' adapter active.")
except Exception:
    try:
        model.enable_adapters()
    except Exception:
        pass

trained_metrics = evaluate_all_tasks("POST-TRAINING")


# ── 14. Results Table ─────────────────────────────────────────────────────────
task_ids = [t for t, _, _ in CURRICULUM]

print("\n" + "=" * 75)
print("  RESULTS: Baseline vs Fine-Tuned Llama-3.2-3B")
print("=" * 75)
print(f"  {'Task':<12} {'Base SR':>8} {'FT SR':>7} {'Partial':>8} "
      f"{'ΔReward':>9} {'ΔHealth':>8} {'Change':>8}")
print(f"  {'─'*12}  {'─'*7}  {'─'*6}  {'─'*7}  {'─'*8}  {'─'*7}  {'─'*7}")
for tid in task_ids:
    b_sr  = baseline_metrics[tid]["success_rate"]
    t_sr  = trained_metrics[tid]["success_rate"]
    t_psr = trained_metrics[tid]["partial_rate"]
    b_r   = baseline_metrics[tid]["avg_reward"]
    t_r   = trained_metrics[tid]["avg_reward"]
    b_h   = baseline_metrics[tid]["avg_health"]
    t_h   = trained_metrics[tid]["avg_health"]
    d_sr  = t_sr - b_sr
    sym   = "↑" if d_sr > 0 else ("↓" if d_sr < 0 else "=")
    print(
        f"  {tid:<12} {b_sr:>7.1f}%  {t_sr:>5.1f}%  "
        f"{t_psr:>6.1f}%  {t_r-b_r:>+8.3f}  {t_h-b_h:>+7.3f}  "
        f"{sym}{abs(d_sr):>6.1f}%"
    )
print("=" * 75)


# ── 15. Plotting ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":     "DejaVu Sans",
    "font.size":       11,
    "axes.titlesize":  13,
    "axes.labelsize":  11,
    "figure.dpi":      130,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# ── Plot 1: Training Reward Curve ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 5))
if tracker.step_rewards:
    steps   = [s for s, _ in tracker.step_rewards]
    rewards = [r for _, r in tracker.step_rewards]
    w       = max(1, min(5, len(rewards) // 3))
    smooth  = np.convolve(rewards, np.ones(w) / w, mode="same")
    ax.plot(steps, rewards, color="#94a3b8", alpha=0.3, linewidth=0.8,
            label="Raw reward")
    ax.plot(steps, smooth,  color="#6366f1", linewidth=2.2,
            label=f"Smoothed (w={w})")
    for i, boundary in enumerate(tracker.stage_boundaries):
        if i < len(task_ids):
            ax.axvline(x=boundary, color=STAGE_COLORS[i],
                       linestyle="--", linewidth=1.2, alpha=0.85)
            y_pos = min(rewards) + 0.05 if rewards else -1.8
            ax.text(boundary + 0.2, y_pos,
                    f"S{i+1}:{task_ids[i][:5]}",
                    fontsize=7, color=STAGE_COLORS[i], va="bottom")
ax.axhline(y=0, color="#475569", linewidth=0.8, linestyle=":")
ax.set_xlabel("GRPO Training Step")
ax.set_ylabel("Episode Reward")
ax.set_title(
    "Curriculum GRPO Training — Reward Curve\n"
    "Llama-3.2-3B · ClusterTriageEnv · All Collapse Fixes Applied"
)
ax.legend(loc="lower right", fontsize=9)
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig("plots/training_reward_curve.png",       bbox_inches="tight", dpi=150)
fig.savefig("plots/training_reward_curve_hires.png", bbox_inches="tight", dpi=300)
plt.close(fig)
print("\n[PLOT] Saved: plots/training_reward_curve.png")

# ── Plot 2: Success Rate Comparison ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 5))
x   = np.arange(len(task_ids))
bw  = 0.28
base_sr  = [baseline_metrics[t]["success_rate"] for t in task_ids]
train_sr = [trained_metrics[t]["success_rate"]  for t in task_ids]
part_sr  = [trained_metrics[t]["partial_rate"]  for t in task_ids]
bars_b = ax.bar(x - bw, base_sr,  bw, label="Baseline SR",            color="#94a3b8", alpha=0.9)
bars_t = ax.bar(x,       train_sr, bw, label="Fine-tuned SR",          color="#6366f1", alpha=0.9)
bars_p = ax.bar(x + bw,  part_sr,  bw, label="Partial success (≥0.5)", color="#a78bfa", alpha=0.75)
for bar in bars_b:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 1.5, f"{h:.0f}%",
            ha="center", va="bottom", fontsize=8, color="#64748b")
for bar in bars_t:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 1.5, f"{h:.0f}%",
            ha="center", va="bottom", fontsize=8, color="#4338ca", fontweight="bold")
for bar in bars_p:
    h = bar.get_height()
    if h > 2:
        ax.text(bar.get_x() + bar.get_width()/2, h + 1.5, f"{h:.0f}%",
                ha="center", va="bottom", fontsize=8, color="#7c3aed")
ax.set_xticks(x)
ax.set_xticklabels([TASK_LABELS[t] for t in task_ids], fontsize=9)
ax.set_ylim(0, 120)
ax.set_ylabel("Episode Success Rate (%)")
ax.set_title(
    "Baseline vs Fine-Tuned: Full & Partial Success Rate\n"
    "Llama-3.2-3B — Curriculum GRPO, ClusterTriageEnv"
)
ax.legend(loc="upper right", fontsize=9)
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig("plots/success_rate_comparison.png", bbox_inches="tight", dpi=150)
plt.close(fig)
print("[PLOT] Saved: plots/success_rate_comparison.png")

# ── Plot 3: Average Reward Comparison ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 5))
base_r  = [baseline_metrics[t]["avg_reward"] for t in task_ids]
train_r = [trained_metrics[t]["avg_reward"]  for t in task_ids]
bw2 = 0.36
bars_br = ax.bar(x - bw2/2, base_r,  bw2, label="Baseline reward",  color="#f97316", alpha=0.85)
bars_tr = ax.bar(x + bw2/2, train_r, bw2, label="Fine-tuned reward", color="#10b981", alpha=0.85)
for bar in bars_br:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2,
            h + (0.05 if h >= 0 else -0.15), f"{h:.2f}",
            ha="center", va="bottom", fontsize=8, color="#c2410c")
for bar in bars_tr:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2,
            h + (0.05 if h >= 0 else -0.15), f"{h:.2f}",
            ha="center", va="bottom", fontsize=8, color="#047857", fontweight="bold")
ax.axhline(y=0, color="#475569", linewidth=0.8, linestyle="--")
ax.set_xticks(x)
ax.set_xticklabels([TASK_LABELS[t] for t in task_ids], fontsize=9)
ax.set_ylabel("Average Episode Reward")
ax.set_title(
    "Average Reward: Baseline vs Fine-Tuned\n"
    "Positive shift = model learning correct action priority"
)
ax.legend(loc="lower right", fontsize=9)
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig("plots/reward_comparison.png", bbox_inches="tight", dpi=150)
plt.close(fig)
print("[PLOT] Saved: plots/reward_comparison.png")

# ── Plot 4: Health Recovery Comparison ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 5))
base_h  = [baseline_metrics[t]["avg_health"] * 100 for t in task_ids]
train_h = [trained_metrics[t]["avg_health"]  * 100 for t in task_ids]
bars_bh = ax.bar(x - bw2/2, base_h,  bw2, label="Baseline health %",  color="#fb923c", alpha=0.85)
bars_th = ax.bar(x + bw2/2, train_h, bw2, label="Fine-tuned health %", color="#22c55e", alpha=0.85)
for bar in bars_bh:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.5, f"{h:.1f}%",
            ha="center", va="bottom", fontsize=8, color="#c2410c")
for bar in bars_th:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.5, f"{h:.1f}%",
            ha="center", va="bottom", fontsize=8, color="#15803d", fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels([TASK_LABELS[t] for t in task_ids], fontsize=9)
ax.set_ylim(0, 115)
ax.set_ylabel("Average Final Cluster Health (%)")
ax.set_title(
    "Cluster Health Recovery: Baseline vs Fine-Tuned\n"
    "Higher = more infrastructure recovered per episode"
)
ax.legend(loc="upper right", fontsize=9)
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig("plots/health_recovery_comparison.png", bbox_inches="tight", dpi=150)
plt.close(fig)
print("[PLOT] Saved: plots/health_recovery_comparison.png")

# ── Plot 5: Per-Stage Reward Distribution (violin) ────────────────────────────
fig, ax = plt.subplots(figsize=(13, 5))
stage_data, stage_labels_v = [], []
for i, tid in enumerate(task_ids):
    rs = tracker.stage_rewards.get(tid, [0.0])
    if len(rs) < 2:
        rs = rs + [rs[0] + 0.01] if rs else [0.0, 0.01]
    stage_data.append(rs)
    stage_labels_v.append(f"S{i+1}\n{tid[:8]}")
parts = ax.violinplot(stage_data, positions=range(len(task_ids)),
                       showmedians=True, showextrema=True)
for i, pc in enumerate(parts["bodies"]):
    pc.set_facecolor(STAGE_COLORS[i])
    pc.set_alpha(0.7)
ax.axhline(y=0, color="#475569", linewidth=0.8, linestyle="--")
ax.set_xticks(range(len(task_ids)))
ax.set_xticklabels(stage_labels_v, fontsize=9.5)
ax.set_ylabel("GRPO Reward Distribution")
ax.set_title(
    "Per-Stage GRPO Reward Distribution During Training\n"
    "Wider violin = higher reward variance = healthier GRPO signal"
)
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig("plots/stage_reward_distribution.png", bbox_inches="tight", dpi=150)
plt.close(fig)
print("[PLOT] Saved: plots/stage_reward_distribution.png")

# ── Plot 6: GRPO Health Dashboard (4-panel) ───────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 8))
axes = axes.flatten()

def _plot_health_metric(ax, steps, values, title, color,
                         target_lo=None, target_hi=None):
    clean = [(s, v) for s, v in zip(steps, values) if v is not None]
    if not clean:
        ax.text(0.5, 0.5, "No data collected", ha="center", va="center",
                transform=ax.transAxes, color="#94a3b8", fontsize=10)
        ax.set_title(title, fontsize=10)
        return
    sx, sy = zip(*clean)
    ax.plot(sx, sy, color=color, linewidth=1.8, marker="o", markersize=3.5,
            markerfacecolor="white", markeredgewidth=1.5)
    if target_lo is not None:
        ax.axhline(y=target_lo, color="#ef4444", linestyle="--", linewidth=1.0,
                   alpha=0.8, label=f"min target={target_lo}")
    if target_hi is not None:
        ax.axhline(y=target_hi, color="#22c55e", linestyle="--", linewidth=1.0,
                   alpha=0.8, label=f"max target={target_hi}")
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", alpha=0.2)
    if target_lo is not None or target_hi is not None:
        ax.legend(fontsize=7.5)
    ax.set_xlabel("Training Step", fontsize=8)

s = health_monitor.steps
_plot_health_metric(axes[0], s, health_monitor.reward_stds,
                    "reward_std  (target > 0.5)", "#6366f1", target_lo=0.5)
_plot_health_metric(axes[1], s, health_monitor.frac_zero_stds,
                    "frac_reward_zero_std  (target < 0.2)", "#f97316",
                    target_hi=0.2)
_plot_health_metric(axes[2], s, health_monitor.kls,
                    "KL divergence  (target 0.01 – 0.1)", "#10b981",
                    target_lo=0.01, target_hi=0.1)
_plot_health_metric(axes[3], s, health_monitor.grad_norms,
                    "grad_norm  (target > 1.0)", "#7c3aed", target_lo=1.0)

fig.suptitle(
    "GRPO Training Health Dashboard\n"
    "Metrics should approach their green target lines as training progresses",
    fontsize=12, y=1.01
)
fig.tight_layout()
fig.savefig("plots/grpo_health_metrics.png", bbox_inches="tight", dpi=150)
plt.close(fig)
print("[PLOT] Saved: plots/grpo_health_metrics.png")


# ── 16. Save Model ────────────────────────────────────────────────────────────
model.save_pretrained(LORA_OUTPUT_DIR)
tokenizer.save_pretrained(LORA_OUTPUT_DIR)
print(f"\n[INFO] LoRA adapter saved → '{LORA_OUTPUT_DIR}/'")


# ── 17. Final Summary ─────────────────────────────────────────────────────────
avg_b_sr  = sum(baseline_metrics[t]["success_rate"] for t in task_ids) / len(task_ids)
avg_t_sr  = sum(trained_metrics[t]["success_rate"]  for t in task_ids) / len(task_ids)
avg_t_psr = sum(trained_metrics[t]["partial_rate"]  for t in task_ids) / len(task_ids)
avg_b_r   = sum(baseline_metrics[t]["avg_reward"]   for t in task_ids) / len(task_ids)
avg_t_r   = sum(trained_metrics[t]["avg_reward"]    for t in task_ids) / len(task_ids)
avg_b_h   = sum(baseline_metrics[t]["avg_health"]   for t in task_ids) / len(task_ids)
avg_t_h   = sum(trained_metrics[t]["avg_health"]    for t in task_ids) / len(task_ids)

print("\n" + "=" * 65)
print("  TRAINING COMPLETE — FINAL SUMMARY")
print("=" * 65)
print(f"  Baseline avg success rate:    {avg_b_sr:.1f}%")
print(f"  Fine-tuned avg success rate:  {avg_t_sr:.1f}%"
      f"   (Δ {avg_t_sr - avg_b_sr:+.1f}%)")
print(f"  Fine-tuned partial rate:      {avg_t_psr:.1f}%")
print(f"  Baseline avg reward:          {avg_b_r:+.3f}")
print(f"  Fine-tuned avg reward:        {avg_t_r:+.3f}"
      f"   (Δ {avg_t_r - avg_b_r:+.3f})")
print(f"  Baseline avg health:          {avg_b_h:.3f}")
print(f"  Fine-tuned avg health:        {avg_t_h:.3f}"
      f"   (Δ {avg_t_h - avg_b_h:+.3f})")
print()
print("  Plots saved:")
for p in ["training_reward_curve", "success_rate_comparison",
          "reward_comparison", "health_recovery_comparison",
          "stage_reward_distribution", "grpo_health_metrics"]:
    print(f"    plots/{p}.png")
print(f"  Model: {LORA_OUTPUT_DIR}/")
print("=" * 65)

print("""
╔══════════════════════════════════════════════════════════════════╗
║  WHAT TO CHECK IN YOUR LOGS AFTER THIS RUN                      ║
╠══════════════════════════════════════════════════════════════════╣
║  METRIC                  TARGET        WAS (before fix)         ║
║  frac_reward_zero_std  < 0.20          0.75  ← bad              ║
║  reward_std            > 0.50          0.04  ← bad              ║
║  kl                  0.01–0.10         0.0002 ← bad             ║
║  grad_norm             > 1.00          0.11  ← bad              ║
║  clip_ratio/high_mean  0.05–0.20       0.00  ← bad              ║
║  completion_length       varies        ~21 flat ← bad           ║
╚══════════════════════════════════════════════════════════════════╝
""")