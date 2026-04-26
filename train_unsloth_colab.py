"""
train_unsloth_colab.py — Curriculum GRPO Training on ClusterTriageEnv
===============================================================================
Trains Llama-3.2-3B-Instruct via GRPO (Group Relative Policy Optimization)
against the live ClusterTriageEnv reward function using curriculum learning.

Environment: 5 tasks (easy → medium → hard → very_hard → nightmare)
  - easy:       Kill 1 rogue job
  - medium:     Clear disk → restart node (2-step sequence)
  - hard:       Kill job → clear 2 nodes → restart 2 nodes (5-step)
  - very_hard:  Kill 2 malware jobs → clear 2 nodes → restart 2 nodes
  - nightmare:  Kill 3 hydra jobs → clear 4 nodes → restart 4 nodes

═══════════════════════════════════════════════════════════════════════════════
  ROOT CAUSES OF PREVIOUS TRAINING COLLAPSE (and fixes applied here)
═══════════════════════════════════════════════════════════════════════════════

BUG 1 — PROMPT-REWARD STATE MISMATCH (PRIMARY CAUSE of medium 40%→0%, nightmare 80%→0%)
  Problem: Dataset builds prompts from mid-episode states (after expert prefixes),
           but reward_fn ALWAYS resets the environment fresh before evaluating.
           Example: a prompt showing "disk cleared, node offline" leads the model
           to output "restart_node worker_03" (correct for that state). But the
           reward_fn evaluates restart_node on a FRESH env where disk is still 99.9%
           → reward = -0.3 (PENALTY). The model gets punished for the CORRECT action.
           This inverted signal is the direct cause of medium SR collapsing to 0%.
           The nightmare collapse (80%→0%) has the same mechanism for later-stage prompts.
  Fix:     reward_fn now replays the expert prefix sequence BEFORE evaluating the
           model's action, so the reward context exactly matches the prompt context.
           Each sample stores its prefix_actions alongside its prompt.

BUG 2 — WRONG_FIRST_ACTIONS is context-blind (amplified BUG 1 for medium)
  Problem: WRONG_FIRST_ACTIONS penalises restart_node globally for medium, but
           restart_node IS the correct first action when the prompt is from the
           post-clear state (sequence 2). Combined with BUG 1, the model received
           both an env penalty (-0.3) AND a wrong_action penalty (-1.0) = -1.3 for
           the optimal action in that context. This aggressively trained the model
           AWAY from restart_node, breaking the 2-step medium sequence entirely.
  Fix:     CORRECT/WRONG first-action bonuses are now computed relative to the
           episode state AFTER the prefix replay, not blindly from a global list.
           Each task now defines what the correct NEXT action is given the prefix
           state, so the bonus/penalty is always contextually appropriate.

BUG 3 — Chunked training loop with resume_from_checkpoint resets optimizer state
  Problem: The while-loop trains in chunks of ROLLING_WINDOW steps, calling
           trainer.train(resume_from_checkpoint=True) each iteration. Resuming
           from a checkpoint re-loads the optimizer state from disk, which means
           momentum/variance accumulated in the previous chunk is discarded.
           This causes the very low KL (0.0002→0.0006) and near-zero grad_norm
           seen in logs — the optimizer was repeatedly cold-restarting.
  Fix:     Removed the chunked training loop entirely. Each stage now calls
           trainer.train() once for all grpo_steps. The early-exit check is
           done via a custom TRL callback that can stop training cleanly without
           needing to resume.

BUG 4 — frac_reward_zero_std=1.0 in later steps (reward variance collapse)
  Problem: When all 6 GRPO generations hit the same reward path (e.g. all 6
           output the "correct" action from the biased prompts or all 6 output
           a penalised action), reward_std = 0 and GRPO gradient = 0. This was
           confirmed in the logs: frac_reward_zero_std=1.0 at steps 3/3.
  Fix:     BUGs 1+2 fixed restore natural variance. Additionally, num_generations
           is raised to 8 (from 6) to increase the chance of mixed outcomes in
           a batch, and temperature=1.2 during GRPO rollout (was 1.0) to force
           output diversity.

BUG 5 — Completion length stuck at 18-21 tokens (output mode collapse)
  Problem: All 6 generations produced completion_length ≈ 18-21 consistently.
           While a valid JSON action is legitimately ~20 tokens, the zero variance
           in length confirms all 6 rollouts were deterministically identical.
           This is a consequence of BUG 4 (collapsed reward → no gradient → model
           does not explore different outputs).
  Fix:     Addressed by BUGs 1-4 fixes. Additionally top_k=50 added to generation
           config to ensure token-level diversity even within short outputs.

HEALTHY TARGET METRICS (watch these in logs):
  frac_reward_zero_std  < 0.2     (was 1.0 at collapse point)
  reward_std            > 0.5     (was 0.0 at collapse point)
  kl                    0.01–0.1  (was 0.0002–0.0006)
  grad_norm             > 1.0     (was 0.000–0.697)
  clip_ratio/high_mean  0.05–0.2  (was 0.0)

Google Colab setup:
  # Cell 1
  !pip install "unsloth[colab] @ git+https://github.com/unslothai/unsloth.git"
  !pip install trl datasets matplotlib pydantic networkx python-dotenv
  # Cell 2
  !git clone https://github.com/<your-repo>/cluster-triage-env
  %cd cluster-triage-env
  # Cell 3
  !python train_unsloth_colab.py
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
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

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

# Smoke-test curriculum (fast, ~15 min on T4)
# Each stage: (task_id, grpo_steps, num_dataset_prompts)
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
    lora_dropout               = 0,
    bias                       = "none",
    use_gradient_checkpointing = "unsloth",
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
    """Generate one action string from the model."""
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
            temperature        = temperature,
            do_sample          = True,
            top_p              = 0.95,
            top_k              = 50,
            repetition_penalty = 1.1,
            pad_token_id       = tokenizer.eos_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


# ── 6. Evaluation ─────────────────────────────────────────────────────────────
def run_eval_episode(task_id: str) -> dict:
    """Full evaluation episode at temperature=0.3 for stable metrics."""
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
            raw_text   = generate_action_text(SYSTEM_PROMPT, usr_prompt, temperature=0.3)
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
#
# FIX for BUG 1: Each dataset sample now stores prefix_actions alongside the prompt.
# The reward_fn will replay these prefix_actions on a fresh env before evaluating
# the model's action, ensuring the reward context matches the prompt context exactly.
#
# For example, for medium with prefix [clear_temp_storage worker_03]:
#   - Prompt shows: disk=20%, node=offline  (post-clear state)
#   - reward_fn: reset fresh, replay [clear_temp_storage worker_03], THEN eval model action
#   - model outputs restart_node worker_03 → reward = +0.5  ✓  (was -0.3 before fix)
#
# FIX for BUG 2: correct/wrong first-action tables are now CONTEXT-AWARE.
# Each prefix sequence maps to the correct NEXT action from that state.

EXPERT_SEQUENCES = {
    # Format: (prefix_actions, correct_next_action_key, description)
    # correct_next_action_key = (action_type, target_id) or None for "no single right answer"
    "easy": [
        # State: fresh (job_rogue_99 hanging)
        ([], ("kill_job", "job_rogue_99"), "fresh"),
        ([], ("kill_job", "job_rogue_99"), "fresh_copy_2"),
        ([], ("kill_job", "job_rogue_99"), "fresh_copy_3"),
    ],
    "medium": [
        # State: fresh (disk full)
        ([], ("clear_temp_storage", "worker_03"), "fresh"),
        # State: after clearing (disk=20%, node offline) → restart is correct
        ([{"action_type": "clear_temp_storage", "target_id": "worker_03"}],
         ("restart_node", "worker_03"), "post_clear"),
        # State: after noop (still fresh, disk full)
        ([{"action_type": "noop", "target_id": "none"}],
         ("clear_temp_storage", "worker_03"), "after_noop"),
    ],
    "hard": [
        ([], ("kill_job", "job_rogue_99"), "fresh"),
        ([{"action_type": "kill_job", "target_id": "job_rogue_99"}],
         ("clear_temp_storage", "worker_01"), "after_kill"),
        ([{"action_type": "kill_job",           "target_id": "job_rogue_99"},
          {"action_type": "clear_temp_storage", "target_id": "worker_01"}],
         ("clear_temp_storage", "worker_02"), "after_kill_clear1"),
        ([{"action_type": "kill_job",           "target_id": "job_rogue_99"},
          {"action_type": "clear_temp_storage", "target_id": "worker_01"},
          {"action_type": "clear_temp_storage", "target_id": "worker_02"}],
         ("restart_node", "worker_01"), "after_kill_clear_both"),
        # Wrong-first example to teach penalty avoidance
        ([], None, "fresh_wrong_example"),
    ],
    "very_hard": [
        ([], ("kill_job", "job_log_spam"), "fresh"),
        ([{"action_type": "kill_job", "target_id": "job_log_spam"}],
         ("kill_job", "job_crypto_miner"), "after_kill_spam"),
        ([{"action_type": "kill_job", "target_id": "job_crypto_miner"}],
         ("kill_job", "job_log_spam"), "after_kill_crypto"),
        ([{"action_type": "kill_job", "target_id": "job_log_spam"},
          {"action_type": "kill_job", "target_id": "job_crypto_miner"}],
         ("clear_temp_storage", "worker_01"), "after_both_killed"),
        ([{"action_type": "kill_job",           "target_id": "job_log_spam"},
          {"action_type": "kill_job",           "target_id": "job_crypto_miner"},
          {"action_type": "clear_temp_storage", "target_id": "worker_01"}],
         ("clear_temp_storage", "worker_02"), "after_clear1"),
        ([{"action_type": "kill_job",           "target_id": "job_log_spam"},
          {"action_type": "kill_job",           "target_id": "job_crypto_miner"},
          {"action_type": "clear_temp_storage", "target_id": "worker_01"},
          {"action_type": "clear_temp_storage", "target_id": "worker_02"}],
         ("restart_node", "worker_01"), "after_both_cleared"),
    ],
    "nightmare": [
        ([], ("kill_job", "job_hydra_1"), "fresh"),
        ([{"action_type": "kill_job", "target_id": "job_hydra_1"}],
         ("kill_job", "job_hydra_2"), "after_h1"),
        ([{"action_type": "kill_job", "target_id": "job_hydra_1"},
          {"action_type": "kill_job", "target_id": "job_hydra_2"}],
         ("kill_job", "job_hydra_3"), "after_h1h2"),
        ([{"action_type": "kill_job", "target_id": "job_hydra_1"},
          {"action_type": "kill_job", "target_id": "job_hydra_2"},
          {"action_type": "kill_job", "target_id": "job_hydra_3"}],
         ("clear_temp_storage", "worker_01"), "all_hydras_dead"),
        ([{"action_type": "kill_job",           "target_id": "job_hydra_1"},
          {"action_type": "kill_job",           "target_id": "job_hydra_2"},
          {"action_type": "kill_job",           "target_id": "job_hydra_3"},
          {"action_type": "clear_temp_storage", "target_id": "worker_01"}],
         ("clear_temp_storage", "worker_02"), "after_clear1"),
        ([{"action_type": "kill_job",           "target_id": "job_hydra_1"},
          {"action_type": "kill_job",           "target_id": "job_hydra_2"},
          {"action_type": "kill_job",           "target_id": "job_hydra_3"},
          {"action_type": "clear_temp_storage", "target_id": "worker_01"},
          {"action_type": "clear_temp_storage", "target_id": "worker_02"}],
         ("clear_temp_storage", "worker_03"), "after_clear2"),
        # Wrong-first from fresh (teaches penalty)
        ([], None, "fresh_wrong_example"),
    ],
}

# Context-blind wrong first actions (only used for fresh-state prompts where prefix=[])
# These are actions that are ALWAYS wrong at episode start regardless of task state.
ALWAYS_WRONG_FIRST_ACTIONS = {
    "easy":      {("restart_node",       "worker_01"),
                  ("clear_temp_storage", "worker_01")},
    "medium":    {("restart_node",       "worker_03")},  # disk still full at fresh
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


def _add_obs_noise(obs_json: str) -> str:
    """Add ±2% noise to cpu/ram to ensure batch diversity."""
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


def _replay_prefix(task_id: str, prefix_actions: list):
    """
    Create a fresh env, replay prefix_actions, and return (env, obs, history).
    This is the core helper that synchronises prompt state with reward_fn state.
    """
    env = ClusterTriageEnv()
    obs = env.reset(task=task_id)
    history = []
    for act_dict in prefix_actions:
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
    return env, obs, history


def build_dataset(task_id: str, num_prompts: int) -> Dataset:
    """
    Build GRPO training dataset. Each sample is a chat prompt at a specific
    mid-episode decision point. Crucially, each sample also stores its
    prefix_actions as a JSON string so the reward_fn can replay them.
    """
    sequences = EXPERT_SEQUENCES.get(task_id, [([], None, "fresh")])
    samples   = []
    per_seq   = max(1, num_prompts // len(sequences))

    for prefix_actions, correct_next, _desc in sequences:
        for _ in range(per_seq):
            _env, obs, history = _replay_prefix(task_id, prefix_actions)

            # Add cosmetic noise for batch diversity
            obs_json   = _add_obs_noise(obs.model_dump_json(indent=2))
            usr_prompt = build_user_prompt(obs_json, history)

            samples.append({
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": usr_prompt},
                ],
                "task_id":        task_id,
                # FIX BUG 1: store prefix so reward_fn can replay it
                "prefix_actions": json.dumps(prefix_actions),
                # FIX BUG 2: store the contextually correct next action
                "correct_next":   json.dumps(correct_next),   # e.g. ["restart_node","worker_03"] or null
                # Whether this is a fresh-start prompt (for applying always-wrong penalties)
                "is_fresh":       json.dumps(len(prefix_actions) == 0),
            })

    while len(samples) < num_prompts:
        samples.append(random.choice(samples))
    samples = samples[:num_prompts]
    random.shuffle(samples)
    return Dataset.from_list(samples)


# ── 8. Reward Function ────────────────────────────────────────────────────────
#
# FIX BUG 1: reward_fn now replays prefix_actions before evaluating model action.
# FIX BUG 2: first-action bonus/penalty is context-aware (uses correct_next per sample).
#
# Reward budget (correct action):
#   env_reward (context-correct step)   ~+0.5
#   first_action_bonus                  +0.5
#   partial_script bonus                ~+0.3
#   completion/health bonus             +0.0–2.0
#   Total correct:                      ~+1.3 to +3.3
#
# Reward budget (wrong action from any state):
#   env_reward (penalty step)           ~-0.1 to -0.3
#   wrong_action_penalty                -1.0
#   Total wrong:                        ~-1.1 to -1.3
#
# Expected reward_std across 6 gens:    > 0.8

# Partial scripted completion steps run AFTER the model's action
# (only the actions AFTER the prefix state, not including the prefix itself)
PARTIAL_SCRIPTED_COMPLETIONS = {
    "easy": [],   # 1-step task — no scripted completion needed
    "medium": [
        # If model correctly cleared, help it see the restart reward
        {"action_type": "restart_node", "target_id": "worker_03"},
    ],
    "hard": [
        {"action_type": "kill_job",           "target_id": "job_rogue_99"},
        {"action_type": "clear_temp_storage", "target_id": "worker_01"},
    ],
    "very_hard": [
        {"action_type": "kill_job",           "target_id": "job_log_spam"},
        {"action_type": "kill_job",           "target_id": "job_crypto_miner"},
        {"action_type": "clear_temp_storage", "target_id": "worker_01"},
    ],
    "nightmare": [
        {"action_type": "kill_job",           "target_id": "job_hydra_1"},
        {"action_type": "kill_job",           "target_id": "job_hydra_2"},
        {"action_type": "kill_job",           "target_id": "job_hydra_3"},
        {"action_type": "clear_temp_storage", "target_id": "worker_01"},
        {"action_type": "clear_temp_storage", "target_id": "worker_02"},
    ],
}


def make_reward_fn(task_id: str):
    """
    GRPO reward function that evaluates each generation in the correct context.
    The key invariant: reward_fn env state == prompt env state.
    """
    always_wrong  = ALWAYS_WRONG_FIRST_ACTIONS.get(task_id, set())
    script        = PARTIAL_SCRIPTED_COMPLETIONS.get(task_id, [])
    max_steps     = TASK_MAX_STEPS[task_id]

    def reward_fn(prompts, completions, prefix_actions=None,
                  correct_next=None, is_fresh=None, **kwargs):
        rewards = []

        n = len(completions)
        # Safely decode per-sample metadata passed through the dataset
        _prefix_list = []
        _correct_list = []
        _is_fresh_list = []
        for i in range(n):
            try:
                _prefix_list.append(
                    json.loads(prefix_actions[i]) if prefix_actions is not None else [])
            except Exception:
                _prefix_list.append([])
            try:
                cn = json.loads(correct_next[i]) if correct_next is not None else None
                _correct_list.append(tuple(cn) if cn else None)
            except Exception:
                _correct_list.append(None)
            try:
                _is_fresh_list.append(
                    json.loads(is_fresh[i]) if is_fresh is not None else True)
            except Exception:
                _is_fresh_list.append(True)

        for i, completion in enumerate(completions):
            # ── Extract generated text ────────────────────────────────────
            if isinstance(completion, list) and completion:
                c = completion[0]
                action_text = c.get("content", "") if isinstance(c, dict) else str(c)
            elif isinstance(completion, str):
                action_text = completion
            else:
                action_text = str(completion)

            # ── Parse model output ────────────────────────────────────────
            first_action  = parse_action(action_text)
            is_parse_fail = (
                first_action.action_type == "noop"
                and "noop"  not in action_text.lower()
                and "{"     not in action_text
            )

            # ── FIX BUG 1: Replay prefix to match prompt state ────────────
            sample_prefix   = _prefix_list[i]
            sample_correct  = _correct_list[i]   # (action_type, target_id) or None
            sample_is_fresh = _is_fresh_list[i]

            env = ClusterTriageEnv()
            env.reset(task=task_id)
            health_before = env.state_data.health_score

            # Replay the same prefix actions that built the prompt
            for act_dict in sample_prefix:
                try:
                    env.step(ClusterAction(**act_dict))
                except Exception:
                    break

            # Now env is in the same state as when the prompt was built
            health_before_action = env.state_data.health_score
            total_reward  = 0.0
            episode_done  = False

            # ── Step 1: execute model's action in the correct context ─────
            try:
                result        = env.step(first_action)
                total_reward += result.reward
                episode_done  = result.done
            except Exception:
                total_reward = -0.5
                episode_done = True

            # ── Steps 2+: partial scripted completion ─────────────────────
            if not episode_done and script:
                steps_used = 1 + len(sample_prefix)
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

            # ── Final health ──────────────────────────────────────────────
            final_health = 0.0
            if env.state_data is not None:
                final_health = env.state_data.health_score

            # ── Completion bonus ──────────────────────────────────────────
            if final_health >= 1.0:
                total_reward += 2.0
            elif final_health >= 0.5:
                total_reward += final_health * 1.0

            # ── Health-delta bonus ────────────────────────────────────────
            delta = final_health - health_before_action
            if delta > 0.0:
                total_reward += min(0.3, round(int(delta * 20) * 0.05, 2))

            # ── FIX BUG 2: Context-aware first-action bonus/penalty ───────
            action_key = (first_action.action_type, first_action.target_id)

            if sample_correct is not None and action_key == sample_correct:
                # Model output the contextually correct action
                total_reward += 0.5
            elif sample_correct is not None and action_key != sample_correct:
                # Model output the wrong action for this specific context
                # Only penalise if it is not just a variant of a correct action
                if first_action.action_type != "noop":
                    total_reward -= 0.5
            elif sample_is_fresh and action_key in always_wrong:
                # Fresh-state prompt and model picked a clearly wrong action
                total_reward -= 1.0

            # ── Parse / noop penalties ────────────────────────────────────
            if is_parse_fail:
                total_reward -= 2.0
            elif first_action.action_type == "noop":
                total_reward -= 1.5

            rewards.append(float(total_reward))

        return rewards

    return reward_fn


# ── 9. Early-Exit Callback (FIX BUG 3) ───────────────────────────────────────
class EarlyExitCallback(TrainerCallback):
    """
    FIX BUG 3: Replaces the chunked training loop. This callback checks for
    convergence after every logging step and raises an exception to stop training
    cleanly. This avoids the resume_from_checkpoint optimizer-state reset problem.
    """
    def __init__(self, tracker, task_id: str):
        self.tracker  = tracker
        self.task_id  = task_id
        self.exited   = False

    def on_log(self, args: TrainingArguments, state: TrainerState,
               control: TrainerControl, logs=None, **kwargs):
        if logs is None:
            return
        r = logs.get("reward", logs.get("rewards/reward_fn/mean", None))
        if r is not None:
            self.tracker.record(float(r), task_id=self.task_id)
            health_monitor.record(logs, state.global_step)

        steps_done = state.global_step
        if steps_done >= MIN_STEPS_BEFORE_EXIT:
            threshold = EARLY_EXIT_THRESHOLDS.get(self.task_id, 0.5)
            mean = self.tracker.rolling_mean(self.task_id)
            if mean >= threshold:
                print(
                    f"\n[EARLY EXIT] '{self.task_id}' converged at step {steps_done}. "
                    f"Rolling mean={mean:.3f} >= threshold={threshold}."
                )
                self.exited = True
                control.should_training_stop = True


# ── 10. GRPO Health Monitor ───────────────────────────────────────────────────
class GRPOHealthMonitor:
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


# ── 11. Metrics Tracker ───────────────────────────────────────────────────────
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


# ── 12. Baseline Evaluation ───────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  PHASE 1: BASELINE EVALUATION (untrained Llama-3.2-3B)")
print("  Expected: ~0% success on hard+. Easy may be non-zero.")
print("=" * 65)

baseline_metrics = evaluate_all_tasks("BASELINE")


# ── 13. Curriculum GRPO Training ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("  PHASE 2: CURRICULUM GRPO TRAINING")
print("  FIX 1: reward_fn replays prefix before evaluating model action")
print("  FIX 2: first-action bonus/penalty is context-aware per sample")
print("  FIX 3: single trainer.train() call per stage (no resume loop)")
print("  FIX 4: num_generations=8, temperature=1.2 for output diversity")
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

    # FIX BUG 3: use callback instead of chunked loop with resume
    early_exit_cb = EarlyExitCallback(tracker, task_id)

    training_args = GRPOConfig(
        output_dir                  = f"checkpoints/stage_{stage_idx+1}_{task_id}",
        learning_rate               = 3e-5,
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 6,
        num_generations             = 8,      # FIX BUG 4: was 6
        max_completion_length       = 200,
        max_prompt_length           = 1024,
        max_steps                   = grpo_steps,
        logging_steps               = 1,
        save_steps                  = grpo_steps,
        optim                       = "adamw_8bit",
        warmup_steps                = 1,      # replaces deprecated warmup_ratio
        report_to                   = "none",
        remove_unused_columns       = False,
        temperature                 = 1.2,    # FIX BUG 4: was 1.0
        beta                        = 0.01,
    )

    trainer = GRPOTrainer(
        model            = model,
        processing_class = tokenizer,
        reward_funcs     = [reward_fn],
        args             = training_args,
        train_dataset    = dataset,
        callbacks        = [early_exit_cb],   # FIX BUG 3: early exit via callback
    )

    t0 = time.time()

    # FIX BUG 3: single call — no chunked loop, no resume_from_checkpoint
    trainer.train()

    try:
        model.enable_adapters()
    except Exception:
        pass
    model.train()

    elapsed    = time.time() - t0
    steps_done = trainer.state.global_step if hasattr(trainer, "state") else grpo_steps
    flag       = " [EARLY EXIT]" if early_exit_cb.exited else ""
    print(f"[INFO] Stage {stage_idx+1} done in {elapsed:.0f}s "
          f"({steps_done}/{grpo_steps} steps){flag}.")

health_monitor.summary()


# ── 14. Post-Training Evaluation ──────────────────────────────────────────────
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


# ── 15. Results Table ─────────────────────────────────────────────────────────
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


# ── 16. Plotting ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "figure.dpi":        130,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# Plot 1: Training Reward Curve
fig, ax = plt.subplots(figsize=(13, 5))
if tracker.step_rewards:
    steps   = [s for s, _ in tracker.step_rewards]
    rewards = [r for _, r in tracker.step_rewards]
    w       = max(1, min(5, len(rewards) // 3))
    smooth  = np.convolve(rewards, np.ones(w) / w, mode="same")
    ax.plot(steps, rewards, color="#94a3b8", alpha=0.3, linewidth=0.8,
            label="Raw reward")
    ax.plot(steps, smooth, color="#6366f1", linewidth=2.2,
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
    "Llama-3.2-3B · ClusterTriageEnv · State-Aligned Reward"
)
ax.legend(loc="lower right", fontsize=9)
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig("plots/training_reward_curve.png",       bbox_inches="tight", dpi=150)
fig.savefig("plots/training_reward_curve_hires.png", bbox_inches="tight", dpi=300)
plt.close(fig)
print("\n[PLOT] Saved: plots/training_reward_curve.png")

# Plot 2: Success Rate Comparison
fig, ax = plt.subplots(figsize=(13, 5))
x   = np.arange(len(task_ids))
bw  = 0.28
base_sr  = [baseline_metrics[t]["success_rate"] for t in task_ids]
train_sr = [trained_metrics[t]["success_rate"]  for t in task_ids]
part_sr  = [trained_metrics[t]["partial_rate"]  for t in task_ids]
bars_b = ax.bar(x - bw,  base_sr,  bw, label="Baseline SR",            color="#94a3b8", alpha=0.9)
bars_t = ax.bar(x,        train_sr, bw, label="Fine-tuned SR",          color="#6366f1", alpha=0.9)
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

# Plot 3: Average Reward Comparison
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

# Plot 4: Health Recovery Comparison
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

# Plot 5: Per-Stage Reward Distribution (violin)
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

# Plot 6: GRPO Health Dashboard (4-panel)
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


# ── 17. Save Model ────────────────────────────────────────────────────────────
model.save_pretrained(LORA_OUTPUT_DIR)
tokenizer.save_pretrained(LORA_OUTPUT_DIR)
print(f"\n[INFO] LoRA adapter saved → '{LORA_OUTPUT_DIR}/'")


# ── 18. Final Summary ─────────────────────────────────────────────────────────
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
║  frac_reward_zero_std  < 0.20          1.00  ← collapse         ║
║  reward_std            > 0.50          0.000 ← collapse         ║
║  kl                  0.01–0.10         0.0002 ← near zero       ║
║  grad_norm             > 1.00          0.000 ← near zero        ║
║  clip_ratio/high_mean  0.05–0.20       0.000 ← no clipping      ║
╚══════════════════════════════════════════════════════════════════╝
""")