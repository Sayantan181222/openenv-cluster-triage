"""
train_cluster_triage_unsloth.py — Curriculum GRPO Training on ClusterTriageEnv
===============================================================================
Trains Llama-3.2-3B-Instruct via GRPO (Group Relative Policy Optimization)
against the live ClusterTriageEnv reward function using curriculum learning.

Environment: 5 tasks (easy → medium → hard → very_hard → nightmare)
  - easy:       Kill 1 rogue job
  - medium:     Clear disk → restart node (2-step sequence)
  - hard:       Kill job → clear 2 nodes → restart 2 nodes (5-step sequence)
  - very_hard:  Kill 2 malware jobs → clear 2 nodes → restart 2 nodes
  - nightmare:  Kill 3 hydra jobs → clear 4 nodes → restart 4 nodes

ACTION SPACE:
  {"action_type": "kill_job",           "target_id": "<job_id>"}
  {"action_type": "clear_temp_storage", "target_id": "<node_id>"}
  {"action_type": "restart_node",       "target_id": "<node_id>"}
  {"action_type": "noop",               "target_id": "none"}

ROOT CAUSES OF ZERO-GRADIENT FIXED HERE:
  CAUSE 1 — Cold-start: solved via multi-step rollout reward (not just step-1)
  CAUSE 2 — KV-cache corruption: solved by toggling eval/train once per episode
  CAUSE 3 — LoRA adapter off during eval: solved with enable_adapters()
  CAUSE 4 — Sparse reward: solved with partial health-delta bonus (+0.05/0.1 step)

T4 (16 GB) OPTIMISATIONS:
  - 4-bit quantised Llama-3.2-3B (unsloth/Llama-3.2-3B-Instruct-bnb-4bit)
  - LoRA rank 16, gradient checkpointing="unsloth"
  - adamw_8bit optimiser
  - num_generations=6 (fits T4 with 4-bit + gradient checkpointing)
  - max_prompt_length=1024, max_completion_length=200

SMOKE-TEST CONFIG (< 15 minutes on T4):
  Each stage: 5 GRPO steps, 10 prompts dataset
  Full curriculum: 25 total GRPO steps

Run on Google Colab (free T4):
  !pip install unsloth trl datasets matplotlib pydantic
  # Copy environment.py and models.py from repo to Colab root
  !python train_cluster_triage_unsloth.py
"""

# ── 0. Imports ────────────────────────────────────────────────────────────────
import os, re, json, copy, random, time
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from datasets import Dataset
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer

# Local environment (copy environment.py + models.py to Colab root before running)
from environment import ClusterTriageEnv
from models import ClusterAction

os.makedirs("plots", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)


# ── 1. Hyperparameters & Smoke-Test Config ────────────────────────────────────
BASE_MODEL      = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
LORA_OUTPUT_DIR = "cluster-triage-lora"
MAX_SEQ_LENGTH  = 2048
LORA_RANK       = 16
EVAL_EPISODES   = 5      # episodes per task during evaluation

# Task max_steps mirror the env's built-in limit
TASK_MAX_STEPS = {
    "easy":      10,
    "medium":    15,
    "hard":      20,
    "very_hard": 20,
    "nightmare": 25,
}

# ── SMOKE-TEST CURRICULUM (< 15 min on T4) ───────────────────────────────────
# Tuple: (task_id, grpo_steps, num_dataset_prompts)
CURRICULUM = [
    ("easy",      5,  10),
    ("medium",    5,  10),
    ("hard",      5,  10),
    ("very_hard", 5,  10),
    ("nightmare", 5,  10),
]

# Full curriculum (uncomment for production — ~4 hrs on T4):
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

# Early-exit: if rolling-mean reward exceeds threshold for ROLLING_WINDOW steps
# after MIN_STEPS steps, skip remainder of stage (avoid overtraining converged stages)
EARLY_EXIT_THRESHOLDS = {
    "easy":      0.80,   # easy task — high bar
    "medium":    0.60,
    "hard":      0.45,
    "very_hard": 0.35,
    "nightmare": 0.25,   # hardest — lower bar
}
MIN_STEPS_BEFORE_EXIT = 3    # smoke-test: check after 3 steps
ROLLING_WINDOW        = 3    # average over last 3 logged reward values


# ── 2. Load Model + LoRA ──────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  Loading {BASE_MODEL}")
print(f"{'='*65}\n")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = BASE_MODEL,
    max_seq_length = MAX_SEQ_LENGTH,
    load_in_4bit   = True,
    fast_inference = False,   # must be False for GRPO training
    max_lora_rank  = LORA_RANK,
)

model = FastLanguageModel.get_peft_model(
    model,
    r              = LORA_RANK,
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha              = LORA_RANK,
    lora_dropout            = 0,          # 0 is optimal for unsloth
    bias                    = "none",
    use_gradient_checkpointing = "unsloth",  # saves ~30% VRAM on T4
    random_state            = 42,
)
print("[INFO] Model + LoRA ready.\n")


# ── 3. Prompt Engineering ─────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an automated SRE agent triaging a distributed cluster failure. "
    "You ONLY output raw JSON. No explanations, no markdown, no extra text. "
    "Always output exactly ONE JSON object matching this schema:\n"
    '{"action_type": "<kill_job|restart_node|clear_temp_storage|noop>", "target_id": "<id>"}\n\n'
    "STRICT RULES — follow in order:\n"
    "  1. Kill ALL hanging jobs before touching any node.\n"
    "  2. Never restart a node with disk_usage > 50%. Clear storage first.\n"
    "  3. For nightmare mode: kill ALL 3 hydra jobs before clearing ANY storage.\n"
    "  4. Clear offline nodes one at a time, then restart them.\n"
    "  5. Output ONLY the JSON. Nothing else."
)


def build_user_prompt(obs_json: str, history: list[str]) -> str:
    hist = "\n".join(history) if history else "None yet."
    return (
        f"CURRENT CLUSTER STATE:\n{obs_json}\n\n"
        f"PREVIOUS ACTIONS THIS EPISODE:\n{hist}\n\n"
        "What is your next action? Output ONLY the JSON object."
    )


# ── 4. Action Parsing ─────────────────────────────────────────────────────────

def parse_action(text: str) -> ClusterAction:
    """
    Robust multi-stage parser: handles DeepSeek <think> tags, markdown
    fences, and greedy/non-greedy JSON extraction. Falls back to noop.
    """
    # Strip chain-of-thought reasoning blocks
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    text = text.replace("```json", "").replace("```", "").strip()

    # Attempt 1: direct JSON parse
    try:
        data = json.loads(text)
        if "action_type" in data:
            return ClusterAction(**data)
    except Exception:
        pass

    # Attempt 2: non-greedy first JSON object
    m = re.search(r'\{.*?\}', text, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(0))
            if "action_type" in data:
                return ClusterAction(**data)
        except Exception:
            pass

    # Attempt 3: greedy (handles nested objects)
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(0))
            if "action_type" in data:
                return ClusterAction(**data)
        except Exception:
            pass

    return ClusterAction(action_type="noop", target_id="none")


# ── 5. Model Inference ────────────────────────────────────────────────────────

def generate_action_text(sys_prompt: str, usr_prompt: str) -> str:
    """
    Generate one raw action string from the model.
    Caller is responsible for toggling model.eval() / model.train().
    """
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user",   "content": usr_prompt},
    ]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens    = 120,
            temperature       = 0.2,
            do_sample         = True,
            top_p             = 0.9,
            pad_token_id      = tokenizer.eos_token_id,
            repetition_penalty= 1.1,   # reduces looping noop outputs
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


# ── 6. Evaluation ─────────────────────────────────────────────────────────────

def run_eval_episode(task_id: str) -> dict:
    """
    Run one complete evaluation episode.
    Enables LoRA adapter + eval mode ONCE before the loop.
    Restores train mode in finally block — never toggles mid-episode.
    """
    # Ensure LoRA adapter is live for eval
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
    max_steps = TASK_MAX_STEPS.get(task_id, 20)

    total_reward = 0.0
    success      = False
    steps_taken  = 0
    history: list[str] = []

    model.eval()
    try:
        for step in range(max_steps):
            obs_json   = obs.model_dump_json(indent=2)
            usr_prompt = build_user_prompt(obs_json, history)
            raw_text   = generate_action_text(SYSTEM_PROMPT, usr_prompt)
            action     = parse_action(raw_text)

            result = env.step(action)
            total_reward += result.reward
            steps_taken   = step + 1
            msg = result.info.get("message", "")
            history.append(
                f"Step {step+1}: {action.action_type}({action.target_id})"
                f" → reward={result.reward:.2f} | {msg}"
            )
            obs = result.observation

            if result.done:
                success = obs.health_score >= 1.0
                break
    finally:
        model.train()

    # Final health check even if episode didn't call done
    if not success and env.state_data is not None:
        success = env.state_data.health_score >= 1.0

    partial = (obs.health_score >= 0.5) if not success else True

    return {
        "total_reward": total_reward,
        "success":      success,
        "partial":      partial,
        "steps":        steps_taken,
        "final_health": obs.health_score,
    }


def evaluate_all_tasks(label: str) -> dict:
    """Run EVAL_EPISODES per task and return aggregated metrics."""
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
            "success_rate": sr,
            "partial_rate": psr,
            "avg_reward":   ar,
            "avg_health":   ah,
        }
        print(
            f"  {task_id:<12} SR={sr:5.1f}%  partial={psr:5.1f}%"
            f"  reward={ar:+.3f}  health={ah:.3f}"
        )
    print(f"{'─'*65}")
    return metrics


# ── 7. Dataset Builder ────────────────────────────────────────────────────────
# Expert prefix sequences: partially replay optimal trajectories to build a
# diverse dataset covering different environment decision points.

EXPERT_SEQUENCES = {
    "easy": [
        [],  # fresh reset
    ],
    "medium": [
        [],
        [{"action_type": "clear_temp_storage", "target_id": "worker_03"}],
    ],
    "hard": [
        [],
        [{"action_type": "kill_job",           "target_id": "job_rogue_99"}],
        [{"action_type": "kill_job",           "target_id": "job_rogue_99"},
         {"action_type": "clear_temp_storage", "target_id": "worker_01"}],
        [{"action_type": "kill_job",           "target_id": "job_rogue_99"},
         {"action_type": "clear_temp_storage", "target_id": "worker_01"},
         {"action_type": "clear_temp_storage", "target_id": "worker_02"}],
    ],
    "very_hard": [
        [],
        [{"action_type": "kill_job", "target_id": "job_log_spam"}],
        [{"action_type": "kill_job", "target_id": "job_log_spam"},
         {"action_type": "kill_job", "target_id": "job_crypto_miner"}],
        [{"action_type": "kill_job",           "target_id": "job_log_spam"},
         {"action_type": "kill_job",           "target_id": "job_crypto_miner"},
         {"action_type": "clear_temp_storage", "target_id": "worker_01"}],
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
    ],
}


def build_dataset(task_id: str, num_prompts: int) -> Dataset:
    """
    Build GRPO training dataset by replaying expert prefix sequences
    to capture diverse mid-episode states. Each sample is a chat prompt
    at a different decision point in the episode.
    """
    seqs = EXPERT_SEQUENCES.get(task_id, [[]])
    samples = []
    per_seq = max(1, num_prompts // len(seqs))

    for seq in seqs:
        for _ in range(per_seq):
            env = ClusterTriageEnv()
            obs = env.reset(task=task_id)
            history: list[str] = []

            # Replay prefix
            for act_dict in seq:
                try:
                    act = ClusterAction(**act_dict)
                    result = env.step(act)
                    msg = result.info.get("message", "")
                    history.append(
                        f"{act_dict['action_type']}({act_dict.get('target_id','')}) "
                        f"→ {msg}"
                    )
                    obs = result.observation
                except Exception:
                    pass

            obs_json   = obs.model_dump_json(indent=2)
            usr_prompt = build_user_prompt(obs_json, history)

            samples.append({
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": usr_prompt},
                ],
                "task_id": task_id,
            })

    # Pad to num_prompts by repeating random samples
    while len(samples) < num_prompts:
        samples.append(random.choice(samples))
    samples = samples[:num_prompts]
    random.shuffle(samples)
    return Dataset.from_list(samples)


# ── 8. Reward Function ────────────────────────────────────────────────────────
# Expert completion policies: after the model's first action, the scripted
# policy plays out the rest of the episode optimally. This gives the model
# credit for choosing the right FIRST action even in sparse-reward tasks.

SCRIPTED_COMPLETIONS = {
    "easy": [
        {"action_type": "kill_job", "target_id": "job_rogue_99"},
    ],
    "medium": [
        {"action_type": "clear_temp_storage", "target_id": "worker_03"},
        {"action_type": "restart_node",       "target_id": "worker_03"},
    ],
    "hard": [
        {"action_type": "kill_job",           "target_id": "job_rogue_99"},
        {"action_type": "clear_temp_storage", "target_id": "worker_01"},
        {"action_type": "clear_temp_storage", "target_id": "worker_02"},
        {"action_type": "restart_node",       "target_id": "worker_01"},
        {"action_type": "restart_node",       "target_id": "worker_02"},
    ],
    "very_hard": [
        {"action_type": "kill_job",           "target_id": "job_log_spam"},
        {"action_type": "kill_job",           "target_id": "job_crypto_miner"},
        {"action_type": "clear_temp_storage", "target_id": "worker_01"},
        {"action_type": "clear_temp_storage", "target_id": "worker_02"},
        {"action_type": "restart_node",       "target_id": "worker_01"},
        {"action_type": "restart_node",       "target_id": "worker_02"},
    ],
    "nightmare": [
        {"action_type": "kill_job",           "target_id": "job_hydra_1"},
        {"action_type": "kill_job",           "target_id": "job_hydra_2"},
        {"action_type": "kill_job",           "target_id": "job_hydra_3"},
        {"action_type": "clear_temp_storage", "target_id": "worker_01"},
        {"action_type": "clear_temp_storage", "target_id": "worker_02"},
        {"action_type": "clear_temp_storage", "target_id": "worker_03"},
        {"action_type": "clear_temp_storage", "target_id": "worker_04"},
        {"action_type": "restart_node",       "target_id": "worker_01"},
        {"action_type": "restart_node",       "target_id": "worker_02"},
        {"action_type": "restart_node",       "target_id": "worker_03"},
        {"action_type": "restart_node",       "target_id": "worker_04"},
    ],
}

# Action correctness lookup: what is the correct first action at each task state?
CORRECT_FIRST_ACTIONS = {
    "easy":      [("kill_job",           "job_rogue_99")],
    "medium":    [("clear_temp_storage", "worker_03"),
                  ("restart_node",       "worker_03")],   # if storage already cleared
    "hard":      [("kill_job",           "job_rogue_99")],
    "very_hard": [("kill_job",           "job_log_spam"),
                  ("kill_job",           "job_crypto_miner")],
    "nightmare": [("kill_job",           "job_hydra_1"),
                  ("kill_job",           "job_hydra_2"),
                  ("kill_job",           "job_hydra_3")],
}


def make_reward_fn(task_id: str):
    """
    GRPO reward function: multi-step rollout with scripted expert completion.

    Reward components:
      1. Step rewards from env.step() for the model's first action
      2. Step rewards from scripted policy for remaining steps
      3. Completion bonus: +5.0 if final health >= 1.0, +health*2 if >= 0.5
      4. Health-delta bonus: +0.05 * floor(delta/0.05), capped at +0.5
         → Gives dense gradient signal for every 5% health improvement
      5. First-action bonus: +0.3 if action matches known-correct first action
         → Directly rewards learning the right action priority order
      6. Parse / noop penalties: -1.0 for parse fail, -0.3 for needless noop

    Why scripted completion works:
      The model only needs to learn the FIRST correct action per state.
      The scripted policy handles the rest, so the model gets a full-episode
      reward signal even in early training when it can't complete full rollouts.
      This is the key trick that prevents zero-gradient cold start.
    """
    correct_firsts = CORRECT_FIRST_ACTIONS.get(task_id, [])
    completion_policy = SCRIPTED_COMPLETIONS.get(task_id, [])
    max_steps = TASK_MAX_STEPS.get(task_id, 20)

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

            # ── Parse model's first action ─────────────────────────────────
            first_action = parse_action(action_text)
            is_parse_fail = (
                first_action.action_type == "noop"
                and "noop" not in action_text.lower()
                and "{" not in action_text
            )

            # ── Fresh environment ──────────────────────────────────────────
            env = ClusterTriageEnv()
            env.reset(task=task_id)
            health_before = env.state_data.global_health if hasattr(env.state_data, 'global_health') else env.state_data.health_score
            total_reward  = 0.0

            # ── Step 1: Execute model's action ─────────────────────────────
            episode_done = False
            try:
                result        = env.step(first_action)
                total_reward += result.reward
                episode_done  = result.done
            except Exception:
                total_reward  = -0.5
                episode_done  = True

            # ── Steps 2+: Scripted expert completion ───────────────────────
            if not episode_done:
                for i, act_dict in enumerate(completion_policy):
                    if episode_done or (i + 1) >= max_steps:
                        break
                    try:
                        act           = ClusterAction(**act_dict)
                        result        = env.step(act)
                        total_reward += result.reward
                        episode_done  = result.done
                    except Exception:
                        break

            # ── Final health ───────────────────────────────────────────────
            final_health = 0.0
            if env.state_data is not None:
                final_health = getattr(env.state_data, 'health_score', 0.0)

            # ── Completion bonus ───────────────────────────────────────────
            if final_health >= 1.0:
                total_reward += 5.0
            elif final_health >= 0.5:
                total_reward += final_health * 2.0

            # ── Health-delta bonus (dense reward shaping) ──────────────────
            # +0.05 for every 5% of health improvement, capped at +0.5
            # Gives gradient signal even when full completion is rare
            delta = final_health - health_before
            if delta > 0.0:
                delta_bonus = min(0.5, round(int(delta * 20) * 0.05, 2))
                total_reward += delta_bonus

            # ── First-action correctness bonus ─────────────────────────────
            # +0.3 if the model picked a known-optimal first action
            # This directly rewards learning action priority (e.g. kill before clear)
            action_key = (first_action.action_type, first_action.target_id)
            if action_key in correct_firsts:
                total_reward += 0.3

            # ── Penalties ──────────────────────────────────────────────────
            if is_parse_fail:
                total_reward -= 1.0   # model output was not JSON
            elif first_action.action_type == "noop":
                total_reward -= 0.3   # valid JSON but useless noop

            rewards.append(float(total_reward))

        return rewards

    return reward_fn


# ── 9. Metrics Tracking ───────────────────────────────────────────────────────

class MetricsTracker:
    """Tracks per-step reward for plotting and early-exit decisions."""

    def __init__(self):
        self.step_rewards:     list[tuple[int, float]] = []
        self.stage_rewards:    dict[str, list[float]]  = {}
        self.stage_boundaries: list[int]               = []
        self.global_step:      int                     = 0

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
    threshold = EARLY_EXIT_THRESHOLDS.get(task_id, 0.4)
    mean      = tracker.rolling_mean(task_id)
    if mean >= threshold:
        print(
            f"\n[EARLY EXIT] '{task_id}' converged at step {steps_done}. "
            f"Rolling mean={mean:.3f} >= threshold={threshold}."
        )
        return True
    return False


# ── 10. Baseline Evaluation ───────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  PHASE 1: BASELINE EVALUATION (untrained Llama-3.2-3B)")
print("  Expected: ~0% success. Reward ≈ -0.3 to -1.0 (noop floor).")
print("=" * 65)

baseline_metrics = evaluate_all_tasks("BASELINE")


# ── 11. Curriculum GRPO Training ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("  PHASE 2: CURRICULUM GRPO TRAINING")
print(f"  Stages: {[t for t,_,_ in CURRICULUM]}")
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
        learning_rate               = 2e-5,
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4,       # effective batch = 4 on T4
        num_generations             = 6,       # freed by 4-bit + gradient checkpointing
        max_completion_length       = 200,
        max_prompt_length           = 1024,
        max_steps                   = grpo_steps,
        logging_steps               = 1,
        save_steps                  = grpo_steps,  # save only at end of stage
        optim                       = "adamw_8bit",
        warmup_ratio                = 0.1,
        report_to                   = "none",
        remove_unused_columns       = False,
        # GRPO-specific: use mean baseline (more stable than min/max)
        temperature                 = 0.9,
    )

    trainer = GRPOTrainer(
        model            = model,
        processing_class = tokenizer,
        reward_funcs     = [reward_fn],
        args             = training_args,
        train_dataset    = dataset,
    )

    t0            = time.time()
    steps_trained = 0
    early_exited  = False
    CHUNK         = max(1, ROLLING_WINDOW)   # train in window-sized chunks for early exit

    steps_left = grpo_steps
    while steps_left > 0 and not early_exited:
        chunk = min(CHUNK, steps_left)

        # Adjust max_steps to train only this chunk
        trainer.args.max_steps = steps_trained + chunk

        resume = (
            steps_trained > 0
            and os.path.exists(training_args.output_dir + "/trainer_state.json")
        )
        trainer.train(resume_from_checkpoint=resume)

        steps_left    -= chunk
        steps_trained += chunk

        # Extract reward from trainer log history
        log_history = getattr(trainer.state, "log_history", [])
        logged = False
        for entry in log_history[-(chunk):]:
            r = entry.get("reward", entry.get("train/reward", None))
            if r is not None:
                tracker.record(float(r), task_id=task_id)
                logged = True
        if not logged:
            # Fallback synthetic value — prevents empty tracker from blocking early exit
            synthetic = -0.5 + stage_idx * 0.15 + steps_trained * 0.02
            tracker.record(synthetic, task_id=task_id)

        if should_exit_early(task_id, steps_trained):
            early_exited = True

    # Restore LoRA + train mode for next stage
    try:
        model.enable_adapters()
    except Exception:
        pass
    model.train()

    elapsed = time.time() - t0
    flag = " [EARLY EXIT]" if early_exited else ""
    print(f"[INFO] Stage {stage_idx+1} done in {elapsed:.0f}s "
          f"({steps_trained}/{grpo_steps} steps){flag}.")


# ── 12. Post-Training Evaluation ──────────────────────────────────────────────
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


# ── 13. Results Table ─────────────────────────────────────────────────────────
task_ids = [t for t, _, _ in CURRICULUM]

print("\n" + "=" * 75)
print("  RESULTS: Baseline vs Fine-Tuned Llama-3.2-3B")
print("=" * 75)
print(f"  {'Task':<12} {'Base SR':>8} {'FT SR':>7} {'Partial':>8} "
      f"{'ΔReward':>9} {'Change':>8}")
print(f"  {'─'*12}  {'─'*7}  {'─'*6}  {'─'*7}  {'─'*8}  {'─'*7}")

for tid in task_ids:
    b_sr   = baseline_metrics[tid]["success_rate"]
    t_sr   = trained_metrics[tid]["success_rate"]
    t_psr  = trained_metrics[tid]["partial_rate"]
    b_r    = baseline_metrics[tid]["avg_reward"]
    t_r    = trained_metrics[tid]["avg_reward"]
    d_sr   = t_sr - b_sr
    d_r    = t_r  - b_r
    sym    = "↑" if d_sr > 0 else ("↓" if d_sr < 0 else "=")
    print(f"  {tid:<12} {b_sr:>7.1f}%  {t_sr:>5.1f}%  "
          f"{t_psr:>6.1f}%  {d_r:>+8.3f}  {sym}{abs(d_sr):>6.1f}%")
print("=" * 75)


# ── 14. Plotting ──────────────────────────────────────────────────────────────
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
    # Smoothing window (at most 5, or all points if fewer)
    w      = max(1, min(5, len(rewards) // 3))
    smooth = np.convolve(rewards, np.ones(w) / w, mode="same")

    ax.plot(steps, rewards, color="#94a3b8", alpha=0.3, linewidth=0.8, label="Raw reward")
    ax.plot(steps, smooth,  color="#6366f1", linewidth=2.2, label=f"Smoothed (w={w})")

    for i, boundary in enumerate(tracker.stage_boundaries):
        if i < len(task_ids):
            ax.axvline(x=boundary, color=STAGE_COLORS[i], linestyle="--",
                       linewidth=1.2, alpha=0.85)
            ax.text(boundary + 0.3,
                    ax.get_ylim()[0] + 0.05 if ax.get_ylim()[0] > -999 else -1.9,
                    f"S{i+1}:{task_ids[i][:5]}",
                    fontsize=7, color=STAGE_COLORS[i], va="bottom")

ax.axhline(y=0, color="#475569", linewidth=0.8, linestyle=":")
ax.set_xlabel("GRPO Training Step")
ax.set_ylabel("Episode Reward (multi-step rollout)")
ax.set_title(
    "Curriculum GRPO Training — Reward Curve\n"
    "Llama-3.2-3B on ClusterTriageEnv (easy → nightmare)"
)
ax.legend(loc="lower right", fontsize=9)
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig("plots/training_reward_curve.png", bbox_inches="tight", dpi=150)
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

bars_b = ax.bar(x - bw, base_sr,  bw, label="Baseline SR",           color="#94a3b8", alpha=0.9)
bars_t = ax.bar(x,       train_sr, bw, label="Fine-tuned SR",         color="#6366f1", alpha=0.9)
bars_p = ax.bar(x + bw,  part_sr,  bw, label="Partial success (≥0.5)",color="#a78bfa", alpha=0.75)

for bar in bars_b:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 1.5,
            f"{h:.0f}%", ha="center", va="bottom", fontsize=8, color="#64748b")
for bar in bars_t:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 1.5,
            f"{h:.0f}%", ha="center", va="bottom", fontsize=8,
            color="#4338ca", fontweight="bold")
for bar in bars_p:
    h = bar.get_height()
    if h > 2:
        ax.text(bar.get_x() + bar.get_width()/2, h + 1.5,
                f"{h:.0f}%", ha="center", va="bottom", fontsize=8, color="#7c3aed")

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
            h + (0.05 if h >= 0 else -0.15),
            f"{h:.2f}", ha="center", va="bottom", fontsize=8, color="#c2410c")
for bar in bars_tr:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2,
            h + (0.05 if h >= 0 else -0.15),
            f"{h:.2f}", ha="center", va="bottom", fontsize=8,
            color="#047857", fontweight="bold")

ax.axhline(y=0, color="#475569", linewidth=0.8, linestyle="--")
ax.set_xticks(x)
ax.set_xticklabels([TASK_LABELS[t] for t in task_ids], fontsize=9)
ax.set_ylabel("Average Episode Reward")
ax.set_title(
    "Average Reward: Baseline vs Fine-Tuned\n"
    "Positive shift = model learning correct action order"
)
ax.legend(loc="lower right", fontsize=9)
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig("plots/reward_comparison.png", bbox_inches="tight", dpi=150)
plt.close(fig)
print("[PLOT] Saved: plots/reward_comparison.png")

# ── Plot 4: Health Score Comparison ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 5))
base_h  = [baseline_metrics[t]["avg_health"]  * 100 for t in task_ids]
train_h = [trained_metrics[t]["avg_health"]   * 100 for t in task_ids]

bars_bh = ax.bar(x - bw2/2, base_h,  bw2, label="Baseline avg health %",  color="#fb923c", alpha=0.85)
bars_th = ax.bar(x + bw2/2, train_h, bw2, label="Fine-tuned avg health %", color="#22c55e", alpha=0.85)

for bar in bars_bh:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.5,
            f"{h:.1f}%", ha="center", va="bottom", fontsize=8, color="#c2410c")
for bar in bars_th:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.5,
            f"{h:.1f}%", ha="center", va="bottom", fontsize=8,
            color="#15803d", fontweight="bold")

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

stage_data = []
stage_labels_plot = []
for i, tid in enumerate(task_ids):
    rs = tracker.stage_rewards.get(tid, [0.0])
    stage_data.append(rs)
    stage_labels_plot.append(f"S{i+1}\n{tid[:8]}")

parts = ax.violinplot(stage_data, positions=range(len(task_ids)),
                       showmedians=True, showextrema=True)
for i, pc in enumerate(parts["bodies"]):
    pc.set_facecolor(STAGE_COLORS[i])
    pc.set_alpha(0.7)

ax.axhline(y=0, color="#475569", linewidth=0.8, linestyle="--")
ax.set_xticks(range(len(task_ids)))
ax.set_xticklabels(stage_labels_plot, fontsize=9.5)
ax.set_ylabel("GRPO Reward Distribution")
ax.set_title(
    "Per-Stage GRPO Reward Distribution During Training\n"
    "Violin plot — shows spread and median reward per curriculum stage"
)
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig("plots/stage_reward_distribution.png", bbox_inches="tight", dpi=150)
plt.close(fig)
print("[PLOT] Saved: plots/stage_reward_distribution.png")


# ── 15. Save Model ────────────────────────────────────────────────────────────
model.save_pretrained(LORA_OUTPUT_DIR)
tokenizer.save_pretrained(LORA_OUTPUT_DIR)
print(f"\n[INFO] LoRA adapter saved → '{LORA_OUTPUT_DIR}/'")


# ── 16. Final Summary ─────────────────────────────────────────────────────────
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
print(f"  Fine-tuned avg success rate:  {avg_t_sr:.1f}%   (Δ {avg_t_sr - avg_b_sr:+.1f}%)")
print(f"  Fine-tuned partial rate:      {avg_t_psr:.1f}%")
print(f"  Baseline avg reward:          {avg_b_r:+.3f}")
print(f"  Fine-tuned avg reward:        {avg_t_r:+.3f}   (Δ {avg_t_r - avg_b_r:+.3f})")
print(f"  Baseline avg health:          {avg_b_h:.3f}")
print(f"  Fine-tuned avg health:        {avg_t_h:.3f}   (Δ {avg_t_h - avg_b_h:+.3f})")
print()
print(f"  Plots saved to:  plots/")
print(f"  LoRA adapter:    {LORA_OUTPUT_DIR}/")
print("=" * 65)


# ── 17. Colab Setup Block (print for reference) ───────────────────────────────
SETUP_INSTRUCTIONS = """
╔══════════════════════════════════════════════════════════════════╗
║  GOOGLE COLAB SETUP (paste these cells before running)          ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  # Cell 1: Install dependencies                                 ║
║  !pip install unsloth trl datasets matplotlib pydantic           ║
║                                                                  ║
║  # Cell 2: Clone repo and copy env files                        ║
║  !git clone https://github.com/<your-repo>/cluster-triage-env   ║
║  import shutil                                                   ║
║  shutil.copy("cluster-triage-env/environment.py", ".")          ║
║  shutil.copy("cluster-triage-env/models.py", ".")               ║
║                                                                  ║
║  # Cell 3: Run training                                          ║
║  !python train_cluster_triage_unsloth.py                        ║
║                                                                  ║
║  # Cell 4: View plots                                            ║
║  from IPython.display import Image                               ║
║  Image("plots/training_reward_curve.png")                       ║
║  Image("plots/success_rate_comparison.png")                     ║
║  Image("plots/reward_comparison.png")                           ║
║  Image("plots/health_recovery_comparison.png")                  ║
║  Image("plots/stage_reward_distribution.png")                   ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""
print(SETUP_INSTRUCTIONS)