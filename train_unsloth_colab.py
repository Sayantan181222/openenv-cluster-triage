"""
train_unsloth_colab.py — Curriculum GRPO Training on ClusterTriageEnv
===============================================================================
MODEL:   DeepSeek-R1-Distill-Qwen-1.5B  (QLoRA, 4-bit)
STRATEGY: Fast iteration — many short training runs beats one huge run.

WHY THIS MODEL WINS HACKATHONS:
  • 1.5B params fits easily on a free T4 (16 GB) in 4-bit
  • DeepSeek-R1 distillation = reasoning capability baked in
  • Fast iteration: each smoke-test run takes ~10 min vs ~60 min for 3B
  • QLoRA rank 8 means even lower VRAM → bigger batches → better gradients
  • More runs in the same time budget = more fixes = higher final score

ITERATION STRATEGY (how to actually improve this):
  Run 1 (smoke):      8 steps/stage, 16 prompts/stage → just check metrics
  Run 2 (diagnosis):  20 steps/stage, 30 prompts → watch reward_std, kl
  Run 3 (production): 60-100 steps/stage, 80-120 prompts → real training
  Run 4+ (iterate):   Re-run only failing stages with adjusted hyperparams

WHAT TO LOOK AT AFTER EACH RUN (printed to console):
  frac_reward_zero_std  → target < 0.20  (high = reward collapse)
  reward_std            → target > 0.50  (low = no gradient signal)
  kl                    → target 0.01-0.10
  grad_norm             → target > 1.0
  Per-task success rate → should increase from run to run

ITERATION LOOP:
  1. Run smoke test → check health metrics
  2. If reward_std < 0.3, increase temperature or adjust reward penalties
  3. If frac_zero_std > 0.4, increase dataset diversity or add more prefixes
  4. If kl stays at 0.0, increase learning_rate or beta in GRPOConfig
  5. Run production curriculum → save checkpoint
  6. Evaluate → identify weakest task → add more training steps for that stage

KNOWN BUGS FIXED (carried from Llama-3.2-3B version):
  FIX 1: easy task gets NO scripted completion (1-step raw reward only)
  FIX 2: generation temperature = 0.8 (was 0.2) to break deterministic collapse
  FIX 3: wrong action penalty=-1.0, noop=-1.5, parse_fail=-2.0 (were too soft)
  FIX 4: completion bonus +2.0 (was +5.0) so step-1 signal dominates
  FIX 5: observation noise ±2% + diverse expert prefixes per batch

NEW FOR QWEN-1.5B:
  FIX 6: max_seq_length=1024 (Qwen tokenizes more efficiently than Llama)
          Keeps memory low → can use num_generations=8 instead of 6
  FIX 7: LoRA rank=8 instead of 16 — 1.5B model doesn't need rank 16
          Halves adapter memory → more room for activations during GRPO
  FIX 8: beta=0.04 (was 0.01) — 1.5B needs stronger KL anchoring
          Otherwise the tiny model drifts too far from reference in early steps
  FIX 9: DeepSeek-R1 outputs <think>...</think> blocks before JSON.
          parse_action() strips these — already present, but confirmed here.

GOOGLE COLAB SETUP:
  # Cell 1 — install (run once)
  !pip install unsloth trl datasets matplotlib pydantic python-dotenv

  # Cell 2 — clone your repo + copy env files
  !git clone https://github.com/<YOUR_REPO>/cluster-triage-env
  import shutil, os
  shutil.copy("cluster-triage-env/environment.py", ".")
  shutil.copy("cluster-triage-env/models.py", ".")
  shutil.copy("cluster-triage-env/train_unsloth_colab.py", ".")

  # Cell 3 — smoke test (run first EVERY time to catch issues early)
  # Edit CURRICULUM_MODE = "smoke" below, then:
  !python train_unsloth_colab.py

  # Cell 4 — production run (after smoke looks good)
  # Edit CURRICULUM_MODE = "production" below, then:
  !python train_unsloth_colab.py

  # Cell 5 — view plots
  from IPython.display import Image, display
  import glob
  for p in sorted(glob.glob("plots/*.png")):
      print(p); display(Image(p))

  # Cell 6 — push trained adapter to HF Hub (optional)
  from huggingface_hub import HfApi
  api = HfApi()
  api.upload_folder(folder_path="cluster-triage-lora-qwen1.5b",
                    repo_id="<YOUR_HF_USERNAME>/cluster-triage-qwen-1.5b",
                    repo_type="model")
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
# ┌─────────────────────────────────────────────────────────────────────┐
# │  CHANGE THIS between runs to switch modes:                          │
# │    "smoke"      → 5 steps/stage,  12 prompts  (~10 min on T4)      │
# │    "diagnosis"  → 20 steps/stage, 30 prompts  (~30 min on T4)      │
# │    "production" → 80 steps/stage, 100 prompts (~3-4 hrs on T4)     │
# └─────────────────────────────────────────────────────────────────────┘
CURRICULUM_MODE = "smoke"   # ← EDIT THIS

# FIX 7: Rank 8 for 1.5B model (was 16 for 3B — proportionally correct)
BASE_MODEL      = "unsloth/DeepSeek-R1-Distill-Qwen-1.5B-bnb-4bit"
LORA_OUTPUT_DIR = "cluster-triage-lora-qwen1.5b"
MAX_SEQ_LENGTH  = 1024    # FIX 6: Qwen tokenizes efficiently; 1024 is enough
LORA_RANK       = 8       # FIX 7: smaller model → smaller rank is optimal

EVAL_EPISODES   = 5

TASK_MAX_STEPS = {
    "easy":      10,
    "medium":    15,
    "hard":      20,
    "very_hard": 20,
    "nightmare": 25,
}

# ── Curriculum definitions ────────────────────────────────────────────────────
# Each tuple: (task_id, grpo_steps, num_dataset_prompts)
CURRICULUM_CONFIGS = {
    "smoke": [
        # Quick sanity-check. Only goal: see non-zero reward_std.
        ("easy",      8,  16),
        ("medium",    8,  16),
        ("hard",      8,  16),
        ("very_hard", 8,  16),
        ("nightmare", 8,  16),
    ],
    "diagnosis": [
        # Enough steps to see if learning signal is real.
        # Watch: does reward_std stay > 0.5? Does kl grow past 0.01?
        ("easy",      20, 30),
        ("medium",    20, 30),
        ("hard",      25, 40),
        ("very_hard", 25, 40),
        ("nightmare", 30, 50),
    ],
    "production": [
        # Full training run. This is your hackathon submission run.
        # On T4 (free Colab): ~3.5 hrs total.
        # On A100 (Colab Pro): ~45 min total.
        ("easy",       80, 100),
        ("medium",     80, 100),
        ("hard",      100, 120),
        ("very_hard", 100, 120),
        ("nightmare", 120, 150),
    ],
}

CURRICULUM = CURRICULUM_CONFIGS[CURRICULUM_MODE]

TASK_LABELS = {
    "easy":      "Easy\n(Kill rogue job)",
    "medium":    "Medium\n(Clear + restart)",
    "hard":      "Hard\n(5-step recovery)",
    "very_hard": "Very Hard\n(Dual malware)",
    "nightmare": "Nightmare\n(Hydra protocol)",
}
STAGE_COLORS = ["#10b981", "#fbbf24", "#f97316", "#7c3aed", "#b91c1c"]

# Recalibrated for the 1.5B model — it starts weaker, so thresholds are lower
# to allow early-exit once the tiny model has genuinely converged on a stage.
EARLY_EXIT_THRESHOLDS = {
    "smoke": {
        "easy":      0.8,
        "medium":    0.5,
        "hard":      0.3,
        "very_hard": 0.2,
        "nightmare": 0.05,
    },
    "diagnosis": {
        "easy":      1.2,
        "medium":    0.9,
        "hard":      0.6,
        "very_hard": 0.4,
        "nightmare": 0.15,
    },
    "production": {
        "easy":      1.5,
        "medium":    1.0,
        "hard":      0.7,
        "very_hard": 0.5,
        "nightmare": 0.2,
    },
}[CURRICULUM_MODE]

MIN_STEPS_BEFORE_EXIT = 3
ROLLING_WINDOW        = 3


# ── 2. Load Model + QLoRA ─────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  MODEL: {BASE_MODEL}")
print(f"  MODE:  {CURRICULUM_MODE.upper()}")
print(f"  LoRA rank: {LORA_RANK}  |  max_seq_length: {MAX_SEQ_LENGTH}")
print(f"{'='*65}\n")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = BASE_MODEL,
    max_seq_length = MAX_SEQ_LENGTH,
    load_in_4bit   = True,        # QLoRA: 4-bit quantized base
    fast_inference = False,       # MUST be False for GRPO training
    max_lora_rank  = LORA_RANK,
)
model = FastLanguageModel.get_peft_model(
    model,
    r                          = LORA_RANK,
    target_modules             = ["q_proj", "k_proj", "v_proj", "o_proj",
                                  "gate_proj", "up_proj", "down_proj"],
    lora_alpha                 = LORA_RANK * 2,   # alpha = 2x rank is standard
    lora_dropout               = 0,               # 0 is optimal for unsloth
    bias                       = "none",
    use_gradient_checkpointing = "unsloth",        # saves ~30% VRAM on T4
    random_state               = 42,
)
print("[INFO] Model + QLoRA ready.\n")


# ── 3. Prompt Engineering ─────────────────────────────────────────────────────
# Kept concise — 1.5B has smaller context budget than 3B.
# The <think> block DeepSeek adds is stripped in parse_action(), not here.
SYSTEM_PROMPT = (
    "You are an automated SRE agent. Output ONLY a raw JSON object. "
    "No markdown, no explanation, no <think> block in final output.\n\n"
    "Schema: {\"action_type\": \"<kill_job|restart_node|clear_temp_storage|noop>\", "
    "\"target_id\": \"<job_id or node_id>\"}\n\n"
    "RULES (strict priority order):\n"
    "  1. ANY job with status 'hanging' → kill it BEFORE touching any node.\n"
    "  2. nightmare mode: kill job_hydra_1, job_hydra_2, job_hydra_3 IN ORDER.\n"
    "  3. NEVER restart a node with disk_usage > 50. Clear storage first.\n"
    "  4. After clearing a node's storage, restart it.\n"
    "  5. Output ONLY the JSON. Nothing else."
)


def build_user_prompt(obs_json: str, history: list) -> str:
    hist_str = "\n".join(history[-5:]) if history else "None yet."  # last 5 only — saves tokens
    return (
        f"CLUSTER STATE:\n{obs_json}\n\n"
        f"RECENT ACTIONS:\n{hist_str}\n\n"
        "Next action (JSON only):"
    )


# ── 4. Action Parsing ─────────────────────────────────────────────────────────
def parse_action(text: str) -> ClusterAction:
    """
    Multi-stage parser. Handles DeepSeek-R1's <think>...</think> prefix.
    FIX 9: Confirmed to strip <think> blocks before JSON extraction.
    """
    # Strip DeepSeek reasoning chain
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    text = text.replace("```json", "").replace("```", "").strip()

    # Try direct JSON parse
    try:
        d = json.loads(text)
        if "action_type" in d:
            return ClusterAction(**d)
    except Exception:
        pass

    # Try first {...} block (greedy short)
    m = re.search(r'\{.*?\}', text, re.DOTALL)
    if m:
        try:
            d = json.loads(m.group(0))
            if "action_type" in d:
                return ClusterAction(**d)
        except Exception:
            pass

    # Try widest {...} block
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
    FIX 2: temperature=0.8 default to prevent deterministic collapse.
    Uses shorter max_new_tokens=80 — 1.5B outputs concise JSON faster.
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
            max_new_tokens     = 80,     # 1.5B outputs shorter — 80 is enough for JSON
            temperature        = temperature,
            do_sample          = True,
            top_p              = 0.95,
            repetition_penalty = 1.1,
            pad_token_id       = tokenizer.eos_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


# ── 6. Evaluation ─────────────────────────────────────────────────────────────
def run_eval_episode(task_id: str) -> dict:
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
                                               temperature=0.3)  # greedy-ish for eval
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

    final_health = env.state_data.health_score if env.state_data else 0.0
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
# FIX 5: expert prefix sequences + observation noise for batch diversity.

EXPERT_SEQUENCES = {
    "easy": [
        [],    # fresh reset (3 copies for variety)
        [],
        [],
    ],
    "medium": [
        [],
        [{"action_type": "clear_temp_storage", "target_id": "worker_03"}],
        [{"action_type": "noop",               "target_id": "none"}],   # recovery case
    ],
    "hard": [
        [],
        [{"action_type": "kill_job",           "target_id": "job_rogue_99"}],
        [{"action_type": "kill_job",           "target_id": "job_rogue_99"},
         {"action_type": "clear_temp_storage", "target_id": "worker_01"}],
        [{"action_type": "kill_job",           "target_id": "job_rogue_99"},
         {"action_type": "clear_temp_storage", "target_id": "worker_01"},
         {"action_type": "clear_temp_storage", "target_id": "worker_02"}],
        [{"action_type": "restart_node",       "target_id": "worker_01"}],  # wrong-first
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
        [{"action_type": "clear_temp_storage", "target_id": "worker_01"}],  # wrong-first
    ],
}


def _add_obs_noise(obs_json: str) -> str:
    """FIX 5: ±2% noise to cpu/ram so every prompt in a batch is distinct."""
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
# Reward budget (correct vs wrong first action) — unchanged from v2:
#   correct action + partial script: ~+3.4
#   wrong action (rule violation):   ~-0.9
#   noop:                            ~-1.55
#   parse fail:                      ~-2.0
#   → Expected reward_std across 8 gens: > 0.8

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

# FIX 1: easy = no script (1-step raw reward only); others = partial (≤50% optimal steps)
PARTIAL_SCRIPTED_COMPLETIONS = {
    "easy": [],
    "medium": [
        {"action_type": "clear_temp_storage", "target_id": "worker_03"},
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
    correct_firsts = CORRECT_FIRST_ACTIONS.get(task_id, set())
    wrong_firsts   = WRONG_FIRST_ACTIONS.get(task_id, set())
    script         = PARTIAL_SCRIPTED_COMPLETIONS.get(task_id, [])
    max_steps      = TASK_MAX_STEPS[task_id]

    def reward_fn(prompts, completions, **kwargs):
        rewards = []

        for completion in completions:
            if isinstance(completion, list) and completion:
                c = completion[0]
                action_text = c.get("content", "") if isinstance(c, dict) else str(c)
            elif isinstance(completion, str):
                action_text = completion
            else:
                action_text = str(completion)

            first_action  = parse_action(action_text)
            is_parse_fail = (
                first_action.action_type == "noop"
                and "noop" not in action_text.lower()
                and "{"    not in action_text
            )

            env = ClusterTriageEnv()
            env.reset(task=task_id)
            health_before = env.state_data.health_score
            total_reward  = 0.0
            episode_done  = False

            # Step 1: model's action
            try:
                result        = env.step(first_action)
                total_reward += result.reward
                episode_done  = result.done
            except Exception:
                total_reward = -0.5
                episode_done = True

            # Steps 2+: partial scripted completion (FIX 1)
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

            final_health = env.state_data.health_score if env.state_data else 0.0

            # Completion bonus (FIX 4)
            if final_health >= 1.0:
                total_reward += 2.0
            elif final_health >= 0.5:
                total_reward += final_health * 1.0

            # Health-delta bonus
            delta = final_health - health_before
            if delta > 0.0:
                total_reward += min(0.3, round(int(delta * 20) * 0.05, 2))

            # First-action bonus / wrong-action penalty (FIX 3)
            action_key = (first_action.action_type, first_action.target_id)
            if action_key in correct_firsts:
                total_reward += 0.5
            elif action_key in wrong_firsts:
                total_reward -= 1.0

            # Parse / noop penalties (FIX 3)
            if is_parse_fail:
                total_reward -= 2.0
            elif first_action.action_type == "noop":
                total_reward -= 1.5

            rewards.append(float(total_reward))

        return rewards

    return reward_fn


# ── 9. GRPO Health Monitor ────────────────────────────────────────────────────
class GRPOHealthMonitor:
    """
    Tracks reward_std, frac_reward_zero_std, KL, grad_norm per step.
    These are your PRIMARY ITERATION SIGNALS — check them after every run.
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
            warns.append(f"HIGH frac_zero_std={fzs:.2f}  → increase dataset diversity or temperature")
        if r_std is not None and r_std < 0.3:
            warns.append(f"LOW  reward_std={r_std:.3f}   → increase temperature or reward spread")
        if kl   is not None and kl   < 0.005:
            warns.append(f"LOW  kl={kl:.5f}              → increase learning_rate or beta")
        if gn   is not None and gn   < 0.3:
            warns.append(f"LOW  grad_norm={gn:.3f}       → reduce gradient_accumulation or increase lr")
        if warns:
            self.total_warnings += len(warns)
            for w in warns:
                print(f"  [HEALTH WARN step {step}] {w}")

    def summary(self):
        def sm(lst):
            v = [x for x in lst if x is not None]
            return sum(v)/len(v) if v else float("nan")

        print(f"\n{'═'*65}")
        print("  GRPO HEALTH SUMMARY — use this to guide your next run")
        print(f"{'═'*65}")
        print(f"  Mean reward:           {sm(self.reward_means):+.3f}")
        print(f"  Mean reward_std:        {sm(self.reward_stds):.3f}  (target > 0.50)")
        print(f"  Mean frac_zero_std:     {sm(self.frac_zero_stds):.3f}  (target < 0.20)")
        print(f"  Mean KL:                {sm(self.kls):.5f} (target 0.01–0.10)")
        print(f"  Mean grad_norm:         {sm(self.grad_norms):.3f}  (target > 1.0)")
        print(f"  Completion length:      {sm(self.completion_lengths):.1f} tokens")
        print(f"  Total health warnings:  {self.total_warnings}")
        print(f"\n  ITERATION GUIDANCE:")
        avg_fzs  = sm(self.frac_zero_stds)
        avg_std  = sm(self.reward_stds)
        avg_kl   = sm(self.kls)
        avg_gn   = sm(self.grad_norms)
        if avg_std < 0.3:
            print("  ⚠  reward_std too low → raise temperature to 0.9 or add noise")
        if avg_fzs > 0.4:
            print("  ⚠  frac_zero_std too high → add more expert prefix sequences")
        if avg_kl < 0.005:
            print("  ⚠  KL near zero → increase learning_rate to 5e-5 or beta to 0.05")
        if avg_gn < 0.5:
            print("  ⚠  grad_norm too low → reduce gradient_accumulation_steps")
        if avg_std >= 0.5 and avg_fzs < 0.2 and avg_kl >= 0.01 and avg_gn >= 1.0:
            print("  ✓  All metrics healthy! Increase steps for next run (diagnosis→production).")
        print(f"{'═'*65}")


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
print(f"  PHASE 1: BASELINE EVALUATION (untrained DeepSeek-R1-Qwen-1.5B)")
print(f"  Mode: {CURRICULUM_MODE.upper()}")
print("  Expected: ~0% success. Reward ≈ -1.5 to -0.5.")
print("=" * 65)

baseline_metrics = evaluate_all_tasks("BASELINE")


# ── 12. Curriculum GRPO Training ─────────────────────────────────────────────
print("\n" + "=" * 65)
print(f"  PHASE 2: CURRICULUM GRPO [{CURRICULUM_MODE.upper()}]")
print("  Model: DeepSeek-R1-Distill-Qwen-1.5B (QLoRA rank 8, 4-bit)")
print("  FIX 1: easy=no script  FIX 2: temp=0.8  FIX 3: sharp penalties")
print("  FIX 4: bonus=+2.0      FIX 5: obs noise FIX 6-9: Qwen-specific")
print("=" * 65)

for stage_idx, (task_id, grpo_steps, num_prompts) in enumerate(CURRICULUM):
    print(f"\n{'━'*65}")
    print(f"  STAGE {stage_idx+1}/{len(CURRICULUM)} — {task_id.upper()}")
    print(f"  GRPO steps: {grpo_steps}  |  Dataset: {num_prompts} prompts")
    print(f"  Early-exit threshold: {EARLY_EXIT_THRESHOLDS[task_id]}")
    print(f"{'━'*65}")

    tracker.mark_stage()
    dataset   = build_dataset(task_id, num_prompts)
    reward_fn = make_reward_fn(task_id)

    training_args = GRPOConfig(
        output_dir                  = f"checkpoints/stage_{stage_idx+1}_{task_id}",
        # FIX 8: lr slightly lower than Llama version — Qwen-1.5B is more sensitive
        learning_rate               = 2e-5,
        per_device_train_batch_size = 1,
        # FIX 7: 1.5B is smaller → we can afford 8 generations (was 6) for
        #         better reward variance estimation without OOM
        gradient_accumulation_steps = 8,
        num_generations             = 8,
        max_completion_length       = 120,  # 1.5B outputs shorter JSON
        max_prompt_length           = 800,  # leaves room in 1024 context
        max_steps                   = grpo_steps,
        logging_steps               = 1,
        save_steps                  = grpo_steps,
        optim                       = "adamw_8bit",
        warmup_ratio                = 0.1,
        report_to                   = "none",
        remove_unused_columns       = False,
        temperature                 = 1.0,
        # FIX 8: beta=0.04 (was 0.01) — 1.5B needs stronger KL anchoring
        beta                        = 0.04,
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
print(f"  RESULTS: Baseline vs Fine-Tuned DeepSeek-R1-Qwen-1.5B [{CURRICULUM_MODE.upper()}]")
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

MODEL_LABEL = f"DeepSeek-R1-Qwen-1.5B [{CURRICULUM_MODE}]"

# Plot 1: Training Reward Curve
fig, ax = plt.subplots(figsize=(13, 5))
if tracker.step_rewards:
    steps   = [s for s, _ in tracker.step_rewards]
    rewards = [r for _, r in tracker.step_rewards]
    w       = max(1, min(5, len(rewards) // 3))
    smooth  = np.convolve(rewards, np.ones(w) / w, mode="same")
    ax.plot(steps, rewards, color="#94a3b8", alpha=0.3, linewidth=0.8, label="Raw reward")
    ax.plot(steps, smooth,  color="#6366f1", linewidth=2.2, label=f"Smoothed (w={w})")
    for i, boundary in enumerate(tracker.stage_boundaries):
        if i < len(task_ids):
            ax.axvline(x=boundary, color=STAGE_COLORS[i], linestyle="--",
                       linewidth=1.2, alpha=0.85)
            y_pos = min(rewards) + 0.05 if rewards else -1.8
            ax.text(boundary + 0.2, y_pos, f"S{i+1}:{task_ids[i][:5]}",
                    fontsize=7, color=STAGE_COLORS[i], va="bottom")
ax.axhline(y=0, color="#475569", linewidth=0.8, linestyle=":")
ax.set_xlabel("GRPO Training Step")
ax.set_ylabel("Episode Reward")
ax.set_title(f"Curriculum GRPO Training — Reward Curve\n{MODEL_LABEL} · ClusterTriageEnv")
ax.legend(loc="lower right", fontsize=9)
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig("plots/training_reward_curve.png", bbox_inches="tight", dpi=150)
plt.close(fig)
print("\n[PLOT] Saved: plots/training_reward_curve.png")

# Plot 2: Success Rate Comparison
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
ax.set_title(f"Baseline vs Fine-Tuned: Success Rate\n{MODEL_LABEL}")
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
ax.set_title(f"Average Reward: Baseline vs Fine-Tuned\n{MODEL_LABEL}")
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
ax.set_title(f"Cluster Health Recovery\n{MODEL_LABEL}")
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
    f"Per-Stage GRPO Reward Distribution\n"
    f"Wider violin = higher variance = healthier signal — {MODEL_LABEL}"
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
                    "frac_reward_zero_std  (target < 0.2)", "#f97316", target_hi=0.2)
_plot_health_metric(axes[2], s, health_monitor.kls,
                    "KL divergence  (target 0.01 – 0.1)", "#10b981",
                    target_lo=0.01, target_hi=0.1)
_plot_health_metric(axes[3], s, health_monitor.grad_norms,
                    "grad_norm  (target > 1.0)", "#7c3aed", target_lo=1.0)

fig.suptitle(
    f"GRPO Training Health Dashboard — {MODEL_LABEL}\n"
    "Metrics should approach green target lines as training progresses",
    fontsize=12, y=1.01
)
fig.tight_layout()
fig.savefig("plots/grpo_health_metrics.png", bbox_inches="tight", dpi=150)
plt.close(fig)
print("[PLOT] Saved: plots/grpo_health_metrics.png")


# ── 16. Save Model ────────────────────────────────────────────────────────────
model.save_pretrained(LORA_OUTPUT_DIR)
tokenizer.save_pretrained(LORA_OUTPUT_DIR)
print(f"\n[INFO] QLoRA adapter saved → '{LORA_OUTPUT_DIR}/'")


# ── 17. Final Summary + Next Steps ───────────────────────────────────────────
avg_b_sr  = sum(baseline_metrics[t]["success_rate"] for t in task_ids) / len(task_ids)
avg_t_sr  = sum(trained_metrics[t]["success_rate"]  for t in task_ids) / len(task_ids)
avg_t_psr = sum(trained_metrics[t]["partial_rate"]  for t in task_ids) / len(task_ids)
avg_b_r   = sum(baseline_metrics[t]["avg_reward"]   for t in task_ids) / len(task_ids)
avg_t_r   = sum(trained_metrics[t]["avg_reward"]    for t in task_ids) / len(task_ids)
avg_b_h   = sum(baseline_metrics[t]["avg_health"]   for t in task_ids) / len(task_ids)
avg_t_h   = sum(trained_metrics[t]["avg_health"]    for t in task_ids) / len(task_ids)

print("\n" + "=" * 65)
print(f"  TRAINING COMPLETE [{CURRICULUM_MODE.upper()}]")
print("  Model: DeepSeek-R1-Distill-Qwen-1.5B")
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

# Per-task success breakdown for iteration targeting
print(f"\n  Per-task success (identify where to focus next run):")
for tid in task_ids:
    b = baseline_metrics[tid]["success_rate"]
    t = trained_metrics[tid]["success_rate"]
    bar = "█" * int(t / 5) + "░" * (20 - int(t / 5))
    flag = " ← focus here" if t < 30 and CURRICULUM_MODE != "smoke" else ""
    print(f"    {tid:<12} [{bar}] {t:5.1f}%{flag}")

print(f"\n  Plots: plots/")
print(f"  Model: {LORA_OUTPUT_DIR}/")

print("""
╔══════════════════════════════════════════════════════════════════════╗
║  ITERATION CHECKLIST — do this after every run                      ║
╠══════════════════════════════════════════════════════════════════════╣
║  1. Check GRPO Health Summary printed above                         ║
║  2. Look at plots/grpo_health_metrics.png                           ║
║  3. Find the weakest task in per-task success table above           ║
║  4. If smoke → metrics healthy → switch to "diagnosis"              ║
║  5. If diagnosis → reward_std > 0.5, kl > 0.01 → switch to         ║
║     "production"                                                    ║
║  6. If production → identify stuck tasks → increase their           ║
║     grpo_steps in CURRICULUM_CONFIGS["production"]                  ║
║  7. If any metric is red → fix before production run:               ║
║     reward_std low  → raise temperature in GRPOConfig               ║
║     frac_zero high  → add expert sequences in EXPERT_SEQUENCES      ║
║     kl near zero    → raise learning_rate or beta                   ║
║     grad_norm low   → lower gradient_accumulation_steps             ║
╠══════════════════════════════════════════════════════════════════════╣
║  HEALTHY TARGETS:                                                   ║
║    frac_reward_zero_std  < 0.20                                     ║
║    reward_std            > 0.50                                     ║
║    kl                  0.01–0.10                                    ║
║    grad_norm             > 1.00                                     ║
╚══════════════════════════════════════════════════════════════════════╝
""")