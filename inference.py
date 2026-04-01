import os
import json
import re
from openai import OpenAI
from environment import ClusterTriageEnv
from models import ClusterAction

# ── 1. Load Required Environment Variables ──────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-ai/DeepSeek-R1-Distill-Llama-70B")
MAX_STEPS = 15

if not API_KEY:
    raise EnvironmentError("CRITICAL: HF_TOKEN or API_KEY environment variable is required.")

# ── 2. Setup the LLM Client ──────────────────────────────────────────────────
client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)


def parse_model_action(response_text: str) -> ClusterAction:
    """
    Extracts a ClusterAction from LLM output.
    Handles JSON objects, markdown fences, and [action] tag format.
    """
    text = response_text.replace("```json", "").replace("```", "").strip()

    # Try to parse JSON object
    match = re.search(r'\{.*?\}', text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            if "action_type" in data:
                return ClusterAction(**data)
        except Exception:
            pass

    # Try [action] tag format (from sample inference template)
    match = re.search(r'\[action\]\s*(.*)', text, re.DOTALL | re.IGNORECASE)
    if match:
        inner = match.group(1).strip()
        try:
            data = json.loads(inner)
            return ClusterAction(**data)
        except Exception:
            pass

    return ClusterAction(action_type="noop", target_id="none")


def run_task(env: ClusterTriageEnv, task_id: str) -> float:
    """
    Runs a single task episode with the LLM agent.
    Returns cumulative reward for the episode.
    """
    print(f"\n{'='*60}")
    print(f"  Starting Task: {task_id.upper()}")
    print(f"{'='*60}")

    observation = env.reset(task=task_id)
    history = []
    cumulative_reward = 0.0

    for step in range(1, MAX_STEPS + 1):
        history_text = "\n".join(history) if history else "None."

        system_prompt = (
            "You are an automated DevOps system. You cannot speak. "
            "You can only output raw JSON commands. No explanations, no extra text."
        )

        user_prompt = f"""You are an SRE agent triaging a distributed cluster failure.

CURRENT CLUSTER STATE:
{observation.model_dump_json(indent=2)}

PREVIOUS ACTIONS (do NOT repeat failed actions):
{history_text}

RULES:
1. If there are any hanging jobs, kill ALL of them before doing anything else.
2. Never restart a node whose disk_usage is above 50%. Clear its storage first.
3. Clear nodes in order after all jobs are killed.
4. Only restart nodes after their disk has been cleared.

Respond with EXACTLY ONE JSON object. No other text.
Valid action_type values: "kill_job", "restart_node", "clear_temp_storage", "noop"

EXAMPLE:
{{"action_type": "kill_job", "target_id": "job_rogue_99"}}
"""

        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
            )
            response_text = completion.choices[0].message.content or ""
        except Exception as e:
            print(f"  [API ERROR] Step {step}: {e}")
            break

        action = parse_model_action(response_text)
        print(f"  Step {step:>2} | Action: {action.action_type:<20} | Target: {action.target_id}")

        result = env.step(action)
        observation = result.observation
        reward = result.reward
        done = result.done
        msg = result.info.get("message", "")

        cumulative_reward += reward
        history.append(
            f"Step {step}: {action.action_type} on {action.target_id} "
            f"→ reward={reward:.2f} | msg={msg}"
        )

        print(f"         | Reward: {reward:+.3f} | Done: {done} | {msg}")

        if done:
            print(f"\n  ✓ Task '{task_id.upper()}' completed in {step} step(s).")
            break

        if action.action_type == "noop":
            history.append("WARNING: Last output was invalid JSON. Output ONLY a JSON object.")

    print(f"  Cumulative Reward: {cumulative_reward:.3f}")
    return cumulative_reward


def main():
    """Run baseline inference on all 3 required tasks."""
    env = ClusterTriageEnv()
    tasks = ["easy", "medium", "hard"]

    print("\n" + "="*60)
    print("  OpenEnv: Distributed Cluster Triage — Baseline Run")
    print(f"  Model: {MODEL_NAME}")
    print("="*60)

    scores = {}
    for task_id in tasks:
        scores[task_id] = run_task(env, task_id)

    print("\n" + "="*60)
    print("  BASELINE SCORES SUMMARY")
    print("="*60)
    for task_id, score in scores.items():
        print(f"  {task_id:<12}: {score:.3f}")
    total = sum(scores.values()) / len(scores)
    print(f"  {'AVERAGE':<12}: {total:.3f}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
