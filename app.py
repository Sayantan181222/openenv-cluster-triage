"""
app.py — OpenEnv Cluster Triage Environment
Serves both:
  • FastAPI HTTP API (/reset, /step, /state, /health) — for openenv validate & agents
  • Gradio UI (/ui) — for interactive human demonstration
All on port 7860 for Hugging Face Spaces.
"""
import os
import json
import re
from dotenv import load_dotenv

import gradio as gr
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
from openai import OpenAI

from environment import ClusterTriageEnv
from models import ClusterAction, ResetRequest

load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-ai/DeepSeek-R1-Distill-Llama-70B")
API_KEY = os.getenv("HF_TOKEN", "").strip().strip('"').strip("'")

if API_KEY:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    print(f"[INFO] LLM client ready. Token prefix: {API_KEY[:6]}... Model: {MODEL_NAME}")
else:
    client = None
    print("[WARN] HF_TOKEN not set. LLM calls will fail. Set HF_TOKEN in Space secrets.")

# ── Shared Environment Instance (for HTTP API) ───────────────────────────────
api_env = ClusterTriageEnv()

# ── FastAPI App ──────────────────────────────────────────────────────────────
fastapi_app = FastAPI(
    title="OpenEnv: Distributed Cluster Triage",
    description=(
        "An OpenEnv-compliant RL environment simulating a 4-node enterprise "
        "data cluster. An AI agent acting as an SRE must triage infrastructure "
        "failures by issuing precise commands."
    ),
    version="1.0.0",
)


@fastapi_app.get("/health")
def health():
    """Health check endpoint — returns 200 OK."""
    return {"status": "ok", "env": "cluster-triage", "version": "1.0.0"}


@fastapi_app.post("/reset")
def reset(request: ResetRequest = None):
    """
    Reset the environment for a given task.
    Body: {"task": "easy" | "medium" | "hard" | "very_hard" | "nightmare"}
    Returns the initial ClusterObservation.
    """
    task = (request.task if request else "easy")
    if task not in ["easy", "medium", "hard", "very_hard", "nightmare"]:
        raise HTTPException(status_code=400, detail=f"Unknown task '{task}'. Valid: easy, medium, hard, very_hard, nightmare")
    obs = api_env.reset(task=task)
    return JSONResponse(content=obs.model_dump())


@fastapi_app.post("/step")
def step(action: ClusterAction):
    """
    Execute one action in the environment.
    Body: {"action_type": "kill_job", "target_id": "job_rogue_99"}
    Returns StepResult with observation, reward, done, info.
    """
    result = api_env.step(action)
    return JSONResponse(content=result.model_dump())


@fastapi_app.get("/state")
def state():
    """Return the current cluster observation without advancing the episode."""
    return JSONResponse(content=api_env.state().model_dump())


@fastapi_app.get("/tasks")
def list_tasks():
    """List all available tasks with difficulty metadata."""
    return {
        "tasks": [
            {"id": "easy",      "difficulty": 1, "max_steps": 5,  "description": "Kill a single rogue job"},
            {"id": "medium",    "difficulty": 2, "max_steps": 10, "description": "Clear disk and restart node"},
            {"id": "hard",      "difficulty": 3, "max_steps": 15, "description": "Cascade failure recovery"},
            {"id": "very_hard", "difficulty": 4, "max_steps": 20, "description": "Multi-vector attack"},
            {"id": "nightmare", "difficulty": 5, "max_steps": 25, "description": "Hydra Protocol — total collapse"},
        ]
    }


# ── Task Briefings for UI ────────────────────────────────────────────────────
TASK_DESCRIPTIONS = {
    "Easy": (
        "🟢 EASY — The Stuck Job\n\n"
        "A single rogue MapReduce job is stuck in an infinite loop, consuming all RAM.\n\n"
        "Goal: Kill the rogue job (job_rogue_99).\n"
        "Expected Steps: 1"
    ),
    "Medium": (
        "🟡 MEDIUM — The Full Disk\n\n"
        "Worker node worker_03 has hit 99.9% disk capacity. All writes are failing.\n\n"
        "Goal: Clear the storage, then restart the node safely.\n"
        "Expected Steps: 2"
    ),
    "Hard": (
        "🟠 HARD — The Cascade Failure\n\n"
        "job_rogue_99 caused a HDFS replication storm, crashing Nodes 1 & 2.\n"
        "Failover traffic is overloading Nodes 3 & 4.\n\n"
        "Goal: Kill root job → clear worker_01 & worker_02 → restart both.\n"
        "Expected Steps: 5"
    ),
    "Very_hard": (
        "🔴 VERY HARD — Multi-Vector Attack\n\n"
        "Two simultaneous malware jobs: a disk-filling log spammer (worker_01/02) "
        "and a RAM-hogging crypto miner (worker_03/04).\n\n"
        "Goal: Kill BOTH jobs → clear dead nodes → reboot them.\n"
        "Expected Steps: 6"
    ),
    "Nightmare": (
        "☠️ NIGHTMARE — The Hydra Protocol\n\n"
        "Total cluster collapse. All 4 nodes offline. THREE self-replicating malware "
        "jobs are writing garbage data to every disk simultaneously.\n\n"
        "⚠️ WARNING: If ANY Hydra survives, it will instantly refill cleared storage!\n\n"
        "Goal: Kill ALL 3 Hydras → clear all 4 nodes → restart all 4 nodes.\n"
        "Expected Steps: 11"
    ),
}


def extract_action_from_llm(response_text: str) -> ClusterAction:
    """Parse LLM output into a ClusterAction."""
    text = response_text.replace("```json", "").replace("```", "").strip()
    match = re.search(r'\{.*?\}', text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            if "action_type" in data:
                return ClusterAction(**data)
        except Exception:
            pass
    return ClusterAction(action_type="noop", target_id="none")


# ── RL Interaction Functions & Visual Widgets ────────────────────────────────

def get_node_status_html(obs):
    """Generates a visual telemetry grid for the 4 cluster nodes to fill the empty middle space."""
    if not obs:
        return "<div style='color:#71717a; font-style:italic;'>System offline. Awaiting reset...</div>"
    
    html = '<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 10px;">'
    for n in obs.nodes:
        if n.status == "healthy":
            color = "#10b981" # Green
        elif n.status == "offline":
            color = "#ef4444" # Red
        else:
            color = "#eab308" # Yellow (High Mem / Disk Full)
            
        html += f"""
        <div style="background: #18181b; border: 1px solid #3f3f46; border-left: 4px solid {color}; padding: 12px; border-radius: 6px;">
            <div style="font-family: 'IBM Plex Mono', monospace; font-size: 13px; font-weight: bold; color: #e4e4e7;">{n.node_id}</div>
            <div style="font-size: 11px; font-weight: 600; color: {color}; text-transform: uppercase; margin-bottom: 6px;">{n.status}</div>
            <div style="font-size: 11px; color: #a1a1aa; display: flex; justify-content: space-between;">
                <span>Disk: {n.disk_usage}%</span><span>RAM: {n.ram_usage}%</span>
            </div>
        </div>
        """
    html += '</div>'
    return html

def get_score_html(reward, health, reward_history):
    """Generates the Episode Score widget AND the Reward History Bar Graph."""
    r_color = "#10b981" if reward > 0 else ("#ef4444" if reward < 0 else "#a1a1aa")
        
    # Build the Bar Graph HTML
    bars_html = ""
    display_history = reward_history[-8:] # Show max 8 bars to fit the box
    for r in display_history:
        bar_color = "#10b981" if r >= 0 else "#ef4444"
        # Map reward magnitude (-1.0 to 1.0) to a CSS height percentage
        h_pct = max(10, int(abs(r) * 100))
        bars_html += f'<div style="flex: 1; background-color: {bar_color}; height: {h_pct}%; border-radius: 2px 2px 0 0; opacity: 0.85;"></div>'
        
    avg_reward = sum(reward_history)/len(reward_history) if reward_history else 0.0

    return f"""
    <div style="padding: 5px 0;">
        <div style="text-align: center;">
            <div style="font-family: 'IBM Plex Mono', 'Courier New', monospace; font-size: 56px; font-weight: 700; color: {r_color}; line-height: 1;">
                {reward:+.3f}
            </div>
            <div style="color: #71717a; font-size: 12px; text-transform: uppercase; letter-spacing: 2px; margin-top: 10px; font-weight: 600;">Step Reward</div>
        </div>
        
        <div style="margin-top: 25px; padding-top: 15px; border-top: 1px solid #27272a; display: flex; justify-content: space-between; align-items: center;">
            <span style="color: #a1a1aa; font-size: 13px; text-transform: uppercase; letter-spacing: 1px;">Overall Health</span>
            <span style="color: #10b981; font-family: 'IBM Plex Mono', monospace; font-size: 22px; font-weight: bold;">{health:.2f}</span>
        </div>

        <div style="margin-top: 30px;">
           <div style="color: #71717a; font-size: 12px; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 15px; font-weight: 600;">Reward History</div>
           <div style="display: flex; align-items: flex-end; height: 60px; gap: 6px; border-bottom: 1px solid #3f3f46; padding-bottom: 2px; background: rgba(0,0,0,0.2); padding: 10px 10px 0 10px; border-radius: 6px 6px 0 0;">
               {bars_html}
           </div>
           <div style="display: flex; justify-content: space-between; color: #71717a; font-size: 10px; font-family: 'IBM Plex Mono', monospace; margin-top: 6px;">
               <span>+1.0</span><span>avg: {avg_reward:.3f}</span><span>-1.0</span>
           </div>
        </div>
    </div>
    """

def ui_reset(task_name):
    env = ClusterTriageEnv()
    obs = env.reset(task_name.lower())
    action_history = []
    reward_history = [] # Wipe the history for the bar graph
    
    log = f"{'='*50}\n  🔄 SYSTEM RESET — Threat Level: {task_name.upper()}\n{'='*50}\n\n"
    log += f"Initial Health Score: {obs.health_score:.2f}\n"
    log += f"Active Jobs: {[j.job_id for j in obs.active_jobs]}\n"
    log += "Awaiting agent... Click '▶ AGENT STEP' to make one move.\n"
    
    # Return 6 items to update the UI
    return env, action_history, reward_history, log, get_score_html(0.000, obs.health_score, []), get_node_status_html(obs)

def ui_step(env, action_history, reward_history, current_log, task_name):
    if env is None:
        return env, action_history, reward_history, current_log + "\n[ERROR] System offline. Click 🔄 RESET first.\n", get_score_html(0.0, 0.0, []), get_node_status_html(None)
    if not client:
        return env, action_history, reward_history, current_log + "\n[ERROR] HF_TOKEN missing.\n", get_score_html(0.0, env.state().health_score, reward_history), get_node_status_html(env.state())
        
    obs = env.state()
    if obs.health_score >= 1.0 or (env.step_count >= env.max_steps):
        return env, action_history, reward_history, current_log + "\n[INFO] Incident resolved or timed out. Please RESET.\n", get_score_html(0.0, obs.health_score, reward_history), get_node_status_html(obs)
        
    step_num = env.step_count + 1
    history_text = "\n".join(action_history) if action_history else "None."
    
    prompt = f"""You are an automated DevOps system. Output ONLY raw JSON. No explanations.
CURRENT CLUSTER STATE:
{obs.model_dump_json(indent=2)}
PREVIOUS ACTIONS:
{history_text}
RULES:
1. Kill ALL hanging jobs before clearing any storage.
2. Never restart a node with disk_usage > 50%. Clear it first.
3. For nightmare task: kill all 3 hydras before ANYTHING else.
Respond with EXACTLY ONE JSON object using this exact format:
{{"action_type": "<insert_action_here>", "target_id": "<insert_target_here>"}}
Valid action_type values: "kill_job", "restart_node", "clear_temp_storage", "noop".
"""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME, messages=[{"role": "user", "content": prompt}], temperature=0.1,
        )
        action = extract_action_from_llm(response.choices[0].message.content.strip())
        result = env.step(action)
        new_obs = result.observation
        msg = result.info.get("message", "")
        
        # Track the new reward for the bar chart
        reward_history.append(result.reward)
        
        log_add = f"Step {step_num:>2} | {action.action_type:<22} → {action.target_id}\n"
        log_add += f"       | reward={result.reward:+.3f} | health={new_obs.health_score:.2f} | {msg}\n"
        action_history.append(f"Step {step_num}: {action.action_type} on {action.target_id} → reward={result.reward:.2f} | {msg}")

        if result.done:
            log_add += f"\n{'='*50}\n  {'✅ SYSTEM NOMINAL' if new_obs.health_score >= 1.0 else '❌ EPISODE ENDED'}\n{'='*50}\n"
            
        return env, action_history, reward_history, current_log + log_add, get_score_html(result.reward, new_obs.health_score, reward_history), get_node_status_html(new_obs)

    except Exception as e:
        return env, action_history, reward_history, current_log + f"\n[API ERROR]: {str(e)}\n", get_score_html(0.0, obs.health_score, reward_history), get_node_status_html(obs)

def ui_get_state(env, current_log):
    if env is None: return current_log + "\n[ERROR] System offline. Click 🔄 RESET.\n"
    obs = env.state()
    return current_log + f"\n--- 📊 STATE SNAPSHOT (Step {env.step_count}) ---\nHealth: {obs.health_score:.2f} | Nodes: {', '.join([f'{n.node_id}({n.status})' for n in obs.nodes])}\n"


# ── COMMAND CENTER UI ────────────────────────────────────────────────────────
CUSTOM_CSS = """
body, html, .gradio-container { background-color: #09090b !important; color: #e4e4e7 !important; }
.panel-card { background-color: #121214 !important; border: 1px solid #27272a !important; border-radius: 8px !important; padding: 20px !important; box-shadow: 0 4px 20px rgba(0,0,0,0.5) !important; }
.sidebar-title h1 { color: #f4f4f5 !important; font-size: 20px !important; letter-spacing: 1px; border-bottom: 1px solid #27272a; padding-bottom: 10px; margin-bottom: 20px !important; }
#task-selector .wrap { display: flex !important; flex-direction: column !important; gap: 10px !important; }
#task-selector label { background-color: #18181b !important; border: 1px solid #3f3f46 !important; border-radius: 6px !important; padding: 10px 15px !important; cursor: pointer; }
#task-selector label:hover { border-color: #8b5cf6 !important; background-color: #27272a !important; }
#task-briefing textarea { background-color: #18181b !important; color: #a1a1aa !important; border: 1px solid #3f3f46 !important; border-radius: 6px !important; font-size: 14px !important; padding: 20px !important; }
#console-logs textarea { background-color: #000000 !important; color: #10b981 !important; font-family: 'IBM Plex Mono', monospace !important; font-size: 13px !important; border: 1px solid #27272a !important; padding: 15px !important; }
.gradio-container .label { display: none !important; }
.action-row { margin-top: 15px !important; gap: 10px !important; }
.action-btn { font-weight: bold !important; border-radius: 6px !important; text-transform: uppercase; letter-spacing: 1px; }
.step-btn { background: #8b5cf6 !important; color: white !important; border: none !important; }
.step-btn:hover { background: #7c3aed !important; }
.util-btn { background: #27272a !important; color: #a1a1aa !important; border: 1px solid #3f3f46 !important; }
.util-btn:hover { background: #3f3f46 !important; color: white !important; }
"""

with gr.Blocks(css=CUSTOM_CSS, theme=gr.themes.Monochrome(text_size="sm")) as gradio_ui:
    
    session_env = gr.State(None)
    session_history = gr.State([])
    session_reward_history = gr.State([]) # NEW: State to track bar graph metrics
    
    with gr.Row():
        # COLUMN 1: SIDEBAR
        with gr.Column(scale=3, elem_classes="panel-card"):
            gr.Markdown("# 🌐 OpenEnv\n**Cluster SRE**", elem_classes="sidebar-title")
            task_selector = gr.Radio(choices=["Easy", "Medium", "Hard", "Very_hard", "Nightmare"], value="Easy", label="Environment Scenarios", elem_id="task-selector")
            with gr.Row(elem_classes="action-row"):
                reset_btn = gr.Button("🔄 Reset", elem_classes=["action-btn", "util-btn"])
                state_btn = gr.Button("📊 State", elem_classes=["action-btn", "util-btn"])
            step_btn = gr.Button("▶ AGENT STEP", elem_classes=["action-btn", "step-btn"])

        # COLUMN 2: BRIEFING & TELEMETRY
        with gr.Column(scale=5, elem_classes="panel-card"):
            gr.Markdown("### 📋 ACTIVE INCIDENT BRIEFING", elem_classes="sidebar-title")
            task_desc = gr.Textbox(value=TASK_DESCRIPTIONS["Easy"], interactive=False, lines=6, elem_id="task-briefing")
            
            # NEW: The Node Status Grid directly below the briefing
            gr.Markdown("### 🖥️ LIVE NODE STATUS", elem_classes="sidebar-title", elem_id="telemetry-title")
            node_display = gr.HTML(value=get_node_status_html(None))

        # COLUMN 3: SCORE & BAR GRAPH
        with gr.Column(scale=3, elem_classes="panel-card"):
            gr.Markdown("### 📈 EPISODE SCORE", elem_classes="sidebar-title")
            score_display = gr.HTML(value=get_score_html(0.000, 0.00, []))

    # ── BOTTOM ROW: TERMINAL ──
    with gr.Row():
        with gr.Column(elem_classes="panel-card"):
            gr.Markdown("### 💻 LIVE AGENT TERMINAL", elem_classes="sidebar-title")
            output_console = gr.Textbox(lines=16, max_lines=30, elem_id="console-logs")

    # ── Button Wiring ──
    task_selector.change(fn=lambda x: TASK_DESCRIPTIONS[x], inputs=task_selector, outputs=task_desc)
    
    reset_btn.click(
        fn=ui_reset, 
        inputs=[task_selector], 
        outputs=[session_env, session_history, session_reward_history, output_console, score_display, node_display]
    )
    
    step_btn.click(
        fn=ui_step, 
        inputs=[session_env, session_history, session_reward_history, output_console, task_selector], 
        outputs=[session_env, session_history, session_reward_history, output_console, score_display, node_display]
    )
    
    state_btn.click(
        fn=ui_get_state, 
        inputs=[session_env, output_console], 
        outputs=[output_console]
    )

# ── Mount Gradio & Launch ──
app = gr.mount_gradio_app(fastapi_app, gradio_ui, path="/")
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860, log_level="info")