import os
import json
import re
from dotenv import load_dotenv
import gradio as gr
from openai import OpenAI
from environment import ClusterTriageEnv
from models import ClusterAction

# Load environment variables
load_dotenv()

# --- Configuration & Auth ---
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-ai/DeepSeek-R1-Distill-Llama-70B")

# Keep ONLY the token as a secure environment variable, aggressively strip junk characters
API_KEY = os.getenv("HF_TOKEN", "").strip().strip('"').strip("'")

if not API_KEY:
    print("CRITICAL ERROR: HF_TOKEN environment variable is missing!")

print(f"[DEBUG] Booting up UI... Using HF_TOKEN starting with: {API_KEY[:6]} (Length: {len(API_KEY)})")

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

# --- Task Descriptions for the UI ---
TASK_DESCRIPTIONS = {
    "easy": "🟢 EASY: A single rogue job is stuck in an infinite loop.\n\nGoal: Kill the rogue job.",
    "medium": "🟡 MEDIUM: A worker node's disk has hit 99.9% capacity.\n\nGoal: Clear the storage and restart the node safely.",
    "hard": "🟠 HARD: Cascade failure. Nodes 1 & 2 crashed due to replication storms.\n\nGoal: Kill the root job, clear the dead nodes, and reboot them.",
    "very_hard": "🔴 VERY HARD: Multi-vector attack. Log spam and crypto miner active simultaneously.\n\nGoal: Terminate both malware instances, clear the junk, and reboot.",
    "nightmare": "☠️ NIGHTMARE: The Hydra Protocol. All nodes offline. 3 self-replicating jobs active.\n\nGoal: You MUST kill ALL 3 Hydras before attempting to clear any storage, or they will instantly cross-infect. Clear all nodes, then restart all nodes."
}

def extract_action_from_llm(response_text: str) -> ClusterAction:
    """Aggressively hunts for valid JSON in the LLM output."""
    try:
        response_text = response_text.replace("```json", "").replace("```", "").strip()
        match = re.search(r'\{.*?\}', response_text, re.DOTALL)
        if match:
            json_str = match.group(0)
            data = json.loads(json_str)
            if "action_type" in data:
                return ClusterAction(**data)
                
        return ClusterAction(action_type="noop", target_id="none")
    except Exception as e:
        return ClusterAction(action_type="noop", target_id="none")

# --- Core Execution Logic (Streams to the Web UI) ---
def execute_triage(task_name):
    if not API_KEY:
        yield "CRITICAL ERROR: HF_TOKEN is missing! Please configure your API key."
        return

    env = ClusterTriageEnv()
    obs = env.reset(task_name)
    action_history = []
    
    # Initialize the log string that will be pushed to the UI
    log_history = f"===== Booting OpenEnv SRE Agent =====\n"
    log_history += f"--- Starting Threat Level: {task_name.upper()} ---\n\n"
    yield log_history 

    for step in range(1, 26):
        history_text = "\n".join(action_history) if action_history else "None."
        
        # Your bulletproof prompt injected with current state and history
        prompt = f"""You are an automated DevOps system. You cannot speak. You can only output raw JSON commands.

CURRENT CLUSTER STATE:
{obs.model_dump_json(indent=2)}

YOUR PREVIOUS ACTIONS & RESULTS (Do NOT repeat failed actions):
{history_text}

INSTRUCTIONS:
1. Analyze the state.
2. If a node has high disk usage, you MUST 'clear_temp_storage' before restarting it.
3. If a node is offline, you MUST 'restart_node' (but ensure disk is not full first).
4. If there are ANY hanging jobs, you MUST 'kill_job' for EVERY SINGLE ONE of them before attempting to clear any storage. Do not leave any alive.
5. CRITICAL: There are multiple nodes and potentially multiple jobs.

Respond with EXACTLY ONE JSON object and absolutely no other text.
Valid action_type values: "kill_job", "restart_node", "clear_temp_storage", "noop".

EXPECTED FORMAT:
{{"action_type": "kill_job", "target_id": "job_rogue_99"}}
"""

        try:
            # Trigger API Call
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1 
            )
            
            raw_output = response.choices[0].message.content.strip()
            action = extract_action_from_llm(raw_output)
            
            # Execute step in environment
            result = env.step(action)
            obs = env.state() 
            
            # Update the UI logs
            log_history += f"Step {step} Agent Action: {action.action_type} on {action.target_id}\n"
            log_history += f"Reward: {result.score} | Note: {result.message}\n"
            yield log_history # Instantly pushes the new line to the web page
            
            # Memory injection for next step
            action_history.append(f"Step {step}: Action: {action.action_type} on {action.target_id} -> Reward: {result.score} -> System Note: {result.message}")
            if action.action_type == "noop":
                 action_history.append(f"CRITICAL WARNING: Your last output was invalid. You MUST output ONLY valid JSON.")

            if result.is_done:
                log_history += f"\nTask '{task_name.upper()}' completed in {step} steps.\n"
                log_history += "===== SYSTEM NOMINAL ====="
                yield log_history
                break
                
        except Exception as e:
            log_history += f"\n[API ERROR]: {str(e)}\n"
            yield log_history
            break

# --- GRADIO WEB INTERFACE ---

# 1. Define custom CSS for interactive colors and terminal aesthetics
CUSTOM_CSS = """
/* Set the main page background to a modern light gray */
body, html {
    background-color: #f3f4f6 !important;
}

/* Base style for component cards */
.content-card {
    background-color: #ffffff !important;
    padding: 25px !important;
    border-radius: 12px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
    border: 1px solid #e5e7eb;
    margin-bottom: 20px !important;
}

.content-card h1 {
    color: #0f172a !important; /* Deep dark text for the main title */
}
.content-card p {
    color: #475569 !important; /* Visible slate gray for the subtitle */
}

#console-logs textarea {
    background-color: #0d1117 !important; /* Deep GitHub dark */
    color: #00ff41 !important; /* Matrix green text */
    font-family: 'Courier New', Courier, monospace !important;
    font-size: 14px !important;
    border: 2px solid #30363d !important;
    border-radius: 8px !important;
    padding: 15px !important;
    box-shadow: inset 0 2px 10px rgba(0,0,0,0.5);
}

/* Interactive, tech-styled Execute button */
#execute-btn {
    background: linear-gradient(135deg, #2563eb, #1e40af) !important; /* Tech blue gradient */
    color: white !important;
    font-weight: bold !important;
    font-size: 16px !important;
    border: none !important;
    border-radius: 8px !important;
    box-shadow: 0 4px 6px rgba(37, 99, 235, 0.2) !important;
    transition: all 0.2s ease-in-out !important;
}

/* Button Hover Effect */
#execute-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 15px rgba(37, 99, 235, 0.35) !important;
    background: linear-gradient(135deg, #3b82f6, #2563eb) !important;
}

/* Task Briefing Box */
#task-briefing textarea {
    background-color: #f8fafc !important;
    border: 1px solid #cbd5e1 !important;
    color: #334155 !important;
    font-weight: 500 !important;
}
"""

# 2. Add the CUSTOM_CSS to your existing Blocks call and apply containers
# We add gr.themes.Soft() to remove the default orange accents everywhere else
with gr.Blocks(css=CUSTOM_CSS, theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate")) as app:    # --- Title Section (Optional decorative bar) ---
    with gr.Column(elem_classes="content-card"):
        gr.Markdown("# 🚀 OpenEnv: Distributed Data Cluster Triage", elem_classes="text-center")
        gr.Markdown("Test the LLM agent's multi-step recovery protocols in simulated infrastructure failures. API calls are only made when you click Execute.", elem_classes="text-center text-muted")

    # --- Main Content Section in a boxed container ---
    with gr.Column(elem_classes="content-card"):
        with gr.Row():
            with gr.Column(scale=1):
                task_selector = gr.Radio(
                    choices=["easy", "medium", "hard", "very_hard", "nightmare"],
                    value="easy",
                    label="1. Select Threat Level",
                    elem_id="task-selector"
                )
                task_desc = gr.Textbox(
                    label="Task Briefing",
                    value=TASK_DESCRIPTIONS["easy"],
                    interactive=False,
                    lines=5,
                    elem_id="task-briefing"
                )
                execute_btn = gr.Button("▶ Execute Protocol", variant="primary", elem_id="execute-btn")

            with gr.Column(scale=2):
                output_console = gr.Textbox(
                    label="Agent Terminal Logs",
                    lines=22,
                    max_lines=30,
                    elem_id="console-logs"
                )

    # Wire up the UI interactions (This part is untouched)
    task_selector.change(fn=lambda x: TASK_DESCRIPTIONS[x], inputs=task_selector, outputs=task_desc)
    execute_btn.click(fn=execute_triage, inputs=task_selector, outputs=output_console)

if __name__ == "__main__":
    # We remove the hard-coded plain Monochrome theme and rely on the new CSS
    app.launch(server_name="0.0.0.0", server_port=7860)