"""
app.py — OpenEnv Cluster Triage Environment
Serves:
  • FastAPI HTTP API (/reset, /step, /agent/step, /state, /health, /agents, /tasks)
  • HTML Frontend (/)
All on port 7860.
"""
import os
import json
import re
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from openai import OpenAI
from pydantic import BaseModel
from typing import Optional

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

# ── Shared Environment Instance ───────────────────────────────
api_env = ClusterTriageEnv()
action_history = []  # Maintain history for LLM prompting

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

@fastapi_app.get("/health")
def health():
    """Health check endpoint — returns 200 OK."""
    return {"status": "ok", "env": "cluster-triage", "version": "1.0.0"}

@fastapi_app.get("/agents")
def get_agents():
    return JSONResponse(content=[
        {
            "id": "cluster_triage",
            "name": "Cluster Triage",
            "description": "Interactive RL environment where an AI acts as an SRE.",
            "icon": "🤖",
            "tasks": [
                {"id": "easy", "name": "Easy", "color": "#10b981", "label": "easy", "max_steps": 5},
                {"id": "medium", "name": "Medium", "color": "#eab308", "label": "medium", "max_steps": 10},
                {"id": "hard", "name": "Hard", "color": "#ef4444", "label": "hard", "max_steps": 15},
                {"id": "very_hard", "name": "Very Hard", "color": "#ef4444", "label": "very_hard", "max_steps": 20},
                {"id": "nightmare", "name": "Nightmare", "color": "#7c3aed", "label": "nightmare", "max_steps": 25},
            ]
        }
    ])

@fastapi_app.get("/tasks")
def list_tasks(agent_id: Optional[str] = None):
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

@fastapi_app.post("/reset")
def reset(request: ResetRequest = None):
    """
    Reset the environment for a given task.
    """
    global action_history
    task = (request.task if request else "easy")
    if task not in ["easy", "medium", "hard", "very_hard", "nightmare"]:
        raise HTTPException(status_code=400, detail=f"Unknown task '{task}'. Valid: easy, medium, hard, very_hard, nightmare")
    obs = api_env.reset(task=task)
    action_history = []
    return JSONResponse(content=obs.model_dump())

@fastapi_app.post("/step")
def step(action: ClusterAction):
    """
    Execute one action in the environment (used by validate.sh).
    """
    result = api_env.step(action)
    return JSONResponse(content=result.model_dump())

class AgentStepRequest(BaseModel):
    agent_id: Optional[str] = "cluster_triage"
    task: str = "easy"

@fastapi_app.post("/agent/step")
def agent_step(req: AgentStepRequest):
    """
    Uses LLM to perform one step. Replaces the old Gradio button logic.
    """
    global action_history
    
    if not client:
        raise HTTPException(status_code=500, detail="HF_TOKEN missing. LLM calls will fail.")
        
    obs = api_env.state()
    if obs.health_score >= 1.0 or (api_env.step_count >= api_env.max_steps):
        # Already done, just return current state
        return JSONResponse(content={
            "step": api_env.step_count,
            "reward": 0.0,
            "done": True,
            "message": "Incident resolved or timed out. Please RESET.",
            "action": {"action_type": "noop", "target_id": "none"},
            "observation": obs.model_dump()
        })
        
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
        result = api_env.step(action)
        new_obs = result.observation
        msg = result.info.get("message", "")
        
        step_num = api_env.step_count
        action_history.append(f"Step {step_num}: {action.action_type} on {action.target_id} → reward={result.reward:.2f} | {msg}")
        
        return JSONResponse(content={
            "step": step_num,
            "reward": result.reward,
            "done": result.done,
            "message": msg,
            "action": action.model_dump(),
            "observation": new_obs.model_dump()
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM API Error: {str(e)}")

@fastapi_app.get("/state")
def state():
    """Return the current cluster observation without advancing the episode."""
    return JSONResponse(content=api_env.state().model_dump())

# Serve static files including index.html at root
fastapi_app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run(fastapi_app, host="0.0.0.0", port=7860, log_level="info")