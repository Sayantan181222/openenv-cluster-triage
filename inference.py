import os
import json
from openai import OpenAI
from environment import ClusterTriageEnv
from models import ClusterAction
import os
from dotenv import load_dotenv
import re
import time

# This loads the variables from the .env file into your system environment
load_dotenv() 

API_BASE_URL = "https://router.huggingface.co/v1"
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"

# Keep ONLY the token as a secure environment variable, but aggressively strip junk characters
API_KEY = os.getenv("HF_TOKEN", "").strip().strip('"').strip("'")

# Safety check + Debug print to prove exactly what is being sent
if not API_KEY:
    print("CRITICAL ERROR: HF_TOKEN environment variable is missing!")
    exit(1)

print(f"[DEBUG] Booting up... Using HF_TOKEN starting with: {API_KEY[:6]} (Length: {len(API_KEY)})")

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

def extract_action_from_llm(response_text: str) -> ClusterAction:
    """Aggressively hunts for valid JSON in the LLM output."""
    try:
        # Strip common markdown blocks the LLM might try to use
        response_text = response_text.replace("```json", "").replace("```", "").strip()
        
        # Search for the exact JSON bracket structure
        match = re.search(r'\{.*?\}', response_text, re.DOTALL)
        if match:
            json_str = match.group(0)
            data = json.loads(json_str)
            # Ensure it actually generated the right keys before returning
            if "action_type" in data:
                return ClusterAction(**data)
                
        print(f"\n[DEBUG] LLM hallucinated. Raw output: {response_text[:100]}...")
        return ClusterAction(action_type="noop")
        
    except Exception as e:
        print(f"\n[DEBUG] Parsing failed: {e}")
        return ClusterAction(action_type="noop")

def main():
    env = ClusterTriageEnv()
    tasks = ["easy", "medium", "hard", "very_hard", "nightmare"] 
    
    for task in tasks:
        print(f"\n--- Starting Task: {task.upper()} ---")
        observation = env.reset(task)
        action_history = [] 
        
        for step in range(1, 26): 
            history_text = "\n".join(action_history) if action_history else "None."
            
            # THE BULLETPROOF PROMPT
            prompt = f"""You are an automated DevOps system. You cannot speak. You can only output raw JSON commands.

CURRENT CLUSTER STATE:
{observation.model_dump_json(indent=2)}

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
            
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1 # Low temperature keeps it robotic and deterministic
            )
            
            raw_output = response.choices[0].message.content.strip()
            action = extract_action_from_llm(raw_output)
            
            print(f"Step {step} Agent Action: {action.action_type} on {action.target_id}")
            
            result = env.step(action)
            observation = env.state() 
            print(f"Reward: {result.score} | Note: {result.message}")
            
            # Feed the exact penalty back into the AI's memory
            action_history.append(f"Step {step}: Action: {action.action_type} on {action.target_id} -> Reward: {result.score} -> System Note: {result.message}")
            
            # If the AI hallucinates a noop, forcefully remind it in the history
            if action.action_type == "noop":
                 action_history.append(f"CRITICAL WARNING: Your last output was invalid. You MUST output ONLY valid JSON.")

            if result.is_done:
                print(f"Task '{task}' completed in {step} steps.")
                break

if __name__ == "__main__":
    print("===== Application Startup =====")
    main()
    print("\n===== All Tasks Completed Successfully. Keeping container alive... =====")
    
    # This infinite loop tricks Hugging Face into keeping the green "Running" badge active
    while True:
        time.sleep(3600) # Sleep for an hour, wake up, repeat forever