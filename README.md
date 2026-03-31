---
title: OpenEnv Cluster Triage
emoji: 🚀
colorFrom: gray
colorTo: red
sdk: docker
pinned: false
license: mit
---

# OpenEnv: Distributed Data Cluster Triage (Enterprise 4-Node)

## Environment Description & Motivation
Managing a distributed data processing cluster (e.g., Hadoop/HDFS) is a complex, high-stakes DevOps challenge. When compute resources are finite, runaway jobs and hardware constraints can cause cascading failures that require immediate, multi-step human intervention.

This OpenEnv simulates an enterprise-level 4-node cluster under stress. It is a genuine infrastructure triage scenario. An AI agent acting as an automated Site Reliability Engineer (SRE) must observe the cluster's health dashboard, identify root causes of failures, and issue precise terminal commands to prevent total system collapse.

## Observation & Action Spaces
The environment strictly adheres to the OpenEnv Pydantic specifications.

### Observation Space
The agent receives a `ClusterObservation` detailing the exact state of the infrastructure:
* **`health_score`**: (Float 0.0 - 1.0) The continuous metric of cluster stability.
* **`nodes`**: A list of 4 `NodeStatus` objects (worker nodes), detailing CPU, RAM, disk usage, and current status (`healthy`, `high_memory`, `disk_full`, `offline`).
* **`active_jobs`**: A list of `JobStatus` objects detailing allocated memory and execution states.
* **`recent_alerts`**: System logs and critical alerts providing context to the failures.

### Action Space
The agent interacts with the environment by outputting a strictly typed `ClusterAction` JSON object:
* **`action_type`**: The command to execute (`kill_job`, `restart_node`, `clear_temp_storage`, `noop`).
* **`target_id`**: The specific `job_id` or `node_id` to apply the action to.

## Task Descriptions & Difficulty

The environment features a 5-tier difficulty scale. It provides a meaningful, continuous reward signal, penalizing destructive actions and rewarding partial progress.

* **EASY (The Stuck Job):** A rogue MapReduce job is stuck in an infinite loop. 
  * *Expected Action:* `kill_job` -> `job_rogue_99`. 
* **MEDIUM (The Full Disk):** A worker node's disk has hit 99.9% capacity.
  * *Expected Sequence:* `clear_temp_storage` -> `restart_node`. 
* **HARD (The Cascade Failure):** A rogue job caused a replication storm, crashing Nodes 1 and 2. Failover traffic is overloading Nodes 3 and 4.
  * *Expected Sequence:* Kill the root job, clear storage on dead nodes, then safely reboot them. (5 steps)
* **VERY HARD (Multi-Vector Attack):** A disk-filling log spammer and a RAM-hogging crypto miner hit the cluster simultaneously.
  * *Expected Sequence:* Isolate and kill both malware jobs before clearing storage and rebooting. (6 steps)
* **NIGHTMARE (The Hydra Protocol):** Total cluster collapse. Three self-replicating malware jobs are writing garbage data to all 4 nodes. 
  * *Expected Sequence:* The agent must kill ALL THREE Hydra jobs before attempting to clear any storage, otherwise, the surviving malware instantly cross-infects and refills the disks. (11 steps)

## Setup & Usage Instructions

### Running via Docker (Recommended)
This environment is containerized for seamless execution and automated validation.
1. Build the image:
   `docker build -t cluster-triage-env .`
2. Run the baseline inference script:
   `docker run -e API_BASE_URL="https://router.huggingface.co/v1" -e MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Llama-70B" -e HF_TOKEN="your_token" cluster-triage-env`

### Running Locally
1. Install dependencies: `pip install -r requirements.txt`
2. Configure your `.env` file with your `HF_TOKEN`, `MODEL_NAME`, and `API_BASE_URL`.
3. Execute the baseline test: `python inference.py`

### How to Use the Web Dashboard (Hugging Face)

The OpenEnv Web UI is designed to be fully interactive and provides a real-time window into the SRE Agent's decision-making process. Here is how to evaluate the simulation:

1. **Access the Dashboard:** Navigate to the public Hugging Face Space URL.
2. **Select a Threat Level:** Use the radio buttons on the left control panel to choose a scenario, ranging from `Easy` (a single stuck job) to `Nightmare` (a multi-vector Hydra attack).
   * *Note:* Selecting a level instantly updates the **Task Briefing** box to explain the specific failure state and the required win conditions.
3. **Deploy the Agent:** Click the blue **▶ Execute Protocol** button. 
   * *API Notice:* To conserve compute, the environment is strictly event-driven. It only makes LLM API calls when this button is clicked. 
4. **Observe the Triage:** Watch the dark-mode **Agent Terminal Logs** on the right. The UI will stream the AI's step-by-step actions, rewards, and system feedback in real-time as it attempts to stabilize the cluster and achieve a `SYSTEM NOMINAL` status.

## Baseline Scores
The baseline inference script utilizes `deepseek-ai/DeepSeek-R1-Distill-Llama-70B` via the Hugging Face Serverless API. It successfully navigates the complex logic gates of all 5 tiers without hallucination.

* **Easy Task:** 1.0 (Completed in 1 step)
* **Medium Task:** 1.0 (Completed in 2 steps)
* **Hard Task:** 1.0 (Completed in 5 steps)
* **Very Hard Task:** 1.0 (Completed in 6 steps)
* **Nightmare Task:** 1.0 (Completed in 11 steps)