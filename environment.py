import copy
from models import ClusterObservation, ClusterAction, StepReward, NodeStatus, JobStatus

class ClusterTriageEnv:
    def __init__(self):
        self.current_task = None
        self.state_data = None
        self.step_count = 0
        self.max_steps = 25 # Increased for Nightmare mode
        
        # Trackers
        self.rogue_job_killed = False
        self.medium_storage_cleared = False
        self.cleared_nodes = set()
        self.vh_log_killed = False
        self.vh_crypto_killed = False
        
        # Trackers for NIGHTMARE task
        self.nm_hydras_killed = set()
        self.nm_nodes_cleared = set()

    def reset(self, task: str) -> ClusterObservation:
        self.current_task = task
        self.step_count = 0
        self.rogue_job_killed = False
        self.medium_storage_cleared = False
        self.cleared_nodes = set()
        self.vh_log_killed = False
        self.vh_crypto_killed = False
        self.nm_hydras_killed = set()
        self.nm_nodes_cleared = set()
        
        base_nodes = [
            NodeStatus(node_id="worker_01", status="healthy", cpu_usage=45.0, ram_usage=60.0, disk_usage=30.0),
            NodeStatus(node_id="worker_02", status="healthy", cpu_usage=50.0, ram_usage=55.0, disk_usage=40.0),
            NodeStatus(node_id="worker_03", status="healthy", cpu_usage=42.0, ram_usage=48.0, disk_usage=25.0),
            NodeStatus(node_id="worker_04", status="healthy", cpu_usage=48.0, ram_usage=52.0, disk_usage=35.0)
        ]
        
        if task == "easy":
            base_jobs = [JobStatus(job_id="job_normal_01", status="running", ram_allocated=10.0), JobStatus(job_id="job_rogue_99", status="hanging", ram_allocated=99.0)]
            self.state_data = ClusterObservation(health_score=0.4, nodes=base_nodes, active_jobs=base_jobs, recent_alerts=["CRITICAL: job_rogue_99 is hanging."])
            
        elif task == "medium":
            base_nodes[2].status = "disk_full"; base_nodes[2].disk_usage = 99.9
            self.state_data = ClusterObservation(health_score=0.3, nodes=base_nodes, active_jobs=[], recent_alerts=["WARNING: worker_03 disk full."])
            
        elif task == "hard":
            base_nodes[0].status = "offline"; base_nodes[0].disk_usage = 100.0
            base_nodes[1].status = "offline"; base_nodes[1].disk_usage = 100.0
            base_nodes[2].status = "high_memory"; base_nodes[2].ram_usage = 95.0
            base_nodes[3].status = "high_memory"; base_nodes[3].ram_usage = 95.0
            base_jobs = [JobStatus(job_id="job_rogue_99", status="hanging", ram_allocated=0.0)]
            self.state_data = ClusterObservation(health_score=0.1, nodes=base_nodes, active_jobs=base_jobs, recent_alerts=["ERROR: job_rogue_99 causing replication storm. Nodes 1 & 2 offline."])
            
        elif task == "very_hard":
            base_nodes[0].status = "offline"; base_nodes[0].disk_usage = 100.0
            base_nodes[1].status = "offline"; base_nodes[1].disk_usage = 100.0
            base_nodes[2].status = "high_memory"; base_nodes[2].ram_usage = 99.0
            base_nodes[3].status = "high_memory"; base_nodes[3].ram_usage = 99.0
            base_jobs = [JobStatus(job_id="job_log_spam", status="hanging", ram_allocated=0.0), JobStatus(job_id="job_crypto_miner", status="hanging", ram_allocated=99.0)]
            self.state_data = ClusterObservation(health_score=0.05, nodes=base_nodes, active_jobs=base_jobs, recent_alerts=["CRITICAL: MULTIPLE FAILURES. Log spam on Nodes 1/2. Crypto miner on Nodes 3/4."])
            
        elif task == "nightmare":
            # Total Cluster Death
            for node in base_nodes:
                node.status = "offline"
                node.cpu_usage = 0.0
                node.ram_usage = 0.0
                node.disk_usage = 100.0
            
            base_jobs = [
                JobStatus(job_id="job_hydra_1", status="hanging", ram_allocated=0.0),
                JobStatus(job_id="job_hydra_2", status="hanging", ram_allocated=0.0),
                JobStatus(job_id="job_hydra_3", status="hanging", ram_allocated=0.0)
            ]
            self.state_data = ClusterObservation(
                health_score=0.0, nodes=base_nodes, active_jobs=base_jobs, 
                recent_alerts=[
                    "FATAL: TOTAL CLUSTER COLLAPSE.",
                    "WARNING: HYDRA PROTOCOL DETECTED. MULTIPLE MALWARE INSTANCES WRITING TO ALL DISKS.",
                    "HINT: If any hydra job survives, it will instantly refill cleared storage."
                ]
            )
        else:
            self.state_data = ClusterObservation(health_score=1.0, nodes=base_nodes, active_jobs=[], recent_alerts=[])
            
        return self.state()

    def state(self) -> ClusterObservation:
        return copy.deepcopy(self.state_data)

    def step(self, action: ClusterAction) -> StepReward:
        self.step_count += 1
        reward_score = 0.0
        done = False
        msg = "Action registered."

        if self.current_task == "easy":
            if action.action_type == "kill_job" and action.target_id == "job_rogue_99":
                reward_score = 1.0; done = True; msg = "Success: Rogue job killed."; self.state_data.health_score = 1.0

        elif self.current_task == "medium":
            if action.action_type == "clear_temp_storage" and action.target_id == "worker_03":
                self.medium_storage_cleared = True; self.state_data.nodes[2].disk_usage = 20.0; self.state_data.nodes[2].status = "offline"
                reward_score = 0.5; msg = "Good: Storage cleared. Node needs restart."
            elif action.action_type == "restart_node" and action.target_id == "worker_03":
                if self.medium_storage_cleared:
                    reward_score = 0.5; done = True; self.state_data.health_score = 1.0; self.state_data.nodes[2].status = "healthy"
                    msg = "Success: Node restarted."

        elif self.current_task == "hard":
            if action.action_type == "kill_job" and action.target_id == "job_rogue_99":
                self.rogue_job_killed = True; self.state_data.active_jobs = [j for j in self.state_data.active_jobs if j.job_id != "job_rogue_99"]
                reward_score = 0.2; msg = "Good. Rogue job terminated."
            elif action.action_type == "clear_temp_storage" and action.target_id in ["worker_01", "worker_02"]:
                if not self.rogue_job_killed:
                    reward_score = -0.1; msg = f"Penalty: job_rogue_99 refilled {action.target_id}."
                else:
                    self.cleared_nodes.add(action.target_id)
                    idx = 0 if action.target_id == "worker_01" else 1
                    self.state_data.nodes[idx].disk_usage = 20.0; reward_score = 0.2; msg = f"Good. Storage cleared on {action.target_id}."
            elif action.action_type == "restart_node" and action.target_id in ["worker_01", "worker_02"]:
                if action.target_id not in self.cleared_nodes:
                    reward_score = -0.2; msg = f"CRITICAL FAILURE: {action.target_id} crashed."
                else:
                    idx = 0 if action.target_id == "worker_01" else 1
                    self.state_data.nodes[idx].status = "healthy"; reward_score = 0.2; msg = f"SUCCESS: {action.target_id} rebooted cleanly."
                    if "worker_01" in self.cleared_nodes and "worker_02" in self.cleared_nodes and self.state_data.nodes[0].status == "healthy" and self.state_data.nodes[1].status == "healthy":
                        self.state_data.nodes[2].status = "healthy"; self.state_data.nodes[3].status = "healthy"
                        self.state_data.health_score = 1.0; done = True; msg += " Cluster stabilized."

        elif self.current_task == "very_hard":
            if action.action_type == "kill_job":
                if action.target_id == "job_log_spam":
                    self.vh_log_killed = True; self.state_data.active_jobs = [j for j in self.state_data.active_jobs if j.job_id != "job_log_spam"]; reward_score = 0.15; msg = "Good. Disk-filling root cause terminated."
                elif action.target_id == "job_crypto_miner":
                    self.vh_crypto_killed = True; self.state_data.active_jobs = [j for j in self.state_data.active_jobs if j.job_id != "job_crypto_miner"]
                    self.state_data.nodes[2].status = "healthy"; self.state_data.nodes[2].ram_usage = 48.0; self.state_data.nodes[3].status = "healthy"; self.state_data.nodes[3].ram_usage = 52.0
                    reward_score = 0.15; msg = "Good. Nodes 3 and 4 relieved."
            elif action.action_type == "clear_temp_storage" and action.target_id in ["worker_01", "worker_02"]:
                if not self.vh_log_killed: reward_score = -0.15; msg = f"Penalty: You must kill job_log_spam first."
                else:
                    self.cleared_nodes.add(action.target_id); idx = 0 if action.target_id == "worker_01" else 1; self.state_data.nodes[idx].disk_usage = 20.0
                    reward_score = 0.15; msg = f"Good. Junk cleared from {action.target_id}."
            elif action.action_type == "restart_node" and action.target_id in ["worker_01", "worker_02"]:
                if action.target_id not in self.cleared_nodes: reward_score = -0.2; msg = f"CRITICAL: {action.target_id} OS crashed."
                else:
                    idx = 0 if action.target_id == "worker_01" else 1; self.state_data.nodes[idx].status = "healthy"; reward_score = 0.2; msg = f"SUCCESS: {action.target_id} rebooted."
                    if self.vh_log_killed and self.vh_crypto_killed and "worker_01" in self.cleared_nodes and "worker_02" in self.cleared_nodes and self.state_data.nodes[0].status == "healthy" and self.state_data.nodes[1].status == "healthy":
                        self.state_data.health_score = 1.0; done = True; msg += " MULTI-VECTOR ATTACK THWARTED."

        # --- NIGHTMARE TASK GRADER ---
        elif self.current_task == "nightmare":
            if action.action_type == "kill_job" and action.target_id.startswith("job_hydra_"):
                self.nm_hydras_killed.add(action.target_id)
                self.state_data.active_jobs = [j for j in self.state_data.active_jobs if j.job_id != action.target_id]
                reward_score = 0.1 
                if len(self.nm_hydras_killed) == 3:
                    msg = f"All Hydra heads severed. Disks are stable and ready for clearing."
                else:
                    msg = f"{action.target_id} terminated. {3 - len(self.nm_hydras_killed)} Hydras remain."

            elif action.action_type == "clear_temp_storage":
                if len(self.nm_hydras_killed) < 3:
                    reward_score = -0.15
                    msg = f"PENALTY: A surviving Hydra instantly refilled {action.target_id} to 100%."
                else:
                    self.nm_nodes_cleared.add(action.target_id)
                    idx = int(action.target_id.split("_")[1]) - 1
                    self.state_data.nodes[idx].disk_usage = 20.0
                    reward_score = 0.1
                    msg = f"Storage cleared on {action.target_id}."

            elif action.action_type == "restart_node":
                if action.target_id not in self.nm_nodes_cleared:
                    reward_score = -0.2
                    msg = f"CRITICAL: {action.target_id} crashed. Disk still full."
                else:
                    idx = int(action.target_id.split("_")[1]) - 1
                    self.state_data.nodes[idx].status = "healthy"
                    reward_score = 0.075 # Remaining score to reach 1.0
                    msg = f"{action.target_id} rebooted cleanly."
                    
                    # Win Condition: All 4 nodes healthy
                    if all(node.status == "healthy" for node in self.state_data.nodes):
                        self.state_data.health_score = 1.0
                        done = True
                        msg += " NIGHTMARE DEFEATED. SYSTEM NOMINAL."

        if not done and self.step_count >= self.max_steps:
            done = True; msg = f"Max steps ({self.max_steps}) reached. Episode failed."

        return StepReward(score=reward_score, is_done=done, message=msg)