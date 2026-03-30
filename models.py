from pydantic import BaseModel, Field
from typing import List, Optional, Literal

# --- OBSERVATION SPACE ---
class NodeStatus(BaseModel):
    node_id: str
    status: Literal["healthy", "high_memory", "disk_full", "offline"]
    cpu_usage: float
    ram_usage: float
    disk_usage: float

class JobStatus(BaseModel):
    job_id: str
    status: Literal["running", "hanging", "queued", "failed"]
    ram_allocated: float

class ClusterObservation(BaseModel):
    health_score: float = Field(..., description="Overall cluster health from 0.0 to 1.0")
    nodes: List[NodeStatus]
    active_jobs: List[JobStatus]
    recent_alerts: List[str]

# --- ACTION SPACE ---
class ClusterAction(BaseModel):
    action_type: Literal["kill_job", "restart_node", "clear_temp_storage", "noop"]
    target_id: Optional[str] = Field(None, description="The job_id or node_id to target")

# --- REWARD SPACE ---
class StepReward(BaseModel):
    score: float = Field(..., description="Reward between 0.0 and 1.0")
    is_done: bool
    message: str