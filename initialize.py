import logging
import os

from hydra.utils import instantiate
from hydra.utils import get_original_cwd

from application import Application
from GithubCode.LLMEngineOnWafer.components.chiplet import Chiplet
from GithubCode.LLMEngineOnWafer.components.die import Die
from GithubCode.LLMEngineOnWafer.components.tile import Tile
from hardware_repo import HardwareRepo
from model_repo import ModelRepo
from orchestrator_repo import OrchestratorRepo
from start_state import load_start_state
from trace import Trace

def init_trace(cfg):
    trace_path = os.path.join(get_original_cwd(), cfg.trace_path)
    trace = Trace.from_csv(trace_path)
    return trace
