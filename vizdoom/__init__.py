"""
VizDoom HeRoN Package
Package per l'implementazione di HeRoN su VizDoom
"""

from .vizdoom_env import VizDoomEnv, create_vizdoom_env
from .vizdoom_agent import DQNCnnAgent, ReplayBuffer, build_dqn_cnn, build_dueling_dqn_cnn
from .vizdoom_action_score import (
    calculate_action_scores,
    evaluate_action_plan,
    generate_corrective_feedback,
    get_best_action
)

__all__ = [
    'VizDoomEnv',
    'create_vizdoom_env',
    'DQNCnnAgent',
    'ReplayBuffer',
    'build_dqn_cnn',
    'build_dueling_dqn_cnn',
    'calculate_action_scores',
    'evaluate_action_plan',
    'generate_corrective_feedback',
    'get_best_action'
]
