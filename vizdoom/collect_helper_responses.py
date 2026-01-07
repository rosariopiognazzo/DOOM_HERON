"""
Script per raccogliere le risposte dell'Helper (LLM zero-shot) su VizDoom.
FASE 1: CNN + Helper Zero-Shot per generare dataset di fine-tuning per il Reviewer.

Questo script:
1. Esegue episodi di VizDoom con l'agente DQN + suggerimenti Helper
2. Logga tutte le risposte dell'Helper con lo stato di gioco
3. Salva i dati in formato CSV per il fine-tuning del Reviewer
4. Genera statistiche e grafici per il report
"""

import numpy as np
import pandas as pd
import re
import os
import time
import requests
import json
from datetime import datetime
from collections import deque
# import lmstudio as lms  <-- Rimosso

from vizdoom_env import VizDoomEnv, create_vizdoom_env
from vizdoom_agent import DQNCnnAgent
from training_stats import TrainingStats, TrainingVisualizer
from vizdoom_action_score import get_action_score


# ================== CONFIGURAZIONE ==================

# Host LM Studio per l'Helper
SERVER_API_HOST = "http://localhost:1234/v1/chat/completions"

# Modello Helper (modificare con il modello scelto)
# Nota: usa il nome esatto che visualizzi in LM Studio
HELPER_MODEL_NAME = "qwen/qwen2.5-vl-7b" 

# Parametri di raccolta dati
EPISODES_TO_COLLECT = 500  # Numero di episodi per scenario
HELPER_CALL_FREQUENCY = 10  # Chiama Helper ogni N step
HELPER_CALLS_PER_EPISODE = 5  # Numero massimo di chiamate Helper per episodio
PLAN_SIZE = 5  # Numero di azioni da chiedere all'Helper

# Scenario da testare
SCENARIO = 'deadly_corridor'  # 'basic', 'deadly_corridor', 'defend_the_center'

# Path per salvare i log
OUTPUT_DIR = "helper_logs"
OUTPUT_FILE = f"helper_responses_{SCENARIO}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"


# ================== FUNZIONI DI UTILITÀ ==================

def setup_lmstudio():
    """Configura la connessione a LM Studio (Verifica HTTP)."""
    try:
        # Check semplice (endpoint models standard OpenAI)
        base_url = SERVER_API_HOST.replace("/chat/completions", "/models")
        response = requests.get(base_url, timeout=5)
        if response.status_code == 200:
            print(f"Connected to LM Studio at {SERVER_API_HOST}")
            return True
        else:
            print(f"Warning: LM Studio returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"Error connecting to LM Studio: {e}")
        print("Please ensure LM Studio server is running on port 1234.")
        raise e


def create_helper_prompt(game_state_description, plan_size=5):
    """
    Crea il prompt per l'Helper LLM.
    
    Args:
        game_state_description: Descrizione testuale dello stato di gioco
        plan_size: Numero di azioni da richiedere
        
    Returns:
        Prompt formattato per l'Helper
    """
    prompt = (
        f"You are an AI assistant helping a player in a first-person shooter game (Doom). "
        f"Your task is to suggest a sequence of {plan_size} actions to help the player achieve their objective.\n\n"
        f"Current game state:\n{game_state_description}\n\n"
        f"Based on this state, suggest the next {plan_size} actions the player should take.\n"
        f"Format your response as a JSON array of action names, like: [\"ACTION1\", \"ACTION2\", \"ACTION3\", \"ACTION4\", \"ACTION5\"]\n"
        f"Only use actions from the available actions list.\n"
        f"Briefly explain your reasoning after the action list (max 50 words).\n"
        f"/no_think"
    )
    return prompt


def parse_action_plan(llm_response, valid_actions):
    """
    Estrae la lista di azioni dalla risposta dell'LLM.
    
    Args:
        llm_response: Risposta grezza dell'Helper
        valid_actions: Lista di nomi di azioni valide
        
    Returns:
        Lista di nomi di azioni estratte, o lista vuota se parsing fallisce
    """
    # Rimuovi tag di pensiero se presenti
    response = re.sub(r"<think>.*?</think>", "", llm_response, flags=re.DOTALL).strip()
    
    # Prova a estrarre JSON array
    json_match = re.search(r'\[([^\]]+)\]', response)
    if json_match:
        try:
            # Estrai le stringhe tra virgolette
            actions_str = json_match.group(1)
            actions = re.findall(r'"([^"]+)"', actions_str)
            
            # Normalizza e valida le azioni
            normalized_actions = []
            for action in actions:
                action_upper = action.upper().strip()
                # Mapping di alias
                aliases = {
                    'SHOOT': 'ATTACK', 'FIRE': 'ATTACK',
                    'LEFT': 'MOVE_LEFT', 'RIGHT': 'MOVE_RIGHT',
                    'FORWARD': 'MOVE_FORWARD', 'BACKWARD': 'MOVE_BACKWARD',
                    'BACK': 'MOVE_BACKWARD', 'STRAFE_LEFT': 'MOVE_LEFT',
                    'STRAFE_RIGHT': 'MOVE_RIGHT', 'ROTATE_LEFT': 'TURN_LEFT',
                    'ROTATE_RIGHT': 'TURN_RIGHT'
                }
                if action_upper in aliases:
                    action_upper = aliases[action_upper]
                    
                if action_upper in valid_actions:
                    normalized_actions.append(action_upper)
                    
            return normalized_actions
            
        except Exception as e:
            print(f"Error parsing JSON: {e}")
            
    # Fallback: cerca azioni singole tra parentesi quadre
    single_actions = re.findall(r'\[([^\]]+)\]', response)
    normalized = []
    for action in single_actions:
        action_upper = action.upper().strip()
        if action_upper in valid_actions:
            normalized.append(action_upper)
            
    return normalized


def call_helper(client, game_state_description, plan_size=5):
    """
    Chiama l'Helper LLM per ottenere un piano di azioni usando requests.
    
    Args:
        client: (Ignorato, mantenuto per compatibilità firma)
        game_state_description: Descrizione dello stato
        plan_size: Numero di azioni da richiedere
        
    Returns:
        Tuple (risposta_grezza, tempo_risposta)
    """
    prompt = create_helper_prompt(game_state_description, plan_size)
    
    start_time = time.time()
    response_text = ""
    
    payload = {
        "model": HELPER_MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a helpful VizDoom assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 500,
        "stream": False
    }

    try:
        response = requests.post(
            SERVER_API_HOST, 
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            response_text = data['choices'][0]['message']['content']
        else:
            print(f"Error from LM Studio: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"Error calling Helper: {e}")
        
    elapsed = time.time() - start_time
    return response_text, elapsed


# ================== DATA COLLECTION ==================

class HelperDataCollector:
    """
    Classe per raccogliere dati dall'interazione Helper-Environment.
    """
    
    def __init__(self, scenario, output_dir=OUTPUT_DIR):
        self.scenario = scenario
        self.output_dir = output_dir
        self.data = []
        
        # Crea directory di output
        os.makedirs(output_dir, exist_ok=True)
        
    def log_interaction(self, episode, step, game_state, helper_response, 
                       parsed_actions, actions_executed, rewards_obtained,
                       was_valid, response_time, episode_outcome=None):
        """
        Logga un'interazione con l'Helper.
        
        Args:
            episode: Numero dell'episodio
            step: Step nell'episodio
            game_state: Descrizione testuale dello stato
            helper_response: Risposta grezza dell'Helper
            parsed_actions: Azioni estratte dalla risposta
            actions_executed: Azioni effettivamente eseguite
            rewards_obtained: Reward ottenuti per ogni azione
            was_valid: Se le azioni erano valide
            response_time: Tempo di risposta dell'Helper
            episode_outcome: Risultato finale dell'episodio (opzionale)
        """
        self.data.append({
            'scenario': self.scenario,
            'episode': episode,
            'step': step,
            'game_state': game_state,
            'helper_response': helper_response,
            'parsed_actions': str(parsed_actions),
            'actions_executed': str(actions_executed),
            'rewards_obtained': str(rewards_obtained),
            'was_valid': was_valid,
            'num_valid_actions': sum(1 for a in parsed_actions if a) if parsed_actions else 0,
            'num_total_actions': len(parsed_actions) if parsed_actions else 0,
            'response_time': response_time,
            'episode_outcome': episode_outcome,
            'timestamp': datetime.now().isoformat()
        })
        
    def save(self, filename=None):
        """Salva i dati raccolti in un file CSV."""
        if filename is None:
            filename = os.path.join(self.output_dir, OUTPUT_FILE)
        else:
            filename = os.path.join(self.output_dir, filename)
            
        df = pd.DataFrame(self.data)
        df.to_csv(filename, index=False)
        print(f"Data saved to {filename}")
        return filename
    
    def get_stats(self):
        """Restituisce statistiche sui dati raccolti."""
        if not self.data:
            return {}
            
        df = pd.DataFrame(self.data)
        stats = {
            'total_interactions': len(df),
            'total_episodes': df['episode'].nunique(),
            'valid_response_rate': df['was_valid'].mean(),
            'avg_response_time': df['response_time'].mean(),
            'avg_valid_actions_per_response': df['num_valid_actions'].mean(),
        }
        return stats


# ================== TRAINING LOOP ==================

def collect_helper_data(episodes=EPISODES_TO_COLLECT, scenario=SCENARIO, 
                        visible=False, config_path=None, generate_plots=False, plan_size=PLAN_SIZE,
                        helper_frequency=HELPER_CALL_FREQUENCY):
    """
    Loop principale per raccogliere dati dall'Helper.
    
    Args:
        episodes: Numero di episodi da eseguire
        scenario: Scenario VizDoom da usare
        visible: Se mostrare la finestra di gioco
        config_path: Path ai file di configurazione VizDoom
        generate_plots: Se generare grafici delle statistiche
        plan_size: Numero di azioni richieste all'Helper
        
    Returns:
        Path del file CSV con i dati raccolti
    """
    print(f"\n{'='*60}")
    print(f"FASE 1: Raccolta dati Helper per scenario '{scenario}'")
    print(f"{'='*60}\n")
    
    # Setup
    setup_lmstudio()
    collector = HelperDataCollector(scenario)
    
    # Inizializza modulo statistiche
    stats_output_dir = os.path.join(OUTPUT_DIR, f"stats_{scenario}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    training_stats = TrainingStats(scenario=scenario, output_dir=stats_output_dir)
    
    # Crea environment
    print("Initializing VizDoom environment...")
    try:
        env = create_vizdoom_env(scenario=scenario, visible=visible, config_path=config_path)
    except Exception as e:
        print(f"Error creating environment: {e}")
        print("Make sure VizDoom config files are in the correct path.")
        return None
    
    # Crea agente DQN
    print("Initializing DQN agent...")
    agent = DQNCnnAgent(
        state_shape=env.state_shape,
        action_size=env.action_size
    )
    
    # Statistiche
    total_rewards = []
    episode_lengths = []
    helper_calls = 0
    valid_responses = 0
    
    print(f"\nStarting data collection for {episodes} episodes...")
    print(f"Helper model: {HELPER_MODEL_NAME}")
    print(f"Helper call frequency: every {helper_frequency} steps")
    print(f"Plan size: {plan_size} actions")
    print(f"Statistics output: {stats_output_dir}\n")
    
    try:
        # Client non necessario con requests, usiamo un placeholder
        client = None

        for episode in range(episodes):
            state = env.reset()
            done = False
            episode_reward = 0
            step = 0
            helper_calls_this_episode = 0
            episode_actions = []  # Traccia azioni per statistiche
            helper_used_this_episode = False
            
            # Coda per le azioni suggerite dall'Helper
            action_queue = deque()
            
            while not done:
                # Decidi se chiamare l'Helper
                should_call_helper = (
                    len(action_queue) == 0 and 
                    step % helper_frequency == 0 and
                    helper_calls_this_episode < HELPER_CALLS_PER_EPISODE
                )
                
                if should_call_helper:
                    # Ottieni descrizione stato
                    game_state_desc = env.describe_game_state()
                    
                    # Chiama Helper
                    helper_response, response_time = call_helper(
                        client, game_state_desc, plan_size
                    )
                    
                    helper_calls += 1
                    helper_calls_this_episode += 1
                    helper_used_this_episode = True
                    
                    # Parsa azioni
                    parsed_actions = parse_action_plan(
                        helper_response, 
                        env.action_names
                    )
                    
                    # Verifica validità
                    valid_actions = env.get_valid_actions()
                    was_valid = len(parsed_actions) > 0
                    
                    # Calcola action score per ogni azione suggerita
                    action_scores = []
                    for action_name in parsed_actions:
                        score = get_action_score(game_state_desc, action_name, scenario)
                        action_scores.append(score)
                    
                    # Registra nelle statistiche
                    avg_action_score = np.mean(action_scores) if action_scores else 0.0
                    training_stats.record_action_score(avg_action_score)
                    training_stats.record_helper_response(
                        was_valid=was_valid,
                        response_time=response_time
                    )
                    
                    # Check per allucinazione (azioni non valide)
                    if not was_valid or len(parsed_actions) < PLAN_SIZE:
                        training_stats.record_hallucination()
                    
                    if was_valid:
                        valid_responses += 1
                        # Aggiungi azioni alla coda
                        for action_name in parsed_actions:
                            action_idx = env.get_action_from_name(action_name)
                            if action_idx is not None and action_idx in valid_actions:
                                action_queue.append((action_name, action_idx))
                    
                    # Log interazione (reward sarà aggiornato dopo)
                    collector.log_interaction(
                        episode=episode,
                        step=step,
                        game_state=game_state_desc,
                        helper_response=helper_response,
                        parsed_actions=parsed_actions,
                        actions_executed=[],  # Sarà aggiornato
                        rewards_obtained=[],  # Sarà aggiornato
                        was_valid=was_valid,
                        response_time=response_time
                    )
                    
                    if episode % 50 == 0:
                        print(f"Episode {episode}, Step {step}: Helper called")
                        print(f"  Parsed actions: {parsed_actions}")
                
                # Seleziona azione
                if len(action_queue) > 0:
                    action_name, action = action_queue.popleft()
                    action_source = "helper"
                else:
                    # Usa policy DQN
                    valid_actions = env.get_valid_actions()
                    action = agent.act(state, valid_actions)
                    action_name = env.action_names[action] if action < len(env.action_names) else "UNKNOWN"
                    action_source = "dqn"
                
                # Traccia azione per statistiche
                episode_actions.append(action_name)
                training_stats.record_action(action_name)
                
                # Esegui azione
                next_state, reward, done, info = env.step(action)
                episode_reward += reward
                
                # Salva transizione per training DQN
                agent.remember(state, action, reward, next_state, done)
                
                # Training step
                if len(agent.memory) > agent.batch_size:
                    agent.replay()
                
                state = next_state
                step += 1
            
            # Fine episodio
            agent.decay_epsilon()
            total_rewards.append(episode_reward)
            episode_lengths.append(step)
            
            # Registra statistiche episodio
            victory = info.get('victory', False)
            training_stats.record_episode(
                episode=episode,
                reward=episode_reward,
                length=step,
                epsilon=agent.epsilon,
                win=victory,
                helper_used=helper_used_this_episode,
                loss=agent.last_loss if hasattr(agent, 'last_loss') else None
            )
            
            # Aggiorna ultimo log con outcome
            if collector.data:
                collector.data[-1]['episode_outcome'] = 'victory' if victory else 'defeat'
            
            # Progress
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(total_rewards[-100:])
                avg_length = np.mean(episode_lengths[-100:])
                valid_rate = valid_responses / max(helper_calls, 1) * 100
                print(f"Episode {episode + 1}/{episodes} | "
                        f"Avg Reward: {avg_reward:.2f} | "
                        f"Avg Length: {avg_length:.1f} | "
                        f"Helper Calls: {helper_calls} | "
                        f"Valid Rate: {valid_rate:.1f}%")
                          
    except KeyboardInterrupt:
        print("\nData collection interrupted by user.")
    finally:
        env.close()
    
    # Salva dati
    print(f"\n{'='*60}")
    print("Data Collection Complete")
    print(f"{'='*60}")
    
    output_path = collector.save()
    stats = collector.get_stats()
    
    print(f"\nHelper Data Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Genera statistiche e grafici
    if generate_plots:
        print(f"\n{'='*60}")
        print("Generating Training Statistics and Plots")
        print(f"{'='*60}")
        
        # Salva statistiche
        training_stats.save_stats()
        
        # Genera grafici
        #visualizer = TrainingVisualizer(training_stats)
        #visualizer.generate_all_plots()
        
        # Genera report Markdown
        report_path = training_stats.generate_markdown_report()
        
        print(f"\nStatistics and plots saved to: {stats_output_dir}")
        print(f"Report: {report_path}")
    
    # Salva anche il modello DQN (opzionale)
    model_path = os.path.join(OUTPUT_DIR, f"dqn_model_{scenario}")
    agent.save(model_path)
    
    return output_path, stats_output_dir if generate_plots else output_path


# ================== MAIN ==================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Collect Helper responses for VizDoom")
    parser.add_argument('--scenario', type=str, default='deadly_corridor',
                        choices=['basic', 'deadly_corridor', 'defend_the_center'],
                        help='VizDoom scenario to use')
    parser.add_argument('--episodes', type=int, default=500,
                        help='Number of episodes to collect')
    parser.add_argument('--visible', action='store_true',
                        help='Show game window')
    parser.add_argument('--config-path', type=str, default=None,
                        help='Path to VizDoom config files')
    parser.add_argument('--helper-model', type=str, default='qwen2.5-7b-instruct',
                        help='Helper model name in LM Studio')
    parser.add_argument('--plan-size', type=int, default=5,
                        help='Number of actions requested from Helper (default: 5)')
    parser.add_argument('--helper-frequency', type=int, default=HELPER_CALL_FREQUENCY,
                        help='Call Helper every N steps (default: 10)')
    parser.add_argument('--no-plots', action='store_true',
                        help='Disable plot generation')
    
    args = parser.parse_args()
    
    # Aggiorna configurazione
    HELPER_MODEL_NAME = args.helper_model
    
    # Esegui raccolta dati
    result = collect_helper_data(
        episodes=args.episodes,
        scenario=args.scenario,
        visible=args.visible,
        config_path=args.config_path,
        generate_plots=not args.no_plots,
        plan_size=args.plan_size,
        helper_frequency=args.helper_frequency
    )
    
    if result:
        if isinstance(result, tuple):
            output_file, stats_dir = result
            print(f"\nData saved to: {output_file}")
            print(f"Statistics saved to: {stats_dir}")
        else:
            print(f"\nData saved to: {result}")
        print("Use this file for Reviewer fine-tuning.")
