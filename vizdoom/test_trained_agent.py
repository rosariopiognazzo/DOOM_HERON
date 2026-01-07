#!/usr/bin/env python3
"""
Script per testare l'agente DQN addestrato su VizDoom.
Mostra la finestra del gioco e le statistiche in tempo reale.

Uso:
    python test_trained_agent.py --scenario basic --model helper_logs/dqn_model_basic --episodes 5
"""

import os
import sys
import time
import argparse
import numpy as np
from datetime import datetime

from vizdoom_env import VizDoomEnv, create_vizdoom_env
from vizdoom_agent import DQNCnnAgent


def test_agent(scenario='basic', model_path=None, episodes=5, epsilon=0.01, visible=True, sleep_time=0.01):
    """
    Testa l'agente addestrato visualizzando il gioco.
    
    Args:
        scenario: Nome dello scenario ('basic', 'deadly_corridor', 'defend_the_center')
        model_path: Path al modello salvato (senza estensione)
        episodes: Numero di episodi da testare
        epsilon: Valore epsilon per esplorazione (0.0 = sempre greedy)
        visible: Se True, mostra la finestra di gioco
        sleep_time: Tempo di attesa tra frame (in secondi)
    """
    
    print(f"\n🎮 TEST DELL'AGENTE ADDESTRATO")
    print(f"{'='*60}")
    print(f"Scenario: {scenario}")
    print(f"Modello: {model_path}")
    print(f"Episodi: {episodes}")
    print(f"Epsilon (esplorazione): {epsilon} (0.0 = sempre greedy)")
    print(f"{'='*60}\n")
    
    # Crea environment (usa create_vizdoom_env per inizializzazione automatica)
    env = create_vizdoom_env(scenario=scenario, visible=visible, frame_stack=4, frame_size=(84, 84))
    
    # Crea agente
    agent = DQNCnnAgent(
        state_shape=env.state_shape,
        action_size=env.action_size
    )
    
    # Carica il modello
    if model_path and os.path.exists(f"{model_path}_policy.keras"):
        agent.load(model_path)
        print(f"✅ Modello caricato da: {model_path}")
    else:
        raise Exception(f"❌ Modello non trovato: {model_path}")
    
    # Imposta epsilon per il test (bassa esplorazione)
    agent.epsilon = epsilon
    print(f"Epsilon impostato a: {epsilon}")
    
    # Statistiche
    episode_rewards = []
    episode_lengths = []
    episode_kills = []
    
    # Testing loop
    for episode in range(episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        
        print(f"\n{'='*60}")
        print(f"Episode {episode + 1}/{episodes}")
        print(f"{'='*60}")
        
        while not done:
            # Scegli azione con policy (usa self.epsilon dell'agente)
            action_idx = agent.act(state)
            action_name = env.action_names[action_idx]
            
            # Esegui azione
            next_state, reward, done, info = env.step(action_idx)
            
            episode_reward += reward
            episode_length += 1
            
            # Mostra informazioni (ogni 10 frame)
            if episode_length % 10 == 0:
                game_vars = info.get('game_variables', {})
                health = game_vars.get('health', 0)
                ammo = game_vars.get('ammo', 0)
                kills = game_vars.get('kills', 0)
                
                print(f"  Step {episode_length:4d} | Action: {action_name:15s} | "
                      f"Reward: {reward:+6.1f} | Total: {episode_reward:+7.1f} | "
                      f"Health: {health:3.0f} | Ammo: {ammo:3.0f} | Kills: {kills}")
            
            state = next_state
            
            # Rallenta per visualizzazione
            if visible and sleep_time > 0:
                time.sleep(sleep_time)
        
        # Fine episodio
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        game_vars = info.get('game_variables', {})
        kills = game_vars.get('kills', 0)
        episode_kills.append(kills)
        
        print(f"\n📊 Episode {episode + 1} Finished!")
        print(f"  Total Reward: {episode_reward:.1f}")
        print(f"  Episode Length: {episode_length} steps")
        print(f"  Kills: {kills}")
    
    # Chiudi environment
    env.close()
    
    # Statistiche finali
    print(f"\n{'='*60}")
    print(f"📊 STATISTICHE FINALI")
    print(f"{'='*60}")
    print(f"Total Episodes: {episodes}")
    print(f"Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Max Reward: {max(episode_rewards):.2f}")
    print(f"Min Reward: {min(episode_rewards):.2f}")
    print(f"Average Length: {np.mean(episode_lengths):.1f} steps")
    print(f"Average Kills: {np.mean(episode_kills):.1f}")
    print(f"{'='*60}\n")
    
    return {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'episode_kills': episode_kills,
        'avg_reward': np.mean(episode_rewards),
        'avg_length': np.mean(episode_lengths),
        'avg_kills': np.mean(episode_kills)
    }


def find_latest_model(scenario, logs_dir='helper_logs'):
    """Trova il modello più recente per lo scenario specificato."""
    model_path = os.path.join(logs_dir, f'dqn_model_{scenario}')
    
    if os.path.exists(f"{model_path}_policy.keras"):
        return model_path
    
    # Cerca modelli con timestamp
    if os.path.exists(logs_dir):
        models = [f for f in os.listdir(logs_dir) if f.startswith(f'dqn_model_{scenario}')]
        if models:
            # Ordina per data di modifica
            latest = max(models, key=lambda f: os.path.getmtime(os.path.join(logs_dir, f)))
            return os.path.join(logs_dir, latest.replace('_policy.keras', ''))
    
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test trained DQN agent on VizDoom')
    parser.add_argument('--scenario', type=str, default='basic',
                        choices=['basic', 'deadly_corridor', 'defend_the_center'],
                        help='Scenario VizDoom da testare')
    parser.add_argument('--model', type=str, default=None,
                        help='Path al modello (senza estensione .keras)')
    parser.add_argument('--episodes', type=int, default=5,
                        help='Numero di episodi da testare')
    parser.add_argument('--epsilon', type=float, default=0.01,
                        help='Epsilon per esplorazione (0.0 = sempre greedy)')
    parser.add_argument('--no-visible', action='store_true',
                        help='Non mostrare la finestra di gioco')
    parser.add_argument('--sleep', type=float, default=0.01,
                        help='Tempo di attesa tra frame (secondi)')
    
    args = parser.parse_args()
    
    # Trova modello se non specificato
    if args.model is None:
        args.model = find_latest_model(args.scenario)
        if args.model is None:
            print(f"\n❌ Nessun modello trovato per lo scenario '{args.scenario}'")
            print(f"Addestra prima un modello usando collect_helper_responses.py")
            sys.exit(1)
        print(f"\n🔍 Modello trovato automaticamente: {args.model}")
    
    # Test
    print(f"\n🚀 Avvio test dell'agente...\n")
    results = test_agent(
        scenario=args.scenario,
        model_path=args.model,
        episodes=args.episodes,
        epsilon=args.epsilon,
        visible=not args.no_visible,
        sleep_time=args.sleep
    )
    
    print("✅ Test completato!")
