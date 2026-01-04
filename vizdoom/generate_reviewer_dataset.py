"""
Script per generare il dataset di fine-tuning del Reviewer
a partire dai log raccolti dalla Fase 1 (Helper zero-shot).
"""

import pandas as pd
import numpy as np
import os
import ast
from datetime import datetime

from vizdoom_action_score import (
    generate_corrective_feedback, 
    evaluate_action_plan,
    calculate_action_scores,
    get_best_action
)


def load_helper_logs(log_path):
    """
    Carica i log delle risposte Helper.
    
    Args:
        log_path: Path al file CSV con i log
        
    Returns:
        DataFrame con i log
    """
    df = pd.read_csv(log_path)
    print(f"Loaded {len(df)} interactions from {log_path}")
    return df


def parse_list_string(s):
    """Converte una stringa rappresentante una lista in una lista Python."""
    if pd.isna(s) or s == '[]' or s == '':
        return []
    try:
        return ast.literal_eval(s)
    except:
        return []


def extract_health_ammo_from_state(game_state):
    """
    Estrae health e ammo dalla descrizione dello stato di gioco.
    
    Args:
        game_state: Stringa con la descrizione dello stato
        
    Returns:
        Tuple (health, ammo)
    """
    import re
    
    health = 50  # Default
    ammo = 10    # Default
    
    # Cerca health
    health_match = re.search(r'health[:\s]+(\d+)', game_state, re.IGNORECASE)
    if health_match:
        health = int(health_match.group(1))
    
    # Cerca ammo
    ammo_match = re.search(r'ammo[:\s]+(\d+)', game_state, re.IGNORECASE)
    if ammo_match:
        ammo = int(ammo_match.group(1))
    
    return health, ammo


def generate_reviewer_dataset(log_df, output_path=None):
    """
    Genera il dataset per il fine-tuning del Reviewer.
    
    Args:
        log_df: DataFrame con i log delle risposte Helper
        output_path: Path per salvare il dataset (opzionale)
        
    Returns:
        DataFrame con il dataset per il Reviewer
    """
    dataset = []
    
    for idx, row in log_df.iterrows():
        # Estrai dati dalla riga
        scenario = row.get('scenario', 'deadly_corridor')
        game_state = row.get('game_state', '')
        helper_response = row.get('helper_response', '')
        parsed_actions = parse_list_string(row.get('parsed_actions', '[]'))
        was_valid = row.get('was_valid', False)
        rewards = parse_list_string(row.get('rewards_obtained', '[]'))
        episode_outcome = row.get('episode_outcome', 'unknown')
        
        # Estrai health e ammo dallo stato
        health, ammo = extract_health_ammo_from_state(game_state)
        
        # Calcola reward medio se disponibile
        avg_reward = np.mean(rewards) if rewards else None
        
        # Genera feedback correttivo
        feedback = generate_corrective_feedback(
            game_state=game_state,
            helper_response=helper_response,
            parsed_actions=parsed_actions,
            was_valid=was_valid,
            reward_obtained=avg_reward,
            health=health,
            ammo=ammo,
            scenario=scenario
        )
        
        # Crea entry per il dataset
        # Formato compatibile con il fine-tuning T5 esistente
        prompt = f"Game state: {game_state}"
        response = helper_response
        instructions = feedback
        
        dataset.append({
            'prompt': prompt,
            'response': response,
            'instructions': instructions,
            'scenario': scenario,
            'was_valid': was_valid,
            'health': health,
            'ammo': ammo,
            'episode_outcome': episode_outcome
        })
        
        # Progress
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(log_df)} interactions")
    
    # Crea DataFrame
    result_df = pd.DataFrame(dataset)
    
    # Salva se richiesto
    if output_path:
        result_df.to_csv(output_path, index=False)
        print(f"Dataset saved to {output_path}")
    
    return result_df


def augment_dataset_with_synthetic(base_df, num_synthetic=1000, scenario='deadly_corridor'):
    """
    Aumenta il dataset con esempi sintetici generati programmaticamente.
    
    Args:
        base_df: DataFrame base
        num_synthetic: Numero di esempi sintetici da generare
        scenario: Scenario per gli esempi sintetici
        
    Returns:
        DataFrame aumentato
    """
    import random
    
    synthetic_data = []
    
    # Azioni disponibili per scenario
    scenario_actions = {
        'basic': ['MOVE_LEFT', 'MOVE_RIGHT', 'ATTACK'],
        'deadly_corridor': ['MOVE_LEFT', 'MOVE_RIGHT', 'ATTACK', 'MOVE_FORWARD', 
                           'MOVE_BACKWARD', 'TURN_LEFT', 'TURN_RIGHT'],
        'defend_the_center': ['TURN_LEFT', 'TURN_RIGHT', 'ATTACK']
    }
    
    actions = scenario_actions.get(scenario, scenario_actions['deadly_corridor'])
    
    for i in range(num_synthetic):
        # Genera stato casuale
        health = random.randint(10, 100)
        ammo = random.randint(0, 30)
        kills = random.randint(0, 10)
        
        # Genera stato testuale
        health_status = "critical" if health < 30 else ("low" if health < 50 else "good")
        ammo_status = "empty" if ammo == 0 else ("low" if ammo < 10 else "plenty")
        
        game_state = (
            f"Player health: {health}% ({health_status}). "
            f"Ammo: {ammo} ({ammo_status}). "
            f"Kills: {kills}. "
            f"Available actions: {', '.join(actions)}."
        )
        
        # Genera piano casuale (simulando Helper)
        plan_size = random.randint(3, 5)
        plan = [random.choice(actions) for _ in range(plan_size)]
        
        # Simula risposta Helper
        helper_response = f'["{"\", \"".join(plan)}"] I suggest these actions based on the current state.'
        
        # Calcola se il piano è valido
        was_valid = True
        if ammo == 0 and 'ATTACK' in plan:
            was_valid = False  # Piano invalido se attacca senza munizioni
        
        # Genera feedback
        feedback = generate_corrective_feedback(
            game_state=game_state,
            helper_response=helper_response,
            parsed_actions=plan,
            was_valid=was_valid,
            reward_obtained=random.uniform(-10, 50) if was_valid else -20,
            health=health,
            ammo=ammo,
            scenario=scenario
        )
        
        synthetic_data.append({
            'prompt': f"Game state: {game_state}",
            'response': helper_response,
            'instructions': feedback,
            'scenario': scenario,
            'was_valid': was_valid,
            'health': health,
            'ammo': ammo,
            'episode_outcome': random.choice(['victory', 'defeat'])
        })
    
    # Combina con dataset base
    synthetic_df = pd.DataFrame(synthetic_data)
    combined_df = pd.concat([base_df, synthetic_df], ignore_index=True)
    
    print(f"Added {num_synthetic} synthetic examples. Total: {len(combined_df)}")
    
    return combined_df


def balance_dataset(df):
    """
    Bilancia il dataset tra risposte valide e non valide.
    
    Args:
        df: DataFrame con il dataset
        
    Returns:
        DataFrame bilanciato
    """
    valid = df[df['was_valid'] == True]
    invalid = df[df['was_valid'] == False]
    
    print(f"Valid responses: {len(valid)}, Invalid: {len(invalid)}")
    
    # Se sbilanciato, sottocampiona la classe maggioritaria
    if len(valid) > len(invalid) * 2:
        valid = valid.sample(n=len(invalid) * 2, random_state=42)
    elif len(invalid) > len(valid) * 2:
        invalid = invalid.sample(n=len(valid) * 2, random_state=42)
    
    balanced = pd.concat([valid, invalid], ignore_index=True)
    balanced = balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Balanced dataset size: {len(balanced)}")
    
    return balanced


def prepare_final_dataset(log_paths, output_path, add_synthetic=True, balance=True):
    """
    Prepara il dataset finale per il fine-tuning del Reviewer.
    
    Args:
        log_paths: Lista di path ai file di log (uno per scenario)
        output_path: Path per salvare il dataset finale
        add_synthetic: Se aggiungere esempi sintetici
        balance: Se bilanciare il dataset
        
    Returns:
        DataFrame con il dataset finale
    """
    all_data = []
    
    # Carica tutti i log
    for log_path in log_paths:
        if os.path.exists(log_path):
            df = load_helper_logs(log_path)
            all_data.append(df)
        else:
            print(f"Warning: {log_path} not found")
    
    if not all_data:
        print("No log files found. Generating synthetic dataset only.")
        combined_logs = pd.DataFrame()
    else:
        combined_logs = pd.concat(all_data, ignore_index=True)
    
    # Genera dataset per Reviewer
    print("\nGenerating Reviewer dataset...")
    dataset = generate_reviewer_dataset(combined_logs)
    
    # Aggiungi esempi sintetici
    if add_synthetic:
        print("\nAdding synthetic examples...")
        for scenario in ['basic', 'deadly_corridor', 'defend_the_center']:
            dataset = augment_dataset_with_synthetic(
                dataset, 
                num_synthetic=500, 
                scenario=scenario
            )
    
    # Bilancia il dataset
    if balance:
        print("\nBalancing dataset...")
        dataset = balance_dataset(dataset)
    
    # Salva dataset finale
    dataset.to_csv(output_path, index=False)
    print(f"\nFinal dataset saved to {output_path}")
    print(f"Total examples: {len(dataset)}")
    
    # Statistiche
    print("\nDataset Statistics:")
    print(f"  Scenarios: {dataset['scenario'].value_counts().to_dict()}")
    print(f"  Valid responses: {dataset['was_valid'].sum()} ({dataset['was_valid'].mean()*100:.1f}%)")
    print(f"  Avg health: {dataset['health'].mean():.1f}")
    print(f"  Avg ammo: {dataset['ammo'].mean():.1f}")
    
    return dataset


# ================== MAIN ==================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate Reviewer fine-tuning dataset")
    parser.add_argument('--logs', nargs='+', type=str,
                        help='Paths to Helper log files')
    parser.add_argument('--output', type=str, 
                        default='vizdoom_reviewer_dataset.csv',
                        help='Output path for the dataset')
    parser.add_argument('--synthetic', type=int, default=1500,
                        help='Number of synthetic examples to add')
    parser.add_argument('--no-balance', action='store_true',
                        help='Do not balance the dataset')
    
    args = parser.parse_args()
    
    # Se non ci sono log, genera solo dataset sintetico
    if not args.logs:
        print("No log files specified. Generating synthetic dataset only.")
        
        # Crea dataset sintetico per tutti gli scenari
        all_synthetic = pd.DataFrame()
        for scenario in ['basic', 'deadly_corridor', 'defend_the_center']:
            all_synthetic = augment_dataset_with_synthetic(
                all_synthetic,
                num_synthetic=args.synthetic // 3,
                scenario=scenario
            )
        
        if not args.no_balance:
            all_synthetic = balance_dataset(all_synthetic)
        
        all_synthetic.to_csv(args.output, index=False)
        print(f"\nSynthetic dataset saved to {args.output}")
        print(f"Total examples: {len(all_synthetic)}")
    else:
        # Usa i log forniti
        prepare_final_dataset(
            log_paths=args.logs,
            output_path=args.output,
            add_synthetic=args.synthetic > 0,
            balance=not args.no_balance
        )
