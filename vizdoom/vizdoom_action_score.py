"""
Action Score Calculator per VizDoom
Calcola lo score di coerenza delle azioni per scenari FPS
"""

import numpy as np


# Definizione azioni con parametri per VizDoom
VIZDOOM_ACTIONS = {
    # Azioni di movimento
    'MOVE_LEFT': {
        'type': 'movement',
        'risk': 0.2,           # Rischio di esporsi
        'escape_value': 0.6,   # Valore per fuggire
        'attack_value': 0.0,   # Valore offensivo
    },
    'MOVE_RIGHT': {
        'type': 'movement',
        'risk': 0.2,
        'escape_value': 0.6,
        'attack_value': 0.0,
    },
    'MOVE_FORWARD': {
        'type': 'movement',
        'risk': 0.5,           # Più rischioso (avvicina ai nemici)
        'escape_value': 0.3,
        'attack_value': 0.4,   # Utile per avvicinarsi all'obiettivo
    },
    'MOVE_BACKWARD': {
        'type': 'movement',
        'risk': 0.1,           # Meno rischioso
        'escape_value': 0.8,   # Ottimo per fuggire
        'attack_value': 0.0,
    },
    
    # Azioni di rotazione
    'TURN_LEFT': {
        'type': 'rotation',
        'risk': 0.1,
        'escape_value': 0.3,
        'attack_value': 0.2,   # Utile per mirare
    },
    'TURN_RIGHT': {
        'type': 'rotation',
        'risk': 0.1,
        'escape_value': 0.3,
        'attack_value': 0.2,
    },
    
    # Azione di attacco
    'ATTACK': {
        'type': 'attack',
        'risk': 0.3,           # Moderato (richiede fermarsi)
        'escape_value': 0.0,
        'attack_value': 1.0,   # Massimo valore offensivo
        'ammo_required': True,
    },
}


def calculate_action_scores(health, ammo, scenario='deadly_corridor', 
                           kills=0, available_actions=None):
    """
    Calcola gli score di coerenza per ogni azione basandosi sullo stato di gioco.
    
    Args:
        health: Punti vita del player (0-100)
        ammo: Munizioni disponibili
        scenario: Nome dello scenario ('basic', 'deadly_corridor', 'defend_the_center')
        kills: Numero di uccisioni
        available_actions: Lista di azioni disponibili (default: tutte)
        
    Returns:
        Dict con score normalizzati [0,1] per ogni azione
    """
    if available_actions is None:
        available_actions = list(VIZDOOM_ACTIONS.keys())
    
    # Parametri dinamici basati sullo stato
    health_ratio = health / 100.0
    ammo_available = ammo > 0
    health_critical = health < 30
    health_low = health < 50
    ammo_low = 0 < ammo < 10
    ammo_empty = ammo <= 0
    
    # Pesi dinamici in base alla situazione
    if health_critical:
        # Priorità: sopravvivenza
        w_escape = 2.0
        w_attack = 0.3
        w_risk_penalty = 2.0
    elif health_low:
        # Bilanciato ma cauto
        w_escape = 1.2
        w_attack = 0.7
        w_risk_penalty = 1.5
    else:
        # Aggressivo
        w_escape = 0.5
        w_attack = 1.5
        w_risk_penalty = 0.8
    
    # Modifica pesi per scenario
    if scenario == 'deadly_corridor':
        # Obiettivo: raggiungere il vest, MOVE_FORWARD è importante
        w_forward_bonus = 1.5 if not health_critical else 0.5
    elif scenario == 'defend_the_center':
        # Obiettivo: sopravvivere uccidendo, ATTACK e rotazione importanti
        w_attack *= 1.3
        w_forward_bonus = 0.3
    else:  # basic
        # Obiettivo: uccidere il mostro
        w_attack *= 1.5
        w_forward_bonus = 0.5
    
    scores = {}
    
    for action_name in available_actions:
        if action_name not in VIZDOOM_ACTIONS:
            scores[action_name] = 0.0
            continue
            
        action = VIZDOOM_ACTIONS[action_name]
        
        # Score base
        score = 0.0
        
        # Componente di attacco
        if action['type'] == 'attack':
            if ammo_available:
                score += action['attack_value'] * w_attack
            else:
                # Penalità forte se non ci sono munizioni
                score = -1.0
        else:
            score += action['attack_value'] * w_attack
        
        # Componente di fuga/sopravvivenza
        score += action['escape_value'] * w_escape
        
        # Penalità per rischio
        score -= action['risk'] * w_risk_penalty
        
        # Bonus speciali
        if action_name == 'MOVE_FORWARD' and scenario == 'deadly_corridor':
            score += w_forward_bonus
            
        if action_name == 'MOVE_BACKWARD' and health_critical:
            score += 0.5  # Bonus per ritirata quando in pericolo
            
        if action_name in ['TURN_LEFT', 'TURN_RIGHT'] and scenario == 'defend_the_center':
            score += 0.3  # Bonus per scanning in scenario circolare
        
        scores[action_name] = max(score, 0.0)
    
    # Normalizzazione
    max_score = max(scores.values()) if scores.values() else 1.0
    if max_score > 0:
        scores = {k: v / max_score for k, v in scores.items()}
    
    return scores


def get_best_action(health, ammo, scenario='deadly_corridor', 
                   available_actions=None, exclude_actions=None):
    """
    Restituisce l'azione migliore basandosi sugli score.
    
    Args:
        health: Punti vita del player
        ammo: Munizioni disponibili
        scenario: Nome dello scenario
        available_actions: Lista di azioni disponibili
        exclude_actions: Azioni da escludere
        
    Returns:
        Nome dell'azione migliore
    """
    scores = calculate_action_scores(health, ammo, scenario, 
                                     available_actions=available_actions)
    
    if exclude_actions:
        for action in exclude_actions:
            if action in scores:
                scores[action] = 0.0
    
    if not scores:
        return 'MOVE_FORWARD'  # Default
        
    return max(scores, key=scores.get)


def evaluate_action_plan(plan, health, ammo, scenario='deadly_corridor'):
    """
    Valuta un piano di azioni proposto dall'Helper.
    
    Args:
        plan: Lista di nomi di azioni
        health: Punti vita del player
        ammo: Munizioni disponibili
        scenario: Nome dello scenario
        
    Returns:
        Tuple (score_totale, feedback_list)
    """
    if not plan:
        return 0.0, ["Piano vuoto"]
    
    feedback = []
    total_score = 0.0
    simulated_ammo = ammo
    simulated_health = health
    
    for i, action in enumerate(plan):
        action = action.upper().strip()
        
        # Calcola score per questa azione
        scores = calculate_action_scores(
            simulated_health, simulated_ammo, scenario
        )
        
        if action not in scores:
            feedback.append(f"Azione '{action}' non riconosciuta")
            continue
            
        action_score = scores.get(action, 0.0)
        total_score += action_score
        
        # Verifica validità
        if action == 'ATTACK' and simulated_ammo <= 0:
            feedback.append(f"Azione {i+1}: ATTACK non valida - munizioni esaurite")
            action_score = 0.0
        elif action == 'ATTACK':
            simulated_ammo -= 1
            
        # Feedback contestuale
        best_action = get_best_action(simulated_health, simulated_ammo, scenario)
        if action != best_action and action_score < scores[best_action] * 0.5:
            feedback.append(
                f"Azione {i+1}: '{action}' subottimale, considera '{best_action}'"
            )
    
    # Normalizza score
    avg_score = total_score / len(plan) if plan else 0.0
    
    # Feedback generale
    if avg_score > 0.7:
        feedback.insert(0, "Piano buono nel complesso")
    elif avg_score > 0.4:
        feedback.insert(0, "Piano accettabile con margini di miglioramento")
    else:
        feedback.insert(0, "Piano da rivedere")
    
    return avg_score, feedback


def generate_corrective_feedback(game_state, helper_response, parsed_actions,
                                was_valid, reward_obtained, health, ammo, 
                                scenario='deadly_corridor'):
    """
    Genera feedback correttivi per il fine-tuning del Reviewer.
    Usato per creare il dataset di training.
    
    Args:
        game_state: Descrizione dello stato di gioco
        helper_response: Risposta originale dell'Helper
        parsed_actions: Azioni estratte
        was_valid: Se le azioni erano valide
        reward_obtained: Reward ottenuto
        health: Punti vita al momento della chiamata
        ammo: Munizioni al momento della chiamata
        scenario: Nome dello scenario
        
    Returns:
        Stringa con il feedback correttivo
    """
    if not was_valid:
        return (
            f"La risposta non contiene azioni valide. "
            f"Fornisci azioni nel formato [\"ACTION1\", \"ACTION2\", ...] "
            f"usando solo azioni disponibili: MOVE_LEFT, MOVE_RIGHT, ATTACK, "
            f"MOVE_FORWARD, MOVE_BACKWARD, TURN_LEFT, TURN_RIGHT."
        )
    
    # Valuta il piano
    avg_score, feedback_list = evaluate_action_plan(
        parsed_actions, health, ammo, scenario
    )
    
    # Genera feedback strutturato
    feedback_parts = []
    
    # Feedback su validità azioni
    if ammo <= 0 and 'ATTACK' in [a.upper() for a in parsed_actions]:
        feedback_parts.append(
            "ATTACK non è possibile senza munizioni. "
            "Rimuovi ATTACK dal piano o suggerisci azioni di movimento."
        )
    
    # Feedback su coerenza con stato salute
    if health < 30:
        if 'MOVE_FORWARD' in [a.upper() for a in parsed_actions]:
            feedback_parts.append(
                "Con salute critica, avanzare è rischioso. "
                "Considera MOVE_BACKWARD o movimenti laterali per sopravvivere."
            )
    
    # Feedback su scenario specifico
    if scenario == 'deadly_corridor':
        forward_count = sum(1 for a in parsed_actions if a.upper() == 'MOVE_FORWARD')
        if forward_count == 0 and health > 50:
            feedback_parts.append(
                "L'obiettivo è raggiungere il vest. "
                "Includi MOVE_FORWARD nel piano quando la salute lo permette."
            )
    elif scenario == 'defend_the_center':
        attack_count = sum(1 for a in parsed_actions if a.upper() == 'ATTACK')
        turn_count = sum(1 for a in parsed_actions if 'TURN' in a.upper())
        if attack_count < 2 and ammo > 5:
            feedback_parts.append(
                "In difesa, prioritizza l'attacco. Aumenta le azioni ATTACK."
            )
        if turn_count == 0:
            feedback_parts.append(
                "Aggiungi TURN_LEFT o TURN_RIGHT per controllare i nemici attorno."
            )
    
    # Feedback su reward
    if reward_obtained is not None and reward_obtained < 0:
        feedback_parts.append(
            "Il piano ha portato a reward negativo. Rivedi la strategia."
        )
    
    # Feedback generale basato su score
    if avg_score < 0.4:
        best_action = get_best_action(health, ammo, scenario)
        feedback_parts.append(
            f"Piano subottimale. L'azione migliore in questo stato è [{best_action}]."
        )
    elif avg_score > 0.7:
        feedback_parts.append("La risposta è appropriata.")
    
    # Combina feedback
    if not feedback_parts:
        return "La risposta è appropriata."
    
    return " ".join(feedback_parts)


# ================== TEST ==================

if __name__ == "__main__":
    # Test delle funzioni
    print("Test Action Score Calculator\n")
    
    # Stato normale
    print("Stato: Health=80, Ammo=20")
    scores = calculate_action_scores(80, 20, 'deadly_corridor')
    for action, score in sorted(scores.items(), key=lambda x: -x[1]):
        print(f"  {action}: {score:.3f}")
    
    print("\nStato critico: Health=20, Ammo=5")
    scores = calculate_action_scores(20, 5, 'deadly_corridor')
    for action, score in sorted(scores.items(), key=lambda x: -x[1]):
        print(f"  {action}: {score:.3f}")
    
    print("\nSenza munizioni: Health=50, Ammo=0")
    scores = calculate_action_scores(50, 0, 'deadly_corridor')
    for action, score in sorted(scores.items(), key=lambda x: -x[1]):
        print(f"  {action}: {score:.3f}")
    
    # Test valutazione piano
    print("\n\nTest Valutazione Piano")
    plan = ['MOVE_FORWARD', 'ATTACK', 'MOVE_LEFT', 'ATTACK', 'MOVE_FORWARD']
    score, feedback = evaluate_action_plan(plan, 60, 10, 'deadly_corridor')
    print(f"Piano: {plan}")
    print(f"Score: {score:.3f}")
    print(f"Feedback: {feedback}")
    
    # Test feedback correttivo
    print("\n\nTest Feedback Correttivo")
    feedback = generate_corrective_feedback(
        game_state="Health 20%, Ammo 0",
        helper_response="[ATTACK, ATTACK, MOVE_FORWARD]",
        parsed_actions=['ATTACK', 'ATTACK', 'MOVE_FORWARD'],
        was_valid=True,
        reward_obtained=-10,
        health=20,
        ammo=0,
        scenario='deadly_corridor'
    )
    print(f"Feedback: {feedback}")
