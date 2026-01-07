"""
VizDoom Environment Wrapper per HeRoN
Supporta i task: Basic, Deadly Corridor, Defend The Center
"""

import numpy as np
import cv2
from vizdoom import DoomGame, Mode, ScreenFormat, ScreenResolution, GameVariable
from collections import deque


class VizDoomEnv:
    """
    Wrapper per l'environment VizDoom che espone un'interfaccia compatibile con HeRoN.
    Gestisce preprocessing dei frame, stacking, e generazione di descrizioni testuali per l'Helper.
    """
    
    # Configurazioni per i tre task
    SCENARIOS = {
        'basic': {
            'config': 'basic.cfg',
            'wad': 'basic.wad',
            'actions': ['MOVE_LEFT', 'MOVE_RIGHT', 'ATTACK'],
            'objective': 'Shoot the monster in front of you',
            'description': 'A simple room with one monster ahead'
        },
        'deadly_corridor': {
            'config': 'deadly_corridor.cfg',
            'wad': 'deadly_corridor.wad',
            'actions': ['MOVE_LEFT', 'MOVE_RIGHT', 'ATTACK', 'MOVE_FORWARD', 'MOVE_BACKWARD', 'TURN_LEFT', 'TURN_RIGHT'],
            'objective': 'Navigate through the corridor and reach the green vest while surviving enemy fire',
            'description': 'A corridor with enemies on both sides, green vest at the end'
        },
        'defend_the_center': {
            'config': 'defend_the_center.cfg',
            'wad': 'defend_the_center.wad',
            'actions': ['TURN_LEFT', 'TURN_RIGHT', 'ATTACK'],
            'objective': 'Survive as long as possible by killing approaching monsters',
            'description': 'Circular arena with monsters approaching from all directions'
        }
    }
    
    def __init__(self, scenario='deadly_corridor', frame_stack=4, frame_size=(84, 84), 
                 visible=False, config_path=None):
        """
        Inizializza l'environment VizDoom.
        
        Args:
            scenario: Nome del task ('basic', 'deadly_corridor', 'defend_the_center')
            frame_stack: Numero di frame da stackare (default 4)
            frame_size: Dimensione del frame preprocessato (default 84x84)
            visible: Se True, mostra la finestra di gioco
            config_path: Path personalizzato per i file di configurazione
        """
        self.scenario_name = scenario
        self.scenario = self.SCENARIOS[scenario]
        self.frame_stack = frame_stack
        self.frame_size = frame_size
        
        # Inizializza il gioco
        self.game = DoomGame()
        
        # Carica configurazione
        if config_path:
            self.game.load_config(f"{config_path}/{self.scenario['config']}")
            self.game.set_doom_scenario_path(f"{config_path}/{self.scenario['wad']}")
        else:
            # Usa i file nella cartella doom_files/ (da creare o specificare)
            self.game.load_config(f"doom_files/{self.scenario['config']}")
            self.game.set_doom_scenario_path(f"doom_files/{self.scenario['wad']}")
        
        # Configura il rendering
        self.game.set_window_visible(visible)
        self.game.set_mode(Mode.PLAYER)  # PLAYER mode per test e training
        self.game.set_screen_format(ScreenFormat.RGB24)
        self.game.set_screen_resolution(ScreenResolution.RES_640X480)
        self.sleep_time = 0.028 if not visible else 0.001

        # Definisci le azioni disponibili (one-hot encoding)
        self.action_names = self.scenario['actions']
        self.action_size = len(self.action_names)
        self.possible_actions = np.eye(self.action_size, dtype=int).tolist()
        
        # Buffer per frame stacking
        self.frames = deque(maxlen=frame_stack)
        
        # Variabili di stato per tracking
        self.last_health = 100
        self.last_ammo = 0
        self.last_kill_count = 0
        self.episode_reward = 0
        self.episode_length = 0
        
        # State size per la CNN
        self.state_shape = (frame_stack, frame_size[0], frame_size[1])
        
    def init(self):
        """Inizializza il gioco VizDoom."""
        self.game.init()
        
    def close(self):
        """Chiude il gioco VizDoom."""
        self.game.close()
        
    def preprocess_frame(self, frame):
        """
        Preprocessa un singolo frame: resize, grayscale, normalizzazione.
        
        Args:
            frame: Frame RGB dal gioco (H, W, C) o (C, H, W)
            
        Returns:
            Frame preprocessato (frame_size[0], frame_size[1])
        """
        if frame is None:
            return np.zeros(self.frame_size, dtype=np.float32)
        
        # Debug: stampa shape del frame (rimuovere dopo debug)
        # print(f"DEBUG: frame.shape = {frame.shape}")
        
        # Converti a grayscale
        if len(frame.shape) == 3:
            # VizDoom può restituire (Channel, Height, Width) o (Height, Width, Channel)
            # Controlliamo quale dimensione è quella dei canali (tipicamente 3 o 4)
            
            # Se la prima dimensione è piccola (3 o 4), è Channel-First
            if frame.shape[0] in [3, 4]:
                frame = np.transpose(frame, (1, 2, 0))
            # Se l'ultima dimensione è grande (es. 480, 640), è ancora Channel-First
            elif frame.shape[2] > 4:
                frame = np.transpose(frame, (1, 2, 0))
            
            # Assicurati che l'array sia C-contiguous per OpenCV
            if not frame.flags['C_CONTIGUOUS']:
                frame = np.ascontiguousarray(frame, dtype=frame.dtype)

            # Ora frame dovrebbe essere (H, W, C) con C in {3, 4}
            if frame.shape[2] == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            elif frame.shape[2] == 4:
                gray = cv2.cvtColor(frame, cv2.COLOR_RGBA2GRAY)
            else:
                # Fallback: prendi solo il primo canale
                gray = frame[:, :, 0]
        elif len(frame.shape) == 2:
            # Già grayscale
            gray = frame
        else:
            # Fallback per shape inattese
            gray = np.zeros(self.frame_size, dtype=np.float32)
            
        # Crop della regione di interesse (rimuovi HUD se necessario)
        # Per deadly_corridor: crop verticale per rimuovere bordi
        if self.scenario_name == 'deadly_corridor':
            # Crop: rimuovi parte superiore e inferiore
            h, w = gray.shape
            crop_top = int(h * 0.1)
            crop_bottom = int(h * 0.9)
            gray = gray[crop_top:crop_bottom, :]
        
        # Resize a dimensione target
        resized = cv2.resize(gray, self.frame_size, interpolation=cv2.INTER_AREA)
        
        # Normalizza a [0, 1]
        normalized = resized.astype(np.float32) / 255.0
        
        return normalized
    
    def stack_frames(self, frame, is_new_episode=False):
        """
        Aggiunge un frame allo stack e restituisce lo stack completo.
        
        Args:
            frame: Frame preprocessato
            is_new_episode: Se True, riempie tutto lo stack con lo stesso frame
            
        Returns:
            Stack di frame (frame_stack, H, W)
        """
        if is_new_episode:
            # Riempi tutto lo stack con il primo frame
            self.frames.clear()
            for _ in range(self.frame_stack):
                self.frames.append(frame)
        else:
            self.frames.append(frame)
            
        # Converti a numpy array
        stacked = np.array(self.frames, dtype=np.float32)
        return stacked
    
    def reset(self):
        """
        Resetta l'environment per un nuovo episodio.
        
        Returns:
            Stato iniziale (stack di frame)
        """
        self.game.new_episode()
        
        # Reset tracking variables
        self.last_health = self._get_health()
        self.last_ammo = self._get_ammo()
        self.last_kill_count = self._get_kill_count()
        self.episode_reward = 0
        self.episode_length = 0
        
        # Ottieni primo frame e crea stack
        frame = self.game.get_state().screen_buffer
        if frame is not None:
            frame = frame.transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
        processed = self.preprocess_frame(frame)
        state = self.stack_frames(processed, is_new_episode=True)
        
        return state
    
    def step(self, action):
        """
        Esegue un'azione nell'environment.
        
        Args:
            action: Indice dell'azione da eseguire
            
        Returns:
            Tuple (next_state, reward, done, info)
        """
        # Esegui l'azione
        reward = self.game.make_action(self.possible_actions[action])
        done = self.game.is_episode_finished()
        
        # Calcola reward shaping aggiuntivo
        shaped_reward = self._shape_reward(reward, done)
        
        self.episode_reward += shaped_reward
        self.episode_length += 1
        
        # Ottieni prossimo stato
        if done:
            next_state = np.zeros(self.state_shape, dtype=np.float32)
            info = self._get_episode_info()
        else:
            frame = self.game.get_state().screen_buffer
            if frame is not None:
                frame = frame.transpose(1, 2, 0)
            processed = self.preprocess_frame(frame)
            next_state = self.stack_frames(processed)
            info = self._get_step_info()
            
        return next_state, shaped_reward, done, info
    
    def _shape_reward(self, base_reward, done):
        """
        Applica reward shaping per migliorare l'apprendimento.
        
        Args:
            base_reward: Reward base dal gioco
            done: Se l'episodio è terminato
            
        Returns:
            Reward modificato
        """
        shaped = base_reward
        
        # Reward per sopravvivenza
        current_health = self._get_health()
        health_delta = current_health - self.last_health
        if health_delta < 0:
            shaped += health_delta * 0.1  # Penalità per danno subito
        self.last_health = current_health
        
        # Reward per uccisioni
        current_kills = self._get_kill_count()
        kill_delta = current_kills - self.last_kill_count
        if kill_delta > 0:
            shaped += kill_delta * 10  # Bonus per uccisioni
        self.last_kill_count = current_kills
        
        # Penalità per morte
        if done and current_health <= 0:
            shaped -= 50
            
        return shaped
    
    def _get_health(self):
        """Restituisce la salute attuale del player."""
        try:
            return self.game.get_game_variable(GameVariable.HEALTH)
        except:
            return 100
    
    def _get_ammo(self):
        """Restituisce le munizioni attuali."""
        try:
            return self.game.get_game_variable(GameVariable.AMMO2)
        except:
            return 0
    
    def _get_armor(self):
        """Restituisce l'armatura attuale."""
        try:
            return self.game.get_game_variable(GameVariable.ARMOR)
        except:
            return 0
    
    def _get_kill_count(self):
        """Restituisce il numero di uccisioni."""
        try:
            return self.game.get_game_variable(GameVariable.KILLCOUNT)
        except:
            return 0
    
    def _get_step_info(self):
        """Restituisce informazioni sullo step corrente."""
        return {
            'health': self._get_health(),
            'ammo': self._get_ammo(),
            'armor': self._get_armor(),
            'kills': self._get_kill_count(),
            'episode_length': self.episode_length
        }
    
    def _get_episode_info(self):
        """Restituisce informazioni sull'episodio terminato."""
        return {
            'health': self._get_health(),
            'ammo': self._get_ammo(),
            'armor': self._get_armor(),
            'kills': self._get_kill_count(),
            'episode_length': self.episode_length,
            'episode_reward': self.episode_reward,
            'victory': self._get_health() > 0
        }
    
    def get_valid_actions(self):
        """
        Restituisce le azioni valide nello stato corrente.
        Per ora tutte le azioni sono sempre valide, ma può essere esteso
        per mascherare ATTACK quando ammo=0.
        
        Returns:
            Lista di indici delle azioni valide
        """
        valid = list(range(self.action_size))
        
        # Maschera ATTACK se non ci sono munizioni
        ammo = self._get_ammo()
        if ammo <= 0 and 'ATTACK' in self.action_names:
            attack_idx = self.action_names.index('ATTACK')
            if attack_idx in valid:
                valid.remove(attack_idx)
                
        # Se nessuna azione valida, permetti tutte (fallback)
        if len(valid) == 0:
            valid = list(range(self.action_size))
            
        return valid
    
    def describe_game_state(self):
        """
        Genera una descrizione testuale dello stato di gioco per l'Helper LLM.
        
        Returns:
            Stringa con la descrizione dello stato
        """
        health = self._get_health()
        ammo = self._get_ammo()
        armor = self._get_armor()
        kills = self._get_kill_count()
        
        # Calcola stato di salute in percentuale e descrizione
        if health > 70:
            health_status = "good"
        elif health > 30:
            health_status = "moderate"
        else:
            health_status = "critical"
            
        # Stato munizioni
        if ammo > 20:
            ammo_status = "plenty"
        elif ammo > 5:
            ammo_status = "moderate"
        elif ammo > 0:
            ammo_status = "low"
        else:
            ammo_status = "empty"
        
        # Costruisci descrizione
        description = (
            f"Player health: {health}% ({health_status}). "
            f"Ammo: {ammo} ({ammo_status}). "
            f"Armor: {armor}. "
            f"Kills: {kills}. "
            f"Scenario: {self.scenario['description']}. "
            f"Objective: {self.scenario['objective']}. "
            f"Available actions: {', '.join([f'[{a}]' for a in self.action_names])}."
        )
        
        # Aggiungi avvisi contestuali
        warnings = []
        if health_status == "critical":
            warnings.append("WARNING: Health is critical, prioritize survival!")
        if ammo_status == "empty":
            warnings.append("WARNING: No ammo, cannot attack!")
        elif ammo_status == "low":
            warnings.append("CAUTION: Ammo is low, conserve shots.")
            
        if warnings:
            description += " " + " ".join(warnings)
            
        return description
    
    def get_action_from_name(self, action_name):
        """
        Converte il nome di un'azione nel suo indice.
        
        Args:
            action_name: Nome dell'azione (es. 'ATTACK', 'MOVE_LEFT')
            
        Returns:
            Indice dell'azione o None se non trovata
        """
        action_name = action_name.upper().strip()
        
        # Mapping di alias comuni
        aliases = {
            'SHOOT': 'ATTACK',
            'FIRE': 'ATTACK',
            'LEFT': 'MOVE_LEFT',
            'RIGHT': 'MOVE_RIGHT',
            'FORWARD': 'MOVE_FORWARD',
            'BACKWARD': 'MOVE_BACKWARD',
            'BACK': 'MOVE_BACKWARD',
            'STRAFE_LEFT': 'MOVE_LEFT',
            'STRAFE_RIGHT': 'MOVE_RIGHT',
            'ROTATE_LEFT': 'TURN_LEFT',
            'ROTATE_RIGHT': 'TURN_RIGHT'
        }
        
        # Applica alias se presente
        if action_name in aliases:
            action_name = aliases[action_name]
            
        # Cerca l'azione
        if action_name in self.action_names:
            return self.action_names.index(action_name)
            
        return None


# Funzione di utilità per creare environment
def create_vizdoom_env(scenario='deadly_corridor', **kwargs):
    """
    Factory function per creare un VizDoomEnv.
    
    Args:
        scenario: Nome del task
        **kwargs: Argomenti aggiuntivi per VizDoomEnv
        
    Returns:
        Istanza di VizDoomEnv inizializzata
    """
    env = VizDoomEnv(scenario=scenario, **kwargs)
    env.init()
    return env
