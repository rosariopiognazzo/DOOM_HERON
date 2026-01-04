"""
DQN Agent con CNN per VizDoom
Implementazione TensorFlow/Keras con replay buffer e target network
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model, Sequential, clone_model, load_model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input, Lambda
from tensorflow.keras.optimizers import Adam
import numpy as np
import random
from collections import deque
import os
import pickle


def build_dqn_cnn(input_shape, action_size):
    """
    Costruisce una rete CNN per DQN che processa stack di frame.
    Architettura: 3 layer convoluzionali + 2 layer fully connected (Nature DQN)
    
    Args:
        input_shape: Tuple (height, width, channels) es. (84, 84, 4)
        action_size: Numero di azioni possibili
        
    Returns:
        Modello Keras compilato
    """
    model = Sequential([
        # Layer convoluzionali (architettura Nature DQN)
        Conv2D(32, kernel_size=8, strides=4, activation='relu', 
               input_shape=input_shape, data_format='channels_last'),
        Conv2D(64, kernel_size=4, strides=2, activation='relu'),
        Conv2D(64, kernel_size=3, strides=1, activation='relu'),
        
        # Flatten e layer fully connected
        Flatten(),
        Dense(512, activation='relu'),
        Dense(action_size, activation='linear')
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse')
    return model


def build_dueling_dqn_cnn(input_shape, action_size):
    """
    Costruisce una rete Dueling DQN - separa value e advantage streams.
    Migliora la stabilità dell'apprendimento.
    
    Args:
        input_shape: Tuple (height, width, channels) es. (84, 84, 4)
        action_size: Numero di azioni possibili
        
    Returns:
        Modello Keras compilato
    """
    inputs = Input(shape=input_shape)
    
    # Shared convolutional layers
    x = Conv2D(32, kernel_size=8, strides=4, activation='relu')(inputs)
    x = Conv2D(64, kernel_size=4, strides=2, activation='relu')(x)
    x = Conv2D(64, kernel_size=3, strides=1, activation='relu')(x)
    x = Flatten()(x)
    
    # Value stream
    value = Dense(256, activation='relu')(x)
    value = Dense(1)(value)
    
    # Advantage stream
    advantage = Dense(256, activation='relu')(x)
    advantage = Dense(action_size)(advantage)
    
    # Combine: Q = V + (A - mean(A))
    def combine_streams(inputs):
        value, advantage = inputs
        return value + (advantage - tf.reduce_mean(advantage, axis=1, keepdims=True))
    
    q_values = Lambda(combine_streams)([value, advantage])
    
    model = Model(inputs=inputs, outputs=q_values)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse')
    return model


class ReplayBuffer:
    """
    Experience Replay Buffer per DQN.
    Memorizza transizioni (state, action, reward, next_state, done).
    """
    
    def __init__(self, capacity=100000):
        """
        Args:
            capacity: Dimensione massima del buffer
        """
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        """Aggiunge una transizione al buffer."""
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        """
        Campiona un batch casuale dal buffer.
        
        Returns:
            Tuple di array numpy (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int32),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )
    
    def __len__(self):
        return len(self.buffer)


class DQNCnnAgent:
    """
    Agente DQN con CNN per VizDoom.
    Implementazione TensorFlow/Keras.
    Supporta Double DQN, target network, e epsilon-greedy exploration.
    """
    
    def __init__(self, state_shape, action_size, load_model_path=None, use_dueling=False):
        """
        Args:
            state_shape: Tuple (channels, height, width) es. (4, 84, 84)
                        NOTA: Viene convertito internamente a (height, width, channels) per Keras
            action_size: Numero di azioni possibili
            load_model_path: Path per caricare un modello esistente
            use_dueling: Se True, usa architettura Dueling DQN
        """
        # Converti da (channels, height, width) a (height, width, channels) per Keras
        if len(state_shape) == 3:
            self.state_shape_keras = (state_shape[1], state_shape[2], state_shape[0])
        else:
            self.state_shape_keras = state_shape
            
        self.state_shape = state_shape
        self.action_size = action_size
        self.use_dueling = use_dueling
        
        # Configura GPU se disponibile
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"DQNCnnAgent using GPU: {gpus}")
            except RuntimeError as e:
                print(f"GPU configuration error: {e}")
        else:
            print("DQNCnnAgent using CPU")
        
        # Hyperparameters
        self.gamma = 0.99           # Discount factor
        self.epsilon = 1.0          # Exploration rate iniziale
        self.epsilon_min = 0.01     # Exploration rate minimo
        self.epsilon_decay = 0.9995 # Decay per episodio
        self.learning_rate = 0.0001
        self.batch_size = 32
        self.target_update_freq = 1000  # Frequenza aggiornamento target network
        
        # Costruisci reti
        if use_dueling:
            self.policy_net = build_dueling_dqn_cnn(self.state_shape_keras, action_size)
            self.target_net = build_dueling_dqn_cnn(self.state_shape_keras, action_size)
        else:
            self.policy_net = build_dqn_cnn(self.state_shape_keras, action_size)
            self.target_net = build_dqn_cnn(self.state_shape_keras, action_size)
        
        # Copia pesi da policy a target
        self.target_net.set_weights(self.policy_net.get_weights())
        
        # Replay buffer
        self.memory = ReplayBuffer(capacity=100000)
        
        # Contatore per target network update
        self.steps_done = 0
        
        # Carica modello se specificato
        if load_model_path:
            self.load(load_model_path)
    
    def _preprocess_state(self, state):
        """
        Preprocessa lo stato per l'input a Keras.
        Converte da (channels, height, width) a (height, width, channels).
        
        Args:
            state: Stato in formato (channels, height, width)
            
        Returns:
            Stato in formato (height, width, channels)
        """
        if len(state.shape) == 3:
            # Singolo stato: (C, H, W) -> (H, W, C)
            return np.transpose(state, (1, 2, 0))
        elif len(state.shape) == 4:
            # Batch: (B, C, H, W) -> (B, H, W, C)
            return np.transpose(state, (0, 2, 3, 1))
        return state
            
    def act(self, state, valid_actions=None):
        """
        Seleziona un'azione usando epsilon-greedy policy.
        
        Args:
            state: Stato corrente (numpy array)
            valid_actions: Lista di azioni valide (opzionale)
            
        Returns:
            Indice dell'azione selezionata
        """
        if valid_actions is None:
            valid_actions = list(range(self.action_size))
            
        # Epsilon-greedy
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        
        # Selezione greedy
        state_processed = self._preprocess_state(state)
        state_batch = np.expand_dims(state_processed, axis=0)
        q_values = self.policy_net.predict(state_batch, verbose=0)[0]
        
        # Maschera azioni non valide
        if len(valid_actions) < self.action_size:
            masked_q = np.full_like(q_values, -np.inf)
            masked_q[valid_actions] = q_values[valid_actions]
            return np.argmax(masked_q)
        
        return np.argmax(q_values)
    
    def remember(self, state, action, reward, next_state, done):
        """Salva una transizione nel replay buffer."""
        self.memory.push(state, action, reward, next_state, done)
        
    def replay(self, batch_size=None):
        """
        Esegue un passo di training usando experience replay.
        
        Args:
            batch_size: Dimensione del batch (default: self.batch_size)
            
        Returns:
            Loss del passo di training
        """
        if batch_size is None:
            batch_size = self.batch_size
            
        if len(self.memory) < batch_size:
            return 0
        
        # Campiona batch
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        
        # Preprocessa stati per Keras (channels last)
        states = self._preprocess_state(states)
        next_states = self._preprocess_state(next_states)
        
        # Calcola Q-values correnti
        current_q_values = self.policy_net.predict(states, verbose=0)
        
        # Double DQN: usa policy net per selezionare azione, target net per valutare
        next_q_policy = self.policy_net.predict(next_states, verbose=0)
        next_q_target = self.target_net.predict(next_states, verbose=0)
        
        # Azione migliore secondo policy network
        best_actions = np.argmax(next_q_policy, axis=1)
        
        # Q-value da target network per l'azione selezionata
        next_q = next_q_target[np.arange(batch_size), best_actions]
        
        # Calcola target Q-values
        target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Aggiorna solo i Q-values delle azioni eseguite
        target_q_values = current_q_values.copy()
        target_q_values[np.arange(batch_size), actions] = target_q
        
        # Training step
        history = self.policy_net.fit(states, target_q_values, 
                                       batch_size=batch_size, 
                                       epochs=1, verbose=0)
        loss = history.history['loss'][0]
        
        # Aggiorna target network
        self.steps_done += 1
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.set_weights(self.policy_net.get_weights())
            
        return loss
    
    def decay_epsilon(self):
        """Decay dell'epsilon per ridurre l'esplorazione nel tempo."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
    def save(self, path):
        """
        Salva il modello e lo stato dell'agente.
        
        Args:
            path: Path base per il salvataggio (senza estensione)
        """
        # Crea directory se non esiste
        dir_path = os.path.dirname(path) if os.path.dirname(path) else '.'
        os.makedirs(dir_path, exist_ok=True)
        
        # Salva modello Keras
        self.policy_net.save(f"{path}_policy.keras")
        self.target_net.save(f"{path}_target.keras")
        
        # Salva stato agente
        agent_state = {
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
            'state_shape': self.state_shape,
            'action_size': self.action_size,
            'use_dueling': self.use_dueling
        }
        with open(f"{path}_state.pkl", 'wb') as f:
            pickle.dump(agent_state, f)
        
        # Salva memoria (opzionale, può essere grande)
        with open(f"{path}_memory.pkl", 'wb') as f:
            pickle.dump(self.memory, f)
        
        print(f"Model saved to {path}")
        
    def load(self, path):
        """
        Carica il modello e lo stato dell'agente.
        
        Args:
            path: Path base del modello (senza estensione)
        """
        # Carica modelli Keras
        self.policy_net = load_model(f"{path}_policy.keras")
        self.target_net = load_model(f"{path}_target.keras")
        
        # Carica stato agente
        with open(f"{path}_state.pkl", 'rb') as f:
            agent_state = pickle.load(f)
            
        self.epsilon = agent_state['epsilon']
        self.steps_done = agent_state['steps_done']
        
        # Carica memoria se esiste
        memory_path = f"{path}_memory.pkl"
        if os.path.exists(memory_path):
            with open(memory_path, 'rb') as f:
                self.memory = pickle.load(f)
        
        print(f"Model loaded from {path}")
        print(f"Epsilon: {self.epsilon}, Steps done: {self.steps_done}")
