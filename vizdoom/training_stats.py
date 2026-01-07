"""
Modulo per raccogliere statistiche e generare grafici durante il training.
Fase 1: CNN + Helper Zero-Shot
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
from collections import defaultdict


class TrainingStats:
    """
    Raccoglie statistiche durante il training e genera grafici per il report.
    """
    
    def __init__(self, scenario, output_dir="training_results"):
        """
        Args:
            scenario: Nome dello scenario VizDoom
            output_dir: Directory per salvare i risultati
        """
        self.scenario = scenario
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Crea directory di output
        self.results_path = os.path.join(output_dir, f"{scenario}_{self.timestamp}")
        os.makedirs(self.results_path, exist_ok=True)
        
        # Metriche per episodio
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_outcomes = []  # 1=vittoria, 0=sconfitta
        self.epsilon_values = []
        self.losses = []
        
        # Metriche Helper
        self.helper_calls_per_episode = []
        self.helper_valid_responses = []
        self.helper_response_times = []
        self.helper_hallucinations = []
        
        # Distribuzione azioni
        self.action_counts = defaultdict(int)
        self.helper_action_counts = defaultdict(int)
        self.rl_action_counts = defaultdict(int)
        
        # Metriche per step
        self.step_rewards = []
        
        # Parametri per media mobile
        self.window_size = 50
        
    def log_episode(self, reward, length, victory, epsilon, loss=0, 
                   helper_calls=0, valid_responses=0, hallucinations=0):
        """
        Logga le metriche di un episodio completato.
        """
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.episode_outcomes.append(1 if victory else 0)
        self.epsilon_values.append(epsilon)
        self.losses.append(loss)
        self.helper_calls_per_episode.append(helper_calls)
        self.helper_valid_responses.append(valid_responses)
        self.helper_hallucinations.append(hallucinations)
        
    def log_helper_response(self, response_time, was_valid):
        """Logga una singola risposta dell'Helper."""
        self.helper_response_times.append(response_time)
        
    def log_action(self, action_name, source='rl'):
        """
        Logga un'azione eseguita.
        
        Args:
            action_name: Nome dell'azione
            source: 'helper' o 'rl'
        """
        self.action_counts[action_name] += 1
        if source == 'helper':
            self.helper_action_counts[action_name] += 1
        else:
            self.rl_action_counts[action_name] += 1
            
    def log_step_reward(self, reward):
        """Logga il reward di un singolo step."""
        self.step_rewards.append(reward)
    
    # ===== ALIAS per compatibilità con collect_helper_responses.py =====
    
    def record_episode(self, episode, reward, length, epsilon, win, helper_used=False, loss=None):
        """Alias per log_episode con interfaccia più flessibile."""
        self.log_episode(
            reward=reward,
            length=length,
            victory=win,
            epsilon=epsilon,
            loss=loss if loss else 0,
            helper_calls=1 if helper_used else 0,
            valid_responses=0,
            hallucinations=0
        )
    
    def record_action(self, action_name, source='rl'):
        """Alias per log_action."""
        self.log_action(action_name, source)
    
    def record_helper_response(self, was_valid, response_time):
        """Alias per log_helper_response."""
        self.log_helper_response(response_time, was_valid)
        if was_valid:
            if self.helper_valid_responses:
                self.helper_valid_responses[-1] += 1
            else:
                self.helper_valid_responses.append(1)
        if self.helper_calls_per_episode:
            self.helper_calls_per_episode[-1] += 1
        else:
            self.helper_calls_per_episode.append(1)
    
    def record_hallucination(self):
        """Registra un'allucinazione dell'Helper."""
        if self.helper_hallucinations:
            self.helper_hallucinations[-1] += 1
        else:
            self.helper_hallucinations.append(1)
    
    def record_action_score(self, score):
        """Registra uno score per le azioni dell'Helper."""
        if not hasattr(self, 'action_scores'):
            self.action_scores = []
        self.action_scores.append(score)
    
    def save_stats(self):
        """Alias per save_raw_data e save_summary."""
        self.save_raw_data()
        self.save_summary()
    
    def generate_markdown_report(self):
        """Genera un report in formato Markdown."""
        summary = self.get_summary()
        
        report_path = os.path.join(self.results_path, 'report.md')
        with open(report_path, 'w') as f:
            f.write(f"# Training Report - {self.scenario.replace('_', ' ').title()}\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Overview\n\n")
            f.write(f"| Metric | Value |\n")
            f.write(f"|--------|-------|\n")
            f.write(f"| Total Episodes | {summary.get('total_episodes', 0)} |\n")
            f.write(f"| Average Reward | {summary.get('avg_reward', 0):.2f} ± {summary.get('std_reward', 0):.2f} |\n")
            f.write(f"| Success Rate | {summary.get('success_rate', 0):.1f}% |\n")
            f.write(f"| Avg Episode Length | {summary.get('avg_length', 0):.1f} steps |\n\n")
            
            f.write("## Helper Performance\n\n")
            f.write(f"| Metric | Value |\n")
            f.write(f"|--------|-------|\n")
            f.write(f"| Total Calls | {summary.get('total_helper_calls', 0)} |\n")
            f.write(f"| Valid Response Rate | {summary.get('helper_valid_rate', 0):.1f}% |\n")
            f.write(f"| Hallucinations | {summary.get('total_hallucinations', 0)} |\n")
            f.write(f"| Avg Response Time | {summary.get('avg_response_time', 0):.3f}s |\n\n")
            
            f.write("## Plots\n\n")
            f.write("![Rewards](rewards.png)\n\n")
            f.write("![Success Rate](success_rate.png)\n\n")
            f.write("![Episode Length](episode_length.png)\n\n")
            f.write("![Action Distribution](action_distribution.png)\n\n")
            f.write("![Helper Stats](helper_stats.png)\n\n")
            f.write("![Learning Curve](learning_curve.png)\n\n")
            
        print(f"Markdown report saved to {report_path}")
        return report_path
        
    def _moving_average(self, data, window):
        """Calcola la media mobile."""
        if len(data) < window:
            window = max(1, len(data))
        return np.convolve(data, np.ones(window)/window, mode='valid')
    
    def get_summary(self):
        """Restituisce un dizionario con le statistiche riassuntive."""
        if not self.episode_rewards:
            return {}
            
        total_episodes = len(self.episode_rewards)
        
        summary = {
            'scenario': self.scenario,
            'total_episodes': total_episodes,
            'avg_reward': np.mean(self.episode_rewards),
            'std_reward': np.std(self.episode_rewards),
            'max_reward': np.max(self.episode_rewards),
            'min_reward': np.min(self.episode_rewards),
            'avg_length': np.mean(self.episode_lengths),
            'success_rate': np.mean(self.episode_outcomes) * 100,
            'final_epsilon': self.epsilon_values[-1] if self.epsilon_values else 1.0,
            'avg_loss': np.mean(self.losses) if self.losses else 0,
            'total_helper_calls': sum(self.helper_calls_per_episode),
            'avg_helper_calls_per_episode': np.mean(self.helper_calls_per_episode),
            'helper_valid_rate': (sum(self.helper_valid_responses) / 
                                  max(sum(self.helper_calls_per_episode), 1)) * 100,
            'total_hallucinations': sum(self.helper_hallucinations),
            'avg_response_time': np.mean(self.helper_response_times) if self.helper_response_times else 0,
        }
        
        # Ultimi 100 episodi
        if total_episodes >= 100:
            summary['last_100_avg_reward'] = np.mean(self.episode_rewards[-100:])
            summary['last_100_success_rate'] = np.mean(self.episode_outcomes[-100:]) * 100
            
        return summary
    
    def plot_rewards(self, save=True, show=False):
        """Grafico dei reward per episodio con media mobile."""
        if not self.episode_rewards:
            return
            
        fig, ax = plt.subplots(figsize=(12, 6))
        
        episodes = range(1, len(self.episode_rewards) + 1)
        
        # Reward per episodio
        ax.plot(episodes, self.episode_rewards, alpha=0.3, color='blue', label='Reward per Episode')
        
        # Media mobile
        if len(self.episode_rewards) >= self.window_size:
            ma = self._moving_average(self.episode_rewards, self.window_size)
            ax.plot(range(self.window_size, len(self.episode_rewards) + 1), ma, 
                   color='red', linewidth=2, label=f'Moving Average ({self.window_size} ep)')
        
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Reward', fontsize=12)
        ax.set_title(f'Training Rewards - {self.scenario.replace("_", " ").title()}', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(self.results_path, 'rewards.png'), dpi=150)
        if show:
            plt.show()
        plt.close()
        
    def plot_success_rate(self, save=True, show=False):
        """Grafico del success rate nel tempo."""
        if not self.episode_outcomes:
            return
            
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Calcola success rate cumulativo
        cumulative_wins = np.cumsum(self.episode_outcomes)
        episodes = np.arange(1, len(self.episode_outcomes) + 1)
        success_rate = (cumulative_wins / episodes) * 100
        
        ax.plot(episodes, success_rate, color='green', linewidth=2)
        
        # Media mobile del success rate
        if len(self.episode_outcomes) >= self.window_size:
            ma = self._moving_average(self.episode_outcomes, self.window_size) * 100
            ax.plot(range(self.window_size, len(self.episode_outcomes) + 1), ma, 
                   color='orange', linewidth=2, linestyle='--', 
                   label=f'Moving Average ({self.window_size} ep)')
        
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Success Rate (%)', fontsize=12)
        ax.set_title(f'Success Rate Over Time - {self.scenario.replace("_", " ").title()}', fontsize=14)
        ax.set_ylim(0, 100)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(self.results_path, 'success_rate.png'), dpi=150)
        if show:
            plt.show()
        plt.close()
        
    def plot_episode_length(self, save=True, show=False):
        """Grafico della lunghezza degli episodi."""
        if not self.episode_lengths:
            return
            
        fig, ax = plt.subplots(figsize=(12, 6))
        
        episodes = range(1, len(self.episode_lengths) + 1)
        
        ax.plot(episodes, self.episode_lengths, alpha=0.3, color='purple')
        
        if len(self.episode_lengths) >= self.window_size:
            ma = self._moving_average(self.episode_lengths, self.window_size)
            ax.plot(range(self.window_size, len(self.episode_lengths) + 1), ma, 
                   color='darkviolet', linewidth=2, label=f'Moving Average ({self.window_size} ep)')
        
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Episode Length (steps)', fontsize=12)
        ax.set_title(f'Episode Length - {self.scenario.replace("_", " ").title()}', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(self.results_path, 'episode_length.png'), dpi=150)
        if show:
            plt.show()
        plt.close()
        
    def plot_action_distribution(self, save=True, show=False):
        """Grafico della distribuzione delle azioni (Helper vs RL)."""
        if not self.action_counts:
            return
            
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Tutte le azioni
        actions = list(self.action_counts.keys())
        counts = [self.action_counts[a] for a in actions]
        
        ax1 = axes[0]
        bars1 = ax1.bar(actions, counts, color='steelblue')
        ax1.set_xlabel('Action', fontsize=10)
        ax1.set_ylabel('Count', fontsize=10)
        ax1.set_title('All Actions', fontsize=12)
        ax1.tick_params(axis='x', rotation=45)
        
        # Azioni Helper
        ax2 = axes[1]
        helper_counts = [self.helper_action_counts.get(a, 0) for a in actions]
        bars2 = ax2.bar(actions, helper_counts, color='coral')
        ax2.set_xlabel('Action', fontsize=10)
        ax2.set_ylabel('Count', fontsize=10)
        ax2.set_title('Helper Actions', fontsize=12)
        ax2.tick_params(axis='x', rotation=45)
        
        # Azioni RL
        ax3 = axes[2]
        rl_counts = [self.rl_action_counts.get(a, 0) for a in actions]
        bars3 = ax3.bar(actions, rl_counts, color='seagreen')
        ax3.set_xlabel('Action', fontsize=10)
        ax3.set_ylabel('Count', fontsize=10)
        ax3.set_title('RL Actions', fontsize=12)
        ax3.tick_params(axis='x', rotation=45)
        
        plt.suptitle(f'Action Distribution - {self.scenario.replace("_", " ").title()}', fontsize=14)
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(self.results_path, 'action_distribution.png'), dpi=150)
        if show:
            plt.show()
        plt.close()
        
    def plot_helper_stats(self, save=True, show=False):
        """Grafico delle statistiche dell'Helper."""
        if not self.helper_calls_per_episode:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        episodes = range(1, len(self.helper_calls_per_episode) + 1)
        
        # Helper calls per episodio
        ax1 = axes[0, 0]
        ax1.plot(episodes, self.helper_calls_per_episode, color='teal')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Helper Calls')
        ax1.set_title('Helper Calls per Episode')
        ax1.grid(True, alpha=0.3)
        
        # Valid response rate nel tempo
        ax2 = axes[0, 1]
        cumulative_valid = np.cumsum(self.helper_valid_responses)
        cumulative_calls = np.cumsum(self.helper_calls_per_episode)
        valid_rate = np.divide(cumulative_valid, cumulative_calls, 
                              out=np.zeros_like(cumulative_valid, dtype=float), 
                              where=cumulative_calls != 0) * 100
        ax2.plot(episodes, valid_rate, color='green')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Valid Response Rate (%)')
        ax2.set_title('Cumulative Helper Valid Response Rate')
        ax2.set_ylim(0, 100)
        ax2.grid(True, alpha=0.3)
        
        # Response time distribution
        ax3 = axes[1, 0]
        if self.helper_response_times:
            ax3.hist(self.helper_response_times, bins=30, color='skyblue', edgecolor='black')
            ax3.axvline(np.mean(self.helper_response_times), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(self.helper_response_times):.2f}s')
            ax3.set_xlabel('Response Time (s)')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Helper Response Time Distribution')
            ax3.legend()
        
        # Hallucinations over time
        ax4 = axes[1, 1]
        cumulative_hall = np.cumsum(self.helper_hallucinations)
        ax4.plot(episodes, cumulative_hall, color='red')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Cumulative Hallucinations')
        ax4.set_title('Helper Hallucinations Over Time')
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle(f'Helper Statistics - {self.scenario.replace("_", " ").title()}', fontsize=14)
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(self.results_path, 'helper_stats.png'), dpi=150)
        if show:
            plt.show()
        plt.close()
        
    def plot_learning_curve(self, save=True, show=False):
        """Grafico della learning curve (epsilon e loss)."""
        if not self.epsilon_values:
            return
            
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        episodes = range(1, len(self.epsilon_values) + 1)
        
        # Epsilon decay
        ax1 = axes[0]
        ax1.plot(episodes, self.epsilon_values, color='orange')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Epsilon')
        ax1.set_title('Epsilon Decay')
        ax1.grid(True, alpha=0.3)
        
        # Loss
        ax2 = axes[1]
        if self.losses and any(l > 0 for l in self.losses):
            ax2.plot(episodes, self.losses, alpha=0.3, color='red')
            if len(self.losses) >= self.window_size:
                ma = self._moving_average(self.losses, self.window_size)
                ax2.plot(range(self.window_size, len(self.losses) + 1), ma, 
                        color='darkred', linewidth=2)
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Loss')
            ax2.set_title('Training Loss')
            ax2.grid(True, alpha=0.3)
        
        plt.suptitle(f'Learning Curve - {self.scenario.replace("_", " ").title()}', fontsize=14)
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(self.results_path, 'learning_curve.png'), dpi=150)
        if show:
            plt.show()
        plt.close()
        
    def generate_all_plots(self, show=False):
        """Genera tutti i grafici."""
        print(f"Generating plots in {self.results_path}...")
        self.plot_rewards(save=True, show=show)
        self.plot_success_rate(save=True, show=show)
        self.plot_episode_length(save=True, show=show)
        self.plot_action_distribution(save=True, show=show)
        self.plot_helper_stats(save=True, show=show)
        self.plot_learning_curve(save=True, show=show)
        print("All plots generated.")
        
    def save_summary(self):
        """Salva il summary delle statistiche in un file CSV e TXT."""
        summary = self.get_summary()
        
        # Salva come CSV
        summary_df = pd.DataFrame([summary])
        summary_df.to_csv(os.path.join(self.results_path, 'summary.csv'), index=False)
        
        # Salva come TXT leggibile
        with open(os.path.join(self.results_path, 'summary.txt'), 'w') as f:
            f.write(f"=" * 60 + "\n")
            f.write(f"TRAINING SUMMARY - {self.scenario.upper()}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"=" * 60 + "\n\n")
            
            f.write("GENERAL STATISTICS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total Episodes:          {summary.get('total_episodes', 0)}\n")
            f.write(f"Average Reward:          {summary.get('avg_reward', 0):.2f} ± {summary.get('std_reward', 0):.2f}\n")
            f.write(f"Max Reward:              {summary.get('max_reward', 0):.2f}\n")
            f.write(f"Min Reward:              {summary.get('min_reward', 0):.2f}\n")
            f.write(f"Average Episode Length:  {summary.get('avg_length', 0):.1f} steps\n")
            f.write(f"Success Rate:            {summary.get('success_rate', 0):.1f}%\n")
            f.write(f"Final Epsilon:           {summary.get('final_epsilon', 0):.4f}\n")
            f.write(f"Average Loss:            {summary.get('avg_loss', 0):.6f}\n\n")
            
            if 'last_100_avg_reward' in summary:
                f.write("LAST 100 EPISODES\n")
                f.write("-" * 40 + "\n")
                f.write(f"Average Reward:          {summary.get('last_100_avg_reward', 0):.2f}\n")
                f.write(f"Success Rate:            {summary.get('last_100_success_rate', 0):.1f}%\n\n")
            
            f.write("HELPER STATISTICS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total Helper Calls:      {summary.get('total_helper_calls', 0)}\n")
            f.write(f"Avg Calls per Episode:   {summary.get('avg_helper_calls_per_episode', 0):.2f}\n")
            f.write(f"Valid Response Rate:     {summary.get('helper_valid_rate', 0):.1f}%\n")
            f.write(f"Total Hallucinations:    {summary.get('total_hallucinations', 0)}\n")
            f.write(f"Avg Response Time:       {summary.get('avg_response_time', 0):.3f}s\n\n")
            
            f.write("ACTION DISTRIBUTION\n")
            f.write("-" * 40 + "\n")
            total_actions = sum(self.action_counts.values())
            for action, count in sorted(self.action_counts.items(), key=lambda x: -x[1]):
                pct = (count / total_actions * 100) if total_actions > 0 else 0
                f.write(f"{action:20s}  {count:6d}  ({pct:5.1f}%)\n")
        
        print(f"Summary saved to {self.results_path}")
        
    def save_raw_data(self):
        """Salva i dati grezzi per analisi successive."""
        # Determina la lunghezza comune (numero di episodi)
        num_episodes = len(self.episode_rewards)
        
        # Funzione per allineare lunghezze degli array
        def pad_to_length(arr, target_len, fill_value=None):
            if len(arr) < target_len:
                return list(arr) + [fill_value] * (target_len - len(arr))
            return list(arr)[:target_len]
        
        # Episode data con padding
        episode_data = pd.DataFrame({
            'episode': range(1, num_episodes + 1),
            'reward': pad_to_length(self.episode_rewards, num_episodes, 0),
            'length': pad_to_length(self.episode_lengths, num_episodes, 0),
            'victory': pad_to_length(self.episode_outcomes, num_episodes, False),
            'epsilon': pad_to_length(self.epsilon_values, num_episodes, 0.0),
            'loss': pad_to_length(self.losses, num_episodes, None),
            'helper_calls': pad_to_length(self.helper_calls_per_episode, num_episodes, 0),
            'valid_responses': pad_to_length(self.helper_valid_responses, num_episodes, 0),
            'hallucinations': pad_to_length(self.helper_hallucinations, num_episodes, 0)
        })
        episode_data.to_csv(os.path.join(self.results_path, 'episode_data.csv'), index=False)
        
        # Action data
        action_data = []
        for action in self.action_counts:
            action_data.append({
                'action': action,
                'total_count': self.action_counts[action],
                'helper_count': self.helper_action_counts.get(action, 0),
                'rl_count': self.rl_action_counts.get(action, 0)
            })
        pd.DataFrame(action_data).to_csv(os.path.join(self.results_path, 'action_data.csv'), index=False)
        
        print(f"Raw data saved to {self.results_path}")
        
    def finalize(self, show_plots=False):
        """
        Finalizza la raccolta: genera tutti i grafici e salva i dati.
        Chiamare alla fine del training.
        """
        print(f"\n{'='*60}")
        print("FINALIZING TRAINING STATISTICS")
        print(f"{'='*60}\n")
        
        self.generate_all_plots(show=show_plots)
        self.save_summary()
        self.save_raw_data()
        
        # Stampa summary
        summary = self.get_summary()
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"Scenario:           {summary.get('scenario', 'N/A')}")
        print(f"Total Episodes:     {summary.get('total_episodes', 0)}")
        print(f"Average Reward:     {summary.get('avg_reward', 0):.2f}")
        print(f"Success Rate:       {summary.get('success_rate', 0):.1f}%")
        print(f"Helper Valid Rate:  {summary.get('helper_valid_rate', 0):.1f}%")
        print(f"Hallucinations:     {summary.get('total_hallucinations', 0)}")
        print(f"\nResults saved to: {self.results_path}")
        print("="*60 + "\n")
        
        return self.results_path


class TrainingVisualizer:
    """
    Wrapper per generare grafici da TrainingStats.
    Fornisce un'interfaccia più pulita per la generazione di grafici.
    """
    
    def __init__(self, training_stats):
        """
        Args:
            training_stats: Istanza di TrainingStats
        """
        self.stats = training_stats
    '''
    def generate_all_plots(self, show=False):
        """Genera tutti i grafici disponibili."""
        self.stats.generate_all_plots(show=show)
    
    def plot_rewards(self, show=False):
        """Grafico dei reward."""
        self.stats.plot_rewards(save=True, show=show)
    
    def plot_success_rate(self, show=False):
        """Grafico del success rate."""
        self.stats.plot_success_rate(save=True, show=show)
    
    def plot_episode_length(self, show=False):
        """Grafico della lunghezza degli episodi."""
        self.stats.plot_episode_length(save=True, show=show)
    
    def plot_action_distribution(self, show=False):
        """Grafico della distribuzione delle azioni."""
        self.stats.plot_action_distribution(save=True, show=show)
    
    def plot_helper_stats(self, show=False):
        """Grafici delle statistiche Helper."""
        self.stats.plot_helper_stats(save=True, show=show)
    
    def plot_learning_curve(self, show=False):
        """Grafico della learning curve."""
        self.stats.plot_learning_curve(save=True, show=show)
    '''


# Test
if __name__ == "__main__":
    # Test con dati simulati
    stats = TrainingStats("deadly_corridor", output_dir="test_results")
    
    import random
    for ep in range(100):
        reward = random.uniform(-50, 100)
        length = random.randint(50, 500)
        victory = random.random() > 0.6
        epsilon = max(0.01, 1.0 * (0.995 ** ep))
        loss = random.uniform(0.001, 0.1)
        helper_calls = random.randint(1, 5)
        valid = random.randint(0, helper_calls)
        hall = helper_calls - valid
        
        stats.log_episode(reward, length, victory, epsilon, loss,
                         helper_calls, valid, hall)
        
        for _ in range(helper_calls):
            stats.log_helper_response(random.uniform(0.5, 3.0), random.random() > 0.3)
        
        for _ in range(length):
            action = random.choice(['MOVE_FORWARD', 'MOVE_LEFT', 'MOVE_RIGHT', 
                                   'ATTACK', 'TURN_LEFT', 'TURN_RIGHT'])
            source = 'helper' if random.random() > 0.7 else 'rl'
            stats.log_action(action, source)
    
    stats.finalize(show_plots=False)
