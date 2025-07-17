import multiprocessing
import json
import numpy as np
import wandb
from datetime import datetime
import os
from typing import Dict, List, Tuple
from dataclasses import dataclass

from training_pipeline import train_ppo_agent, make_env
from quadrotor_env import QuadPole2D

@dataclass
class PolicyProfile:
    """Data class to store policy behavior profile"""
    name: str
    pos_cost_multiplier: float
    avg_episode_reward: float
    waypoint_capture_rate: float
    pendulum_drop_rate: float
    avg_time_to_capture: float
    avg_time_balanced: float
    meets_criteria: bool

class RewardTuningSweep:
    """Class to manage reward tuning sweep and policy classification"""
    
    def __init__(self, base_config_path: str, use_wandb: bool = True):
        # Load base configuration
        with open(base_config_path, 'r') as file:
            self.base_config = json.load(file)
        
        self.use_wandb = use_wandb
        self.results = []
        
        # Create directories for results
        self.sweep_dir = f"./sweep_results/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.sweep_dir, exist_ok=True)
        os.makedirs(f"{self.sweep_dir}/models", exist_ok=True)
        
    def generate_sweep_configs(self) -> List[Dict]:
        """Generate list of configs with different pos_cost_multiplier values"""
        # Define sweep range for position cost multiplier
        # Lower values = less penalty for being away from goal = more aggressive waypoint capture
        # Higher values = more penalty for being away from goal = more cautious, prioritize balance
        pos_cost_multipliers = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0]
        
        configs = []
        for multiplier in pos_cost_multipliers:
            config = self.base_config.copy()
            config['pos_cost_multiplier'] = multiplier
            config['sweep_id'] = f"pos_{multiplier}"
            configs.append(config)
            
        return configs
    
    def evaluate_policy(self, model, config: Dict, n_episodes: int = 20) -> Dict:
        """Evaluate a trained policy to extract behavioral metrics"""
        eval_env = make_env(config, "eval", manual_goal_position=None)
        
        episode_rewards = []
        waypoint_captures = []
        pendulum_drops = []
        times_to_capture = []
        times_balanced = []
        
        for episode in range(n_episodes):
            obs, info = eval_env.reset()
            episode_reward = 0
            episode_length = 0
            pendulum_dropped = False
            waypoint_captured = False
            time_to_capture = None
            episode_time_balanced = 0
            
            # Track initial goal position
            initial_goal = obs[-2:]
            goal_changed = False
            
            for step in range(500):  # Max episode length
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                
                episode_reward += reward
                episode_length += 1
                
                # Check if goal position changed (for dynamic goals)
                current_goal = obs[-2:]
                if not np.allclose(current_goal, initial_goal, atol=0.1) and not goal_changed:
                    goal_changed = True
                    initial_goal = current_goal
                
                # Check waypoint capture (close to goal position)
                position = obs[:2]
                goal_position = obs[-2:]
                distance_to_goal = np.linalg.norm(position - goal_position)
                
                if distance_to_goal < 0.3 and not waypoint_captured:  # Within capture radius
                    waypoint_captured = True
                    time_to_capture = step * eval_env.env.env.timestep
                
                # Check pendulum state
                s_phi, c_phi = obs[7], obs[8]
                phi = np.arctan2(s_phi, c_phi)
                
                # Pendulum is "up" if cos(phi) < -0.8 (phi close to pi)
                if c_phi < -0.8:
                    episode_time_balanced += eval_env.env.env.timestep
                else:
                    # Check if pendulum "dropped" (swung significantly from upright)
                    if not pendulum_dropped and c_phi > -0.5:  # Significant deviation from upright
                        pendulum_dropped = True
                
                if terminated or truncated:
                    break
            
            # Store episode metrics
            episode_rewards.append(episode_reward)
            waypoint_captures.append(1 if waypoint_captured else 0)
            pendulum_drops.append(1 if pendulum_dropped else 0)
            times_to_capture.append(time_to_capture if time_to_capture else episode_length * eval_env.env.env.timestep)
            times_balanced.append(episode_time_balanced)
        
        # Calculate aggregate metrics
        metrics = {
            'avg_episode_reward': np.mean(episode_rewards),
            'std_episode_reward': np.std(episode_rewards),
            'waypoint_capture_rate': np.mean(waypoint_captures),
            'pendulum_drop_rate': np.mean(pendulum_drops),
            'avg_time_to_capture': np.mean(times_to_capture),
            'avg_time_balanced': np.mean(times_balanced),
            'episodes_evaluated': n_episodes
        }
        
        eval_env.close()
        return metrics
    
    def classify_policy(self, metrics: Dict, pos_cost_multiplier: float) -> Tuple[str, bool]:
        """Classify policy based on behavioral metrics"""
        capture_rate = metrics['waypoint_capture_rate']
        drop_rate = metrics['pendulum_drop_rate']
        time_to_capture = metrics['avg_time_to_capture']
        
        # Define thresholds for classification
        # Baseline: Captures waypoints relatively quickly, pendulum drops ~once per 5 episodes (20% drop rate)
        if (capture_rate >= 0.8 and drop_rate <= 0.25 and time_to_capture <= 8.0):
            return "Baseline", True
            
        # Aggressive: Captures waypoints quickly, pendulum can drop more frequently
        elif (capture_rate >= 0.85 and time_to_capture <= 6.0 and drop_rate <= 0.6):
            return "Aggressive", True
            
        # Cautious: Never drops pendulum, may be slower to capture
        elif (drop_rate <= 0.05 and capture_rate >= 0.7):
            return "Cautious", True
            
        else:
            return "Unclassified", False
    
    def run_sweep(self, n_episodes_eval: int = 20):
        """Run the complete reward tuning sweep"""
        print("Starting reward tuning sweep...")
        
        if self.use_wandb:
            sweep_run = wandb.init(
                project='dl-project-sweep',
                name=f"reward_sweep_{datetime.now().strftime('%m%d_%H%M')}",
                job_type="sweep"
            )
        
        configs = self.generate_sweep_configs()
        
        for i, config in enumerate(configs):
            print(f"\n--- Training {i+1}/{len(configs)}: pos_cost_multiplier = {config['pos_cost_multiplier']} ---")
            
            # Train the model
            try:
                model = train_ppo_agent(config, "train", use_wandb=False)  # Disable wandb for individual runs
                
                # Evaluate the trained model
                print("Evaluating policy...")
                metrics = self.evaluate_policy(model, config, n_episodes_eval)
                
                # Classify the policy
                policy_type, meets_criteria = self.classify_policy(metrics, config['pos_cost_multiplier'])
                
                # Create policy profile
                profile = PolicyProfile(
                    name=f"pos_{config['pos_cost_multiplier']}",
                    pos_cost_multiplier=config['pos_cost_multiplier'],
                    avg_episode_reward=metrics['avg_episode_reward'],
                    waypoint_capture_rate=metrics['waypoint_capture_rate'],
                    pendulum_drop_rate=metrics['pendulum_drop_rate'],
                    avg_time_to_capture=metrics['avg_time_to_capture'],
                    avg_time_balanced=metrics['avg_time_balanced'],
                    meets_criteria=meets_criteria
                )
                
                self.results.append(profile)
                
                # Log to wandb if enabled
                if self.use_wandb:
                    wandb.log({
                        "pos_cost_multiplier": config['pos_cost_multiplier'],
                        "policy_type": policy_type,
                        "meets_criteria": meets_criteria,
                        **metrics
                    })
                
                # Save model if it meets criteria
                if meets_criteria:
                    model_path = f"{self.sweep_dir}/models/{policy_type.lower()}_{config['pos_cost_multiplier']}"
                    model.save(model_path)
                    print(f"✓ Saved {policy_type} policy to {model_path}")
                
                print(f"Policy Type: {policy_type}")
                print(f"Meets Criteria: {meets_criteria}")
                print(f"Capture Rate: {metrics['waypoint_capture_rate']:.2f}")
                print(f"Drop Rate: {metrics['pendulum_drop_rate']:.2f}")
                print(f"Avg Time to Capture: {metrics['avg_time_to_capture']:.2f}s")
                
            except Exception as e:
                print(f"Error training/evaluating pos_cost_multiplier {config['pos_cost_multiplier']}: {e}")
                continue
        
        # Save results summary
        self.save_results_summary()
        
        if self.use_wandb:
            sweep_run.finish()
        
        print(f"\nSweep completed! Results saved to {self.sweep_dir}")
        return self.results
    
    def save_results_summary(self):
        """Save summary of all results"""
        summary = {
            'sweep_timestamp': datetime.now().isoformat(),
            'total_configs_tested': len(self.results),
            'policies_found': {},
            'detailed_results': []
        }
        
        # Group results by policy type
        for result in self.results:
            metrics, meets_criteria = self.classify_policy({
                'waypoint_capture_rate': result.waypoint_capture_rate,
                'pendulum_drop_rate': result.pendulum_drop_rate,
                'avg_time_to_capture': result.avg_time_to_capture
            }, result.pos_cost_multiplier)
            
            if meets_criteria:
                if metrics not in summary['policies_found']:
                    summary['policies_found'][metrics] = []
                summary['policies_found'][metrics].append({
                    'pos_cost_multiplier': result.pos_cost_multiplier,
                    'capture_rate': result.waypoint_capture_rate,
                    'drop_rate': result.pendulum_drop_rate,
                    'time_to_capture': result.avg_time_to_capture
                })
            
            # Add to detailed results
            summary['detailed_results'].append({
                'pos_cost_multiplier': result.pos_cost_multiplier,
                'policy_type': metrics,
                'meets_criteria': meets_criteria,
                'avg_episode_reward': result.avg_episode_reward,
                'waypoint_capture_rate': result.waypoint_capture_rate,
                'pendulum_drop_rate': result.pendulum_drop_rate,
                'avg_time_to_capture': result.avg_time_to_capture,
                'avg_time_balanced': result.avg_time_balanced
            })
        
        # Save to JSON
        with open(f"{self.sweep_dir}/sweep_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        print("\n" + "="*50)
        print("SWEEP SUMMARY")
        print("="*50)
        for policy_type, policies in summary['policies_found'].items():
            print(f"\n{policy_type} Policies Found: {len(policies)}")
            for policy in policies:
                print(f"  pos_cost_multiplier: {policy['pos_cost_multiplier']}")
                print(f"    Capture Rate: {policy['capture_rate']:.2f}")
                print(f"    Drop Rate: {policy['drop_rate']:.2f}")
                print(f"    Time to Capture: {policy['time_to_capture']:.2f}s")


def main():
    """Main function to run the reward tuning sweep"""
    # Configuration
    base_config_path = './configs/config_v5.json'
    use_wandb = False  # Set to True if you want to log to wandb
    n_episodes_eval = 25  # Number of episodes for evaluation
    
    # Create and run sweep
    sweep = RewardTuningSweep(base_config_path, use_wandb)
    results = sweep.run_sweep(n_episodes_eval)
    
    print("\nReward tuning sweep completed!")
    print(f"Results saved to: {sweep.sweep_dir}")

if __name__ == "__main__":
    main()
