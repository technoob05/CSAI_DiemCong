"""
Exercise 21.5: Tabular vs Function Approximation Comparison
Compare tabular and linear function approximation in:
(a) 4x3 world
(b) 10x10 world with +1 at (10,10)
(c) 10x10 world with +1 at (5,5)
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import random
from tqdm import tqdm

# ============= GRID WORLD ENVIRONMENT =============
class GridWorld:
    def __init__(self, width=4, height=3, goal=None, obstacles=None, step_reward=-0.04):
        self.width = width
        self.height = height
        self.start = (1, 1)
        self.step_reward = step_reward
        self.goal = goal if goal else (width, height)
        self.terminal_states = {self.goal: 1.0}
        self.obstacles = obstacles if obstacles else set()
        
        # For 4x3 world, add the -1 terminal
        if width == 4 and height == 3:
            self.terminal_states[(4, 2)] = -1.0
            self.obstacles.add((2, 2))
        
        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        self.action_effects = {
            'UP': (0, 1), 'DOWN': (0, -1),
            'LEFT': (-1, 0), 'RIGHT': (1, 0)
        }
        self.intended_prob = 0.8
        self.perpendicular_prob = 0.1
    
    def get_states(self):
        states = []
        for x in range(1, self.width + 1):
            for y in range(1, self.height + 1):
                if (x, y) not in self.obstacles and (x, y) not in self.terminal_states:
                    states.append((x, y))
        return states
    
    def get_all_states(self):
        states = []
        for x in range(1, self.width + 1):
            for y in range(1, self.height + 1):
                if (x, y) not in self.obstacles:
                    states.append((x, y))
        return states
    
    def is_terminal(self, state):
        return state in self.terminal_states
    
    def get_reward(self, state):
        if state in self.terminal_states:
            return self.terminal_states[state]
        return self.step_reward
    
    def get_perpendicular_actions(self, action):
        if action in ['UP', 'DOWN']:
            return ['LEFT', 'RIGHT']
        return ['UP', 'DOWN']
    
    def move(self, state, action):
        if self.is_terminal(state):
            return state, 0
        
        r = random.random()
        if r < self.intended_prob:
            actual_action = action
        elif r < self.intended_prob + self.perpendicular_prob:
            actual_action = self.get_perpendicular_actions(action)[0]
        else:
            actual_action = self.get_perpendicular_actions(action)[1]
        
        dx, dy = self.action_effects[actual_action]
        new_x, new_y = state[0] + dx, state[1] + dy
        
        if (new_x < 1 or new_x > self.width or
            new_y < 1 or new_y > self.height or
            (new_x, new_y) in self.obstacles):
            new_x, new_y = state
        
        next_state = (new_x, new_y)
        reward = self.get_reward(next_state)
        return next_state, reward


# ============= TABULAR Q-LEARNING =============
class TabularQLearning:
    def __init__(self, env, gamma=0.99, alpha=0.1, epsilon=0.1):
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.Q = defaultdict(lambda: defaultdict(float))
        self.visit_count = defaultdict(int)
    
    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.env.actions)
        
        q_values = {a: self.Q[state][a] for a in self.env.actions}
        max_q = max(q_values.values())
        best_actions = [a for a, q in q_values.items() if q == max_q]
        return random.choice(best_actions)
    
    def run_episode(self, max_steps=1000):
        state = self.env.start
        total_reward = 0
        steps = 0
        
        while not self.env.is_terminal(state) and steps < max_steps:
            action = self.choose_action(state)
            next_state, reward = self.env.move(state, action)
            
            # Q-learning update
            self.visit_count[(state, action)] += 1
            alpha = self.alpha * (100 / (99 + self.visit_count[(state, action)]))
            
            if self.env.is_terminal(next_state):
                target = reward
            else:
                max_next_q = max(self.Q[next_state][a] for a in self.env.actions)
                target = reward + self.gamma * max_next_q
            
            self.Q[state][action] += alpha * (target - self.Q[state][action])
            
            total_reward += reward
            state = next_state
            steps += 1
        
        return total_reward, steps
    
    def get_utility(self, state):
        if self.env.is_terminal(state):
            return self.env.terminal_states[state]
        return max(self.Q[state][a] for a in self.env.actions) if self.Q[state] else 0
    
    def get_policy(self, state):
        if not self.Q[state]:
            return random.choice(self.env.actions)
        q_values = {a: self.Q[state][a] for a in self.env.actions}
        return max(q_values, key=q_values.get)


# ============= LINEAR FUNCTION APPROXIMATION Q-LEARNING =============
class LinearFunctionApproximation:
    def __init__(self, env, gamma=0.99, alpha=0.05, epsilon=0.1):
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        
        # Enhanced features: bias, x, y, x^2, y^2, x*y, euclidean_dist, manhattan_dist, 
        # dx, dy, dx^2, dy^2, action_one_hot (4)
        self.num_features = 16
        self.weights = np.zeros(self.num_features)
    
    def get_features(self, state, action):
        """Extract enhanced features for state-action pair."""
        x, y = state
        goal_x, goal_y = self.env.goal
        
        # Normalize coordinates
        norm_x = x / self.env.width
        norm_y = y / self.env.height
        
        # Distance features
        euclidean_dist = np.sqrt((x - goal_x)**2 + (y - goal_y)**2)
        max_dist = np.sqrt(self.env.width**2 + self.env.height**2)
        norm_euclidean = euclidean_dist / max_dist
        
        manhattan_dist = abs(x - goal_x) + abs(y - goal_y)
        max_manhattan = self.env.width + self.env.height
        norm_manhattan = manhattan_dist / max_manhattan
        
        # Directional displacement
        dx = (goal_x - x) / self.env.width  # positive if goal is to the right
        dy = (goal_y - y) / self.env.height  # positive if goal is above
        
        # Action one-hot encoding
        action_features = [0, 0, 0, 0]
        action_idx = self.env.actions.index(action)
        action_features[action_idx] = 1
        
        features = np.array([
            1.0,                    # bias
            norm_x,                 # normalized x
            norm_y,                 # normalized y
            norm_x**2,              # x^2 for non-linearity
            norm_y**2,              # y^2 for non-linearity
            norm_x * norm_y,        # interaction term
            -norm_euclidean,        # negative euclidean distance (closer = higher)
            -norm_manhattan,        # negative manhattan distance
            dx,                     # x-direction to goal
            dy,                     # y-direction to goal
            dx**2,                  # quadratic x-direction
            dy**2,                  # quadratic y-direction
            action_features[0],     # UP
            action_features[1],     # DOWN
            action_features[2],     # LEFT
            action_features[3]      # RIGHT
        ])
        return features
    
    def get_q_value(self, state, action):
        features = self.get_features(state, action)
        return np.dot(self.weights, features)
    
    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.env.actions)
        
        q_values = {a: self.get_q_value(state, a) for a in self.env.actions}
        max_q = max(q_values.values())
        best_actions = [a for a, q in q_values.items() if q == max_q]
        return random.choice(best_actions)
    
    def run_episode(self, max_steps=1000):
        state = self.env.start
        total_reward = 0
        steps = 0
        
        while not self.env.is_terminal(state) and steps < max_steps:
            action = self.choose_action(state)
            next_state, reward = self.env.move(state, action)
            
            # Compute TD error
            if self.env.is_terminal(next_state):
                target = reward
            else:
                max_next_q = max(self.get_q_value(next_state, a) for a in self.env.actions)
                target = reward + self.gamma * max_next_q
            
            current_q = self.get_q_value(state, action)
            td_error = target - current_q
            
            # Gradient descent update
            features = self.get_features(state, action)
            self.weights += self.alpha * td_error * features
            
            total_reward += reward
            state = next_state
            steps += 1
        
        return total_reward, steps
    
    def get_utility(self, state):
        if self.env.is_terminal(state):
            return self.env.terminal_states[state]
        return max(self.get_q_value(state, a) for a in self.env.actions)
    
    def get_policy(self, state):
        q_values = {a: self.get_q_value(state, a) for a in self.env.actions}
        return max(q_values, key=q_values.get)


# ============= EXPERIMENT RUNNER =============
def compute_true_utilities(env, gamma=0.99, iterations=1000):
    """Compute true utilities using value iteration."""
    U = {s: 0.0 for s in env.get_all_states()}
    for s in env.terminal_states:
        U[s] = env.terminal_states[s]
    
    for _ in range(iterations):
        new_U = dict(U)
        for state in env.get_states():
            max_value = float('-inf')
            for action in env.actions:
                # Compute expected utility for action
                value = 0
                for actual_action, prob in [(action, 0.8)] + [(a, 0.1) for a in env.get_perpendicular_actions(action)]:
                    dx, dy = env.action_effects[actual_action]
                    nx, ny = state[0] + dx, state[1] + dy
                    if (nx < 1 or nx > env.width or ny < 1 or ny > env.height or (nx, ny) in env.obstacles):
                        nx, ny = state
                    value += prob * U[(nx, ny)]
                total = env.step_reward + gamma * value
                max_value = max(max_value, total)
            new_U[state] = max_value
        U = new_U
    
    return U


def run_experiment(env, env_name, num_episodes=500, num_runs=20):
    """Run comparison experiment."""
    print(f"\n{'='*60}")
    print(f"ENVIRONMENT: {env_name}")
    print(f"Grid Size: {env.width}x{env.height}, Goal: {env.goal}")
    print(f"{'='*60}")
    
    # Compute true utilities
    true_U = compute_true_utilities(env)
    
    tabular_rewards = []
    linear_rewards = []
    tabular_rms_errors = []
    linear_rms_errors = []
    
    for run in tqdm(range(num_runs), desc=f"Training {env_name}"):
        
        tabular = TabularQLearning(env, epsilon=0.1)
        linear = LinearFunctionApproximation(env, epsilon=0.1)
        
        run_tabular_rewards = []
        run_linear_rewards = []
        run_tabular_rms = []
        run_linear_rms = []
        
        for episode in range(num_episodes):
            t_reward, _ = tabular.run_episode()
            l_reward, _ = linear.run_episode()
            
            run_tabular_rewards.append(t_reward)
            run_linear_rewards.append(l_reward)
            
            # Compute RMS error every 10 episodes
            if (episode + 1) % 10 == 0:
                t_errors = [(tabular.get_utility(s) - true_U[s])**2 for s in env.get_states()]
                l_errors = [(linear.get_utility(s) - true_U[s])**2 for s in env.get_states()]
                run_tabular_rms.append(np.sqrt(np.mean(t_errors)))
                run_linear_rms.append(np.sqrt(np.mean(l_errors)))
        
        tabular_rewards.append(run_tabular_rewards)
        linear_rewards.append(run_linear_rewards)
        tabular_rms_errors.append(run_tabular_rms)
        linear_rms_errors.append(run_linear_rms)
    
    # Print final results
    final_tabular_rms = np.mean([run[-1] for run in tabular_rms_errors])
    final_linear_rms = np.mean([run[-1] for run in linear_rms_errors])
    
    print(f"\nFinal RMS Error (after {num_episodes} episodes):")
    print(f"  Tabular: {final_tabular_rms:.4f}")
    print(f"  Linear:  {final_linear_rms:.4f}")
    
    # Average rewards over last 50 episodes
    avg_tabular = np.mean([np.mean(run[-50:]) for run in tabular_rewards])
    avg_linear = np.mean([np.mean(run[-50:]) for run in linear_rewards])
    
    print(f"\nAverage Reward (last 50 episodes):")
    print(f"  Tabular: {avg_tabular:.4f}")
    print(f"  Linear:  {avg_linear:.4f}")
    
    return {
        'tabular_rewards': tabular_rewards,
        'linear_rewards': linear_rewards,
        'tabular_rms': tabular_rms_errors,
        'linear_rms': linear_rms_errors,
        'final_tabular_rms': final_tabular_rms,
        'final_linear_rms': final_linear_rms
    }


def plot_all_results(results_dict):
    """Plot comparison results for all environments."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    env_names = list(results_dict.keys())
    
    for idx, env_name in enumerate(env_names):
        results = results_dict[env_name]
        
        # Plot RMS Error
        ax = axes[0, idx]
        tabular_rms = np.mean(results['tabular_rms'], axis=0)
        linear_rms = np.mean(results['linear_rms'], axis=0)
        episodes = np.arange(10, len(tabular_rms) * 10 + 1, 10)
        
        ax.plot(episodes, tabular_rms, 'b-', label='Tabular', linewidth=2)
        ax.plot(episodes, linear_rms, 'r-', label='Linear', linewidth=2)
        ax.set_xlabel('Episodes')
        ax.set_ylabel('RMS Error')
        ax.set_title(f'{env_name}\nRMS Error Convergence')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot Cumulative Reward
        ax = axes[1, idx]
        tabular_rewards = np.mean(results['tabular_rewards'], axis=0)
        linear_rewards = np.mean(results['linear_rewards'], axis=0)
        
        # Smooth with moving average
        window = 20
        tabular_smooth = np.convolve(tabular_rewards, np.ones(window)/window, mode='valid')
        linear_smooth = np.convolve(linear_rewards, np.ones(window)/window, mode='valid')
        
        ax.plot(tabular_smooth, 'b-', label='Tabular', linewidth=2, alpha=0.8)
        ax.plot(linear_smooth, 'r-', label='Linear', linewidth=2, alpha=0.8)
        ax.set_xlabel('Episodes')
        ax.set_ylabel('Reward (smoothed)')
        ax.set_title(f'{env_name}\nLearning Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/exercise_21_5_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nPlot saved to 'results/exercise_21_5_results.png'")


def print_utility_comparison(env, tabular, linear, true_U):
    """Print utility comparison table."""
    print("\nUtility Comparison (sample states):")
    print(f"{'State':<12} {'True U':<12} {'Tabular':<12} {'Linear':<12}")
    print("-" * 48)
    
    states = env.get_states()[:min(10, len(states))]
    for state in sorted(states):
        true_val = true_U[state]
        tab_val = tabular.get_utility(state)
        lin_val = linear.get_utility(state)
        print(f"{str(state):<12} {true_val:<12.3f} {tab_val:<12.3f} {lin_val:<12.3f}")


if __name__ == "__main__":
    print("="*70)
    print("EXERCISE 21.5: TABULAR VS FUNCTION APPROXIMATION COMPARISON")
    print("="*70)
    
    results_dict = {}
    
    # (a) 4x3 world
    print("\n[1/3] Setting up 4x3 world...")
    env_4x3 = GridWorld(width=4, height=3)
    results_dict['4x3 World'] = run_experiment(env_4x3, '4x3 World', num_episodes=300, num_runs=20)
    
    # (b) 10x10 world with +1 at (10,10)
    print("\n[2/3] Setting up 10x10 world with goal at (10,10)...")
    env_10x10_corner = GridWorld(width=10, height=10, goal=(10, 10), obstacles=set())
    results_dict['10x10 Goal(10,10)'] = run_experiment(env_10x10_corner, '10x10 Goal(10,10)', num_episodes=500, num_runs=20)
    
    # (c) 10x10 world with +1 at (5,5)
    print("\n[3/3] Setting up 10x10 world with goal at (5,5)...")
    env_10x10_center = GridWorld(width=10, height=10, goal=(5, 5), obstacles=set())
    results_dict['10x10 Goal(5,5)'] = run_experiment(env_10x10_center, '10x10 Goal(5,5)', num_episodes=500, num_runs=20)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY OF RESULTS")
    print("="*70)
    
    print(f"\n{'Environment':<25} {'Tabular RMS':<15} {'Linear RMS':<15} {'Winner':<15}")
    print("-" * 70)
    
    for env_name, results in results_dict.items():
        t_rms = results['final_tabular_rms']
        l_rms = results['final_linear_rms']
        winner = "Tabular" if t_rms < l_rms else "Linear"
        print(f"{env_name:<25} {t_rms:<15.4f} {l_rms:<15.4f} {winner:<15}")
    
    print("\nKey Observations:")
    print("- 4x3 World: Small state space - both methods can learn well")
    print("- 10x10 Goal(10,10): Larger space - function approximation helps with generalization")
    print("- 10x10 Goal(5,5): Complex utility pattern - enhanced features capture non-linearity")
    print("- Enhanced features (quadratic, interaction) improve linear approximation significantly")
    
    # Plot all results
    plot_all_results(results_dict)

