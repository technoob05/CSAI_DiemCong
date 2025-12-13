"""
Exercise 21.1: Passive Learning Agent Comparison
Compare DUE, TD, and ADP algorithms in the 4x3 grid world.
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import random

# ============= GRID WORLD ENVIRONMENT =============
class GridWorld:
    def __init__(self, width=4, height=3):
        self.width = width
        self.height = height
        self.start = (1, 1)
        self.terminal_states = {(4, 3): 1.0, (4, 2): -1.0}  # (x, y): reward
        self.obstacle = (2, 2)
        self.step_reward = -0.04
        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        self.action_effects = {
            'UP': (0, 1), 'DOWN': (0, -1), 
            'LEFT': (-1, 0), 'RIGHT': (1, 0)
        }
        # Stochastic transition: 0.8 intended, 0.1 each perpendicular
        self.intended_prob = 0.8
        self.perpendicular_prob = 0.1
        
    def get_states(self):
        """Return all non-terminal, non-obstacle states."""
        states = []
        for x in range(1, self.width + 1):
            for y in range(1, self.height + 1):
                if (x, y) != self.obstacle and (x, y) not in self.terminal_states:
                    states.append((x, y))
        return states
    
    def get_all_states(self):
        """Return all states including terminals."""
        states = []
        for x in range(1, self.width + 1):
            for y in range(1, self.height + 1):
                if (x, y) != self.obstacle:
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
        """Execute action and return (next_state, reward)."""
        if self.is_terminal(state):
            return state, 0
        
        # Stochastic action selection
        r = random.random()
        if r < self.intended_prob:
            actual_action = action
        elif r < self.intended_prob + self.perpendicular_prob:
            actual_action = self.get_perpendicular_actions(action)[0]
        else:
            actual_action = self.get_perpendicular_actions(action)[1]
        
        dx, dy = self.action_effects[actual_action]
        new_x, new_y = state[0] + dx, state[1] + dy
        
        # Check boundaries and obstacles
        if (new_x < 1 or new_x > self.width or 
            new_y < 1 or new_y > self.height or 
            (new_x, new_y) == self.obstacle):
            new_x, new_y = state  # Stay in place
        
        next_state = (new_x, new_y)
        reward = self.get_reward(next_state)
        return next_state, reward
    
    def get_transition_probs(self, state, action):
        """Return dict of {next_state: probability}."""
        if self.is_terminal(state):
            return {state: 1.0}
        
        probs = defaultdict(float)
        for actual_action, prob in [(action, self.intended_prob)] + \
                                   [(a, self.perpendicular_prob) for a in self.get_perpendicular_actions(action)]:
            dx, dy = self.action_effects[actual_action]
            new_x, new_y = state[0] + dx, state[1] + dy
            
            if (new_x < 1 or new_x > self.width or 
                new_y < 1 or new_y > self.height or 
                (new_x, new_y) == self.obstacle):
                new_x, new_y = state
            
            probs[(new_x, new_y)] += prob
        
        return dict(probs)


# ============= OPTIMAL POLICY =============
def get_optimal_policy():
    """Return the optimal policy for the 4x3 world."""
    return {
        (1, 1): 'UP', (1, 2): 'UP', (1, 3): 'RIGHT',
        (2, 1): 'RIGHT', (2, 3): 'RIGHT',
        (3, 1): 'UP', (3, 2): 'UP', (3, 3): 'RIGHT',
        (4, 1): 'UP'  # Added for completeness
    }

def get_random_policy(env):
    """Return a random policy."""
    policy = {}
    for state in env.get_states():
        policy[state] = random.choice(env.actions)
    return policy


# ============= TRUE UTILITY VALUES (for comparison) =============
TRUE_UTILITIES = {
    (1, 1): 0.705, (1, 2): 0.762, (1, 3): 0.812,
    (2, 1): 0.655, (2, 3): 0.868,
    (3, 1): 0.611, (3, 2): 0.660, (3, 3): 0.918,
    (4, 1): 0.388,  # Added
    (4, 3): 1.0, (4, 2): -1.0
}


# ============= DIRECT UTILITY ESTIMATION =============
class DirectUtilityEstimation:
    def __init__(self, env, gamma=1.0):
        self.env = env
        self.gamma = gamma
        self.returns = defaultdict(list)  # state -> list of observed returns
        self.U = defaultdict(float)  # utility estimates
        
    def run_trial(self, policy):
        """Run one trial and update utility estimates."""
        state = self.env.start
        trajectory = [(state, self.env.get_reward(state))]
        
        while not self.env.is_terminal(state):
            action = policy[state]
            next_state, reward = self.env.move(state, action)
            trajectory.append((next_state, reward))
            state = next_state
        
        # Calculate returns for each state in trajectory
        G = 0
        for i in range(len(trajectory) - 1, -1, -1):
            state, reward = trajectory[i]
            G = reward + self.gamma * G
            if state not in self.env.terminal_states:
                self.returns[state].append(G)
                self.U[state] = np.mean(self.returns[state])
        
        return trajectory
    
    def get_rms_error(self):
        """Calculate RMS error compared to true utilities."""
        errors = []
        for state in self.env.get_states():
            if state in self.U and state in TRUE_UTILITIES:
                errors.append((self.U[state] - TRUE_UTILITIES[state]) ** 2)
        return np.sqrt(np.mean(errors)) if errors else 1.0


# ============= TEMPORAL DIFFERENCE LEARNING =============
class TDLearning:
    def __init__(self, env, gamma=1.0, alpha=0.1):
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.U = defaultdict(float)
        # Initialize terminal state utilities
        for term_state, reward in env.terminal_states.items():
            self.U[term_state] = reward
        self.visit_count = defaultdict(int)
        
    def run_trial(self, policy):
        """Run one trial with TD updates."""
        state = self.env.start
        trajectory = []
        
        while not self.env.is_terminal(state):
            action = policy[state]
            next_state, reward = self.env.move(state, action)
            trajectory.append((state, action, reward, next_state))
            
            # TD update
            self.visit_count[state] += 1
            # Decaying learning rate
            alpha = self.alpha * (60 / (59 + self.visit_count[state]))
            
            if self.env.is_terminal(next_state):
                target = reward
            else:
                target = reward + self.gamma * self.U[next_state]
            
            self.U[state] += alpha * (target - self.U[state])
            state = next_state
        
        return trajectory
    
    def get_rms_error(self):
        errors = []
        for state in self.env.get_states():
            if state in TRUE_UTILITIES:
                errors.append((self.U[state] - TRUE_UTILITIES[state]) ** 2)
        return np.sqrt(np.mean(errors)) if errors else 1.0


# ============= ADAPTIVE DYNAMIC PROGRAMMING =============
class AdaptiveDynamicProgramming:
    def __init__(self, env, gamma=1.0):
        self.env = env
        self.gamma = gamma
        self.U = defaultdict(float)
        # Initialize terminal state utilities
        for term_state, reward in env.terminal_states.items():
            self.U[term_state] = reward
        # Model learning
        self.transition_counts = defaultdict(lambda: defaultdict(int))  # (s,a) -> s' -> count
        self.state_action_counts = defaultdict(int)  # (s,a) -> count
        self.reward_sum = defaultdict(float)
        self.reward_count = defaultdict(int)
        
    def run_trial(self, policy):
        """Run one trial, update model, and solve Bellman equations."""
        state = self.env.start
        trajectory = []
        
        while not self.env.is_terminal(state):
            action = policy[state]
            next_state, reward = self.env.move(state, action)
            trajectory.append((state, action, reward, next_state))
            
            # Update model
            self.transition_counts[(state, action)][next_state] += 1
            self.state_action_counts[(state, action)] += 1
            self.reward_sum[state] += self.env.step_reward
            self.reward_count[state] += 1
            
            state = next_state
        
        # Solve Bellman equations with learned model
        self._policy_evaluation(policy)
        
        return trajectory
    
    def _get_learned_transition_probs(self, state, action):
        """Get learned transition probabilities."""
        if self.state_action_counts[(state, action)] == 0:
            return {}
        probs = {}
        total = self.state_action_counts[(state, action)]
        for next_state, count in self.transition_counts[(state, action)].items():
            probs[next_state] = count / total
        return probs
    
    def _policy_evaluation(self, policy, iterations=50):
        """Iterative policy evaluation."""
        states = self.env.get_states()
        
        for _ in range(iterations):
            new_U = defaultdict(float, self.U)
            for state in states:
                if state not in policy:
                    continue
                action = policy[state]
                probs = self._get_learned_transition_probs(state, action)
                if not probs:
                    continue
                
                reward = self.env.step_reward
                expected_utility = sum(p * new_U[s_next] for s_next, p in probs.items())
                new_U[state] = reward + self.gamma * expected_utility
            
            self.U = new_U
        
        # Set terminal state utilities
        for term_state, reward in self.env.terminal_states.items():
            self.U[term_state] = reward
    
    def get_rms_error(self):
        errors = []
        for state in self.env.get_states():
            if state in TRUE_UTILITIES:
                errors.append((self.U[state] - TRUE_UTILITIES[state]) ** 2)
        return np.sqrt(np.mean(errors)) if errors else 1.0


# ============= MAIN EXPERIMENT =============
def run_experiment(num_trials=60, num_runs=10):
    """Run comparison experiment."""
    env = GridWorld()
    optimal_policy = get_optimal_policy()
    
    # Store results
    results = {
        'DUE_optimal': [], 'TD_optimal': [], 'ADP_optimal': [],
        'DUE_random': [], 'TD_random': [], 'ADP_random': []
    }
    
    print("Running experiments...")
    print(f"Number of trials: {num_trials}, Number of runs: {num_runs}")
    
    for run in range(num_runs):
        if (run + 1) % 10 == 0:
            print(f"  Run {run + 1}/{num_runs}")
        
        # Optimal policy experiments
        due_opt = DirectUtilityEstimation(env)
        td_opt = TDLearning(env, alpha=0.1)
        adp_opt = AdaptiveDynamicProgramming(env)
        
        due_errors_opt = []
        td_errors_opt = []
        adp_errors_opt = []
        
        for trial in range(num_trials):
            due_opt.run_trial(optimal_policy)
            td_opt.run_trial(optimal_policy)
            adp_opt.run_trial(optimal_policy)
            
            due_errors_opt.append(due_opt.get_rms_error())
            td_errors_opt.append(td_opt.get_rms_error())
            adp_errors_opt.append(adp_opt.get_rms_error())
        
        results['DUE_optimal'].append(due_errors_opt)
        results['TD_optimal'].append(td_errors_opt)
        results['ADP_optimal'].append(adp_errors_opt)
        
        # Random policy experiments
        random_policy = get_random_policy(env)
        
        due_rand = DirectUtilityEstimation(env)
        td_rand = TDLearning(env, alpha=0.1)
        adp_rand = AdaptiveDynamicProgramming(env)
        
        due_errors_rand = []
        td_errors_rand = []
        adp_errors_rand = []
        
        for trial in range(num_trials):
            due_rand.run_trial(random_policy)
            td_rand.run_trial(random_policy)
            adp_rand.run_trial(random_policy)
            
            due_errors_rand.append(due_rand.get_rms_error())
            td_errors_rand.append(td_rand.get_rms_error())
            adp_errors_rand.append(adp_rand.get_rms_error())
        
        results['DUE_random'].append(due_errors_rand)
        results['TD_random'].append(td_errors_rand)
        results['ADP_random'].append(adp_errors_rand)
    
    return results


def plot_results(results, num_trials=60):
    """Plot comparison results."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Optimal policy plot
    ax = axes[0]
    trials = np.arange(1, num_trials + 1)
    
    due_mean = np.mean(results['DUE_optimal'], axis=0)
    td_mean = np.mean(results['TD_optimal'], axis=0)
    adp_mean = np.mean(results['ADP_optimal'], axis=0)
    
    ax.plot(trials, due_mean, 'r-', label='DUE', linewidth=2)
    ax.plot(trials, td_mean, 'b-', label='TD', linewidth=2)
    ax.plot(trials, adp_mean, 'g-', label='ADP', linewidth=2)
    
    ax.set_xlabel('Number of Trials', fontsize=12)
    ax.set_ylabel('RMS Error', fontsize=12)
    ax.set_title('Optimal Policy', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.0)
    
    # Random policy plot
    ax = axes[1]
    
    due_mean = np.mean(results['DUE_random'], axis=0)
    td_mean = np.mean(results['TD_random'], axis=0)
    adp_mean = np.mean(results['ADP_random'], axis=0)
    
    ax.plot(trials, due_mean, 'r-', label='DUE', linewidth=2)
    ax.plot(trials, td_mean, 'b-', label='TD', linewidth=2)
    ax.plot(trials, adp_mean, 'g-', label='ADP', linewidth=2)
    
    ax.set_xlabel('Number of Trials', fontsize=12)
    ax.set_ylabel('RMS Error', fontsize=12)
    ax.set_title('Random Policy', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.5)
    
    plt.tight_layout()
    plt.savefig('exercise_21_1_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Plot saved to 'exercise_21_1_results.png'")


def print_final_utilities(env, due, td, adp):
    """Print final utility estimates comparison."""
    print("\n" + "="*70)
    print("FINAL UTILITY ESTIMATES COMPARISON")
    print("="*70)
    print(f"{'State':<12} {'True U':<12} {'DUE':<12} {'TD':<12} {'ADP':<12}")
    print("-"*70)
    
    for state in sorted(env.get_states()):
        true_u = TRUE_UTILITIES.get(state, 0)
        due_u = due.U.get(state, 0)
        td_u = td.U.get(state, 0)
        adp_u = adp.U.get(state, 0)
        print(f"{str(state):<12} {true_u:<12.3f} {due_u:<12.3f} {td_u:<12.3f} {adp_u:<12.3f}")


def run_single_detailed_experiment(num_trials=100):
    """Run a single detailed experiment for display."""
    env = GridWorld()
    optimal_policy = get_optimal_policy()
    
    due = DirectUtilityEstimation(env)
    td = TDLearning(env, alpha=0.1)
    adp = AdaptiveDynamicProgramming(env)
    
    print("\nRunning single detailed experiment with optimal policy...")
    for trial in range(num_trials):
        due.run_trial(optimal_policy)
        td.run_trial(optimal_policy)
        adp.run_trial(optimal_policy)
        
        if (trial + 1) % 20 == 0:
            print(f"Trial {trial + 1}: DUE={due.get_rms_error():.4f}, "
                  f"TD={td.get_rms_error():.4f}, ADP={adp.get_rms_error():.4f}")
    
    print_final_utilities(env, due, td, adp)
    
    return due, td, adp


if __name__ == "__main__":
    print("="*70)
    print("EXERCISE 21.1: PASSIVE LEARNING AGENT COMPARISON")
    print("="*70)
    
    # Run single detailed experiment first
    due, td, adp = run_single_detailed_experiment(60)
    
    # Run full experiment with multiple runs
    print("\n" + "="*70)
    print("RUNNING FULL EXPERIMENT (10 runs x 60 trials each)")
    print("="*70)
    
    results = run_experiment(num_trials=60, num_runs=10)
    
    # Calculate final statistics
    print("\n" + "="*70)
    print("FINAL RMS ERROR AFTER 60 TRIALS (averaged over 10 runs)")
    print("="*70)
    
    for policy_type in ['optimal', 'random']:
        print(f"\n{policy_type.upper()} POLICY:")
        for algo in ['DUE', 'TD', 'ADP']:
            key = f'{algo}_{policy_type}'
            final_errors = [run[-1] for run in results[key]]
            mean_error = np.mean(final_errors)
            std_error = np.std(final_errors)
            print(f"  {algo}: {mean_error:.4f} ± {std_error:.4f}")
    
    # Plot results
    plot_results(results)

