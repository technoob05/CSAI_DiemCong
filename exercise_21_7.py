"""
Exercise 21.7: Reinforcement Learning for Game Playing (Tic-Tac-Toe)
Implement TD learning with self-play for Tic-Tac-Toe.
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import random
import copy
from tqdm import tqdm

# ============= TIC-TAC-TOE ENVIRONMENT =============
class TicTacToe:
    def __init__(self):
        self.board = [' '] * 9  # 3x3 board flattened
        self.current_player = 'X'  # X always starts
    
    def reset(self):
        self.board = [' '] * 9
        self.current_player = 'X'
        return self.get_state()
    
    def get_state(self):
        """Return state as a tuple (hashable)."""
        return (tuple(self.board), self.current_player)
    
    def get_available_actions(self):
        """Return list of empty positions."""
        return [i for i in range(9) if self.board[i] == ' ']
    
    def make_move(self, position):
        """Make a move and return (next_state, reward, done, winner)."""
        if self.board[position] != ' ':
            raise ValueError(f"Invalid move: position {position} is occupied")
        
        self.board[position] = self.current_player
        
        winner = self.check_winner()
        if winner:
            reward = 1.0 if winner == self.current_player else -1.0
            return self.get_state(), reward, True, winner
        
        if ' ' not in self.board:  # Draw
            return self.get_state(), 0.0, True, None
        
        # Switch player
        self.current_player = 'O' if self.current_player == 'X' else 'X'
        return self.get_state(), 0.0, False, None
    
    def check_winner(self):
        """Check if there's a winner. Return 'X', 'O', or None."""
        lines = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
            [0, 4, 8], [2, 4, 6]              # Diagonals
        ]
        for line in lines:
            if self.board[line[0]] == self.board[line[1]] == self.board[line[2]] != ' ':
                return self.board[line[0]]
        return None
    
    def display(self):
        """Print the board."""
        for i in range(3):
            row = ' | '.join(self.board[i*3:(i+1)*3])
            print(f" {row} ")
            if i < 2:
                print("-----------")
        print()
    
    def copy(self):
        """Return a copy of the game."""
        new_game = TicTacToe()
        new_game.board = self.board.copy()
        new_game.current_player = self.current_player
        return new_game


# ============= TD LEARNING AGENT =============
class TDAgent:
    def __init__(self, player, alpha=0.1, gamma=1.0, epsilon=0.1):
        self.player = player  # 'X' or 'O'
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.V = defaultdict(lambda: 0.0)  # State value function
        self.episode_states = []  # States visited in current episode
    
    def get_features(self, state):
        """Extract features from board state."""
        board, player = state
        
        # Count winning threats, blocking needs, etc.
        features = {
            'my_threats_2': 0,      # Lines with 2 of mine and 0 opponent
            'opp_threats_2': 0,     # Lines with 2 opponent and 0 mine
            'my_threats_1': 0,      # Lines with 1 of mine and 0 opponent
            'center': 0,            # 1 if I control center
            'corners': 0,           # Number of corners I control
        }
        
        lines = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],
            [0, 3, 6], [1, 4, 7], [2, 5, 8],
            [0, 4, 8], [2, 4, 6]
        ]
        
        me = self.player
        opp = 'O' if me == 'X' else 'X'
        
        for line in lines:
            my_count = sum(1 for i in line if board[i] == me)
            opp_count = sum(1 for i in line if board[i] == opp)
            
            if my_count == 2 and opp_count == 0:
                features['my_threats_2'] += 1
            elif opp_count == 2 and my_count == 0:
                features['opp_threats_2'] += 1
            elif my_count == 1 and opp_count == 0:
                features['my_threats_1'] += 1
        
        if board[4] == me:
            features['center'] = 1
        
        for corner in [0, 2, 6, 8]:
            if board[corner] == me:
                features['corners'] += 1
        
        return features
    
    def get_state_value(self, state):
        """Get value of a state."""
        board, player = state
        
        # Check terminal states
        game = TicTacToe()
        game.board = list(board)
        winner = game.check_winner()
        
        if winner == self.player:
            return 1.0
        elif winner is not None:
            return -1.0
        elif ' ' not in board:
            return 0.0
        
        return self.V[state]
    
    def choose_action(self, game, training=True):
        """Choose action using epsilon-greedy policy."""
        actions = game.get_available_actions()
        
        if training and random.random() < self.epsilon:
            return random.choice(actions)
        
        # Choose action with highest expected value
        best_action = None
        best_value = float('-inf')
        
        for action in actions:
            game_copy = game.copy()
            next_state, reward, done, winner = game_copy.make_move(action)
            
            if done:
                if winner == self.player:
                    value = 1.0
                elif winner is not None:
                    value = -1.0
                else:
                    value = 0.0
            else:
                # For the opponent's perspective, negate the value
                value = -self.get_state_value(next_state)
            
            if value > best_value:
                best_value = value
                best_action = action
        
        return best_action
    
    def update(self, state, next_state, reward, done):
        """TD(0) update."""
        if done:
            target = reward
        else:
            # From my perspective, opponent's good state is my bad state
            target = reward + self.gamma * (-self.get_state_value(next_state))
        
        current_value = self.V[state]
        self.V[state] += self.alpha * (target - current_value)


# ============= RANDOM AGENT =============
class RandomAgent:
    def __init__(self, player):
        self.player = player
    
    def choose_action(self, game, training=True):
        return random.choice(game.get_available_actions())


# ============= MINIMAX AGENT (for testing) =============
class MinimaxAgent:
    def __init__(self, player):
        self.player = player
    
    def choose_action(self, game, training=True):
        _, best_action = self.minimax(game, True)
        return best_action
    
    def minimax(self, game, is_maximizing, alpha=float('-inf'), beta=float('inf')):
        winner = game.check_winner()
        if winner == self.player:
            return 1, None
        elif winner is not None:
            return -1, None
        elif ' ' not in game.board:
            return 0, None
        
        actions = game.get_available_actions()
        best_action = actions[0]
        
        if is_maximizing:
            best_value = float('-inf')
            for action in actions:
                game_copy = game.copy()
                game_copy.make_move(action)
                value, _ = self.minimax(game_copy, False, alpha, beta)
                if value > best_value:
                    best_value = value
                    best_action = action
                alpha = max(alpha, value)
                if beta <= alpha:
                    break
            return best_value, best_action
        else:
            best_value = float('inf')
            for action in actions:
                game_copy = game.copy()
                game_copy.make_move(action)
                value, _ = self.minimax(game_copy, True, alpha, beta)
                if value < best_value:
                    best_value = value
                    best_action = action
                beta = min(beta, value)
                if beta <= alpha:
                    break
            return best_value, best_action


# ============= TRAINING AND EVALUATION =============
def play_game(agent1, agent2, game, training=True):
    """Play one game between two agents."""
    state = game.reset()
    agents = {'X': agent1, 'O': agent2}
    states_by_player = {'X': [], 'O': []}
    
    done = False
    while not done:
        current_player = game.current_player
        agent = agents[current_player]
        
        action = agent.choose_action(game, training)
        old_state = game.get_state()
        states_by_player[current_player].append(old_state)
        
        next_state, reward, done, winner = game.make_move(action)
        
        # TD update for learning agents
        if training and isinstance(agent, TDAgent):
            if done:
                if winner == agent.player:
                    agent.update(old_state, next_state, 1.0, done)
                elif winner is not None:
                    agent.update(old_state, next_state, -1.0, done)
                else:
                    agent.update(old_state, next_state, 0.0, done)
            else:
                agent.update(old_state, next_state, 0.0, done)
    
    # Update for the other agent on game end
    if training:
        for player in ['X', 'O']:
            agent = agents[player]
            if isinstance(agent, TDAgent) and states_by_player[player]:
                last_state = states_by_player[player][-1]
                if winner == agent.player:
                    final_reward = 1.0
                elif winner is not None:
                    final_reward = -1.0
                else:
                    final_reward = 0.0
                # Final state update
                agent.V[last_state] = agent.V[last_state] + agent.alpha * (final_reward - agent.V[last_state])
    
    return winner


def evaluate_agent(agent, opponent, num_games=100):
    """Evaluate agent against opponent."""
    wins = 0
    losses = 0
    draws = 0
    
    for i in range(num_games):
        game = TicTacToe()
        
        # Alternate who goes first
        if i % 2 == 0:
            winner = play_game(agent, opponent, game, training=False)
            if winner == agent.player:
                wins += 1
            elif winner is not None:
                losses += 1
            else:
                draws += 1
        else:
            # Swap players for opponent
            winner = play_game(opponent, agent, game, training=False)
            if winner == agent.player:
                wins += 1
            elif winner is not None:
                losses += 1
            else:
                draws += 1
    
    return wins, losses, draws


def train_self_play(num_episodes=10000):
    """Train two TD agents through self-play."""
    print("Training through self-play...")
    
    agent_x = TDAgent('X', alpha=0.1, epsilon=0.3)
    agent_o = TDAgent('O', alpha=0.1, epsilon=0.3)
    
    # Tracking metrics
    x_wins = []
    o_wins = []
    draws = []
    win_vs_random = []
    
    window = 500
    x_win_count = 0
    o_win_count = 0
    draw_count = 0
    
    for episode in tqdm(range(num_episodes), desc="Self-play training"):
        game = TicTacToe()
        winner = play_game(agent_x, agent_o, game, training=True)
        
        if winner == 'X':
            x_win_count += 1
        elif winner == 'O':
            o_win_count += 1
        else:
            draw_count += 1
        
        # Decay epsilon
        if episode % 1000 == 0 and episode > 0:
            agent_x.epsilon = max(0.05, agent_x.epsilon * 0.9)
            agent_o.epsilon = max(0.05, agent_o.epsilon * 0.9)
        
        # Record stats every window episodes
        if (episode + 1) % window == 0:
            x_wins.append(x_win_count / window * 100)
            o_wins.append(o_win_count / window * 100)
            draws.append(draw_count / window * 100)
            
            # Evaluate against random
            random_agent = RandomAgent('O')
            wins, losses, draws_count = evaluate_agent(agent_x, random_agent, num_games=100)
            win_vs_random.append(wins)
            
            x_win_count = 0
            o_win_count = 0
            draw_count = 0
    
    return agent_x, agent_o, x_wins, o_wins, draws, win_vs_random


def plot_results(x_wins, o_wins, draws, win_vs_random):
    """Plot training results."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Self-play results
    ax = axes[0]
    episodes = np.arange(500, len(x_wins) * 500 + 1, 500)
    ax.plot(episodes, x_wins, 'b-', label='X Wins', linewidth=2)
    ax.plot(episodes, o_wins, 'r-', label='O Wins', linewidth=2)
    ax.plot(episodes, draws, 'g-', label='Draws', linewidth=2)
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Episodes', fontsize=12)
    ax.set_ylabel('Percentage', fontsize=12)
    ax.set_title('Self-Play Training Results', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)
    
    # Win rate vs random
    ax = axes[1]
    ax.plot(episodes, win_vs_random, 'b-', linewidth=2)
    ax.axhline(y=90, color='green', linestyle='--', alpha=0.5, label='90% target')
    ax.set_xlabel('Training Episodes', fontsize=12)
    ax.set_ylabel('Win Rate vs Random (%)', fontsize=12)
    ax.set_title('Win Rate Against Random Opponent', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig('results/exercise_21_7_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nPlot saved to 'results/exercise_21_7_results.png'")


def demo_games(agent):
    """Show some demo games."""
    print("\n" + "="*50)
    print("DEMO: Trained Agent vs Random")
    print("="*50)
    
    random_agent = RandomAgent('O')
    game = TicTacToe()
    game.reset()
    
    print("\nGame 1:")
    agents = {'X': agent, 'O': random_agent}
    
    while True:
        print(f"\n{game.current_player}'s turn:")
        action = agents[game.current_player].choose_action(game, training=False)
        print(f"Plays position {action}")
        state, reward, done, winner = game.make_move(action)
        game.display()
        
        if done:
            if winner:
                print(f"Winner: {winner}!")
            else:
                print("Draw!")
            break


if __name__ == "__main__":
    print("="*70)
    print("EXERCISE 21.7: TD LEARNING FOR TIC-TAC-TOE")
    print("="*70)
    
    # Train agents
    agent_x, agent_o, x_wins, o_wins, draws, win_vs_random = train_self_play(num_episodes=10000)
    
    # Final evaluation
    print("\n" + "="*70)
    print("FINAL EVALUATION")
    print("="*70)
    
    random_agent = RandomAgent('O')
    wins, losses, draws_count = evaluate_agent(agent_x, random_agent, num_games=1000)
    print(f"\nAgent X vs Random (1000 games):")
    print(f"  Wins: {wins/10:.1f}%, Losses: {losses/10:.1f}%, Draws: {draws_count/10:.1f}%")
    
    random_agent_x = RandomAgent('X')
    wins, losses, draws_count = evaluate_agent(agent_o, random_agent_x, num_games=1000)
    print(f"\nAgent O vs Random (1000 games):")
    print(f"  Wins: {wins/10:.1f}%, Losses: {losses/10:.1f}%, Draws: {draws_count/10:.1f}%")
    
    # Self-play evaluation
    print("\nSelf-play (trained agents, 1000 games):")
    x_count = 0
    o_count = 0
    draw_count = 0
    for _ in tqdm(range(1000), desc="Self-play evaluation"):
        game = TicTacToe()
        winner = play_game(agent_x, agent_o, game, training=False)
        if winner == 'X':
            x_count += 1
        elif winner == 'O':
            o_count += 1
        else:
            draw_count += 1
    print(f"  X Wins: {x_count/10:.1f}%, O Wins: {o_count/10:.1f}%, Draws: {draw_count/10:.1f}%")
    
    # Print number of learned states
    print(f"\nLearned States:")
    print(f"  Agent X: {len(agent_x.V)} states")
    print(f"  Agent O: {len(agent_o.V)} states")
    
    # Plot results
    plot_results(x_wins, o_wins, draws, win_vs_random)
    
    # Demo game
    demo_games(agent_x)

