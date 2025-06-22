import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Disable GPU before TensorFlow import
import numpy as np
import pandas as pd
import time
import psutil
import random
from itertools import combinations
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf
from collections import deque
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Force CPU execution
tf.config.set_soft_device_placement(True)
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    logger.warning(f"GPUs detected: {physical_devices}. Forcing CPU execution.")
    tf.config.set_visible_devices([], 'GPU')

# Fix random seed for reproducibility
np.random.seed(111)
tf.random.set_seed(111)

# Define the output directory
output_dir = "~/RLAR/results"
os.makedirs(output_dir, exist_ok=True)

# Load the dataset
try:
    transactions_df = pd.read_csv(os.path.join(output_dir, "synthetic_transactions.csv"))
    transactions = [eval(t) for t in transactions_df['Transaction']]
    logger.info(f'Transactions 0-5: {transactions[:5]}')
except FileNotFoundError:
    logger.error("synthetic_transactions.csv not found in the specified directory.")
    exit(1)

# Get number of items
try:
    X_df = pd.read_csv(os.path.join(output_dir, "feature_matrix.csv"))
    num_items = X_df.shape[0]
    logger.info(f"Number of items: {num_items}")
except FileNotFoundError:
    logger.error("feature_matrix.csv not found in the specified directory.")
    exit(1)

# Convert transactions to binary matrix
def transactions_to_binary_matrix(transactions, num_items):
    binary_matrix = np.zeros((len(transactions), num_items), dtype=bool)
    for i, t in enumerate(transactions):
        try:
            binary_matrix[i, t] = True
        except IndexError:
            logger.error(f"Invalid item index in transaction {i}: {t}")
            exit(1)
    return binary_matrix

binary_matrix = transactions_to_binary_matrix(transactions, num_items)
if binary_matrix.shape[1] != num_items:
    logger.error(f"Binary matrix has {binary_matrix.shape[1]} items, expected {num_items}")
    exit(1)
T_df = pd.DataFrame(binary_matrix, columns=[f'item_{i}' for i in range(num_items)])

# Function to get current memory usage
def get_memory_usage():
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024  # Convert bytes to MB
    return round(memory_mb, 2)

# DQN Model
class DQN(tf.keras.Model):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(128, activation='relu')
        self.dense3 = tf.keras.layers.Dense(128, activation='relu')
        self.output_layer = tf.keras.layers.Dense(action_size)

    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        x = self.dense3(x)
        return self.output_layer(x)

# RLAR Environment
class RLAR_Environment:
    def __init__(self, binary_matrix, min_supp, min_conf, max_itemset_size):
        self.binary_matrix = binary_matrix
        self.N = binary_matrix.shape[0]
        self.M = binary_matrix.shape[1]
        self.min_supp = min_supp
        self.min_conf = min_conf
        self.max_itemset_size = max_itemset_size
        self.reset()

    def reset(self):
        self.itemset = np.zeros(self.M, dtype=bool)
        self.step_count = 0
        support = self.compute_support(self.itemset)
        max_conf = 0.0
        size = np.sum(self.itemset)
        state = np.concatenate([self.itemset.astype(float), [support, max_conf, size]])
        logger.debug(f"Reset state shape: {state.shape}")
        return state

    def compute_support(self, itemset):
        indices = np.where(itemset)[0]
        if len(indices) == 0:
            return 1.0
        subset_matrix = self.binary_matrix[:, indices]
        support_count = np.sum(np.all(subset_matrix, axis=1))
        return support_count / self.N

    def compute_reward(self, itemset):
        support = self.compute_support(itemset)
        if support < self.min_supp:
            return -1.0
        indices = np.where(itemset)[0]
        valid_rules = []
        for r in range(1, len(indices)):
            for antecedent in combinations(indices, r):
                antecedent = set(antecedent)
                consequent = set(indices) - antecedent
                p_A = self.compute_support(np.array([1 if i in antecedent else 0 for i in range(self.M)]))
                p_B = self.compute_support(np.array([1 if i in consequent else 0 for i in range(self.M)]))
                if p_A > 0:
                    conf = support / p_A
                    if conf >= self.min_conf:
                        lift = conf / p_B if p_B > 0 else 0
                        score = 0.5 * conf + 0.5 * lift
                        valid_rules.append(score)
        reward = max(valid_rules) if valid_rules else 0.0
        if reward > 0:
            logger.debug(f"Positive reward: {reward} for itemset: {np.where(itemset)[0]}")
        return reward

    def step(self, action):
        self.step_count += 1
        new_itemset = self.itemset.copy()
        new_itemset[action] = not new_itemset[action]
        support = self.compute_support(new_itemset)
        max_conf = 0.0
        indices = np.where(new_itemset)[0]
        if support >= self.min_supp:
            for r in range(1, len(indices)):
                for antecedent in combinations(indices, r):
                    antecedent = set(antecedent)
                    p_A = self.compute_support(np.array([1 if i in antecedent else 0 for i in range(self.M)]))
                    if p_A > 0:
                        conf = support / p_A
                        max_conf = max(max_conf, conf)
        size = np.sum(new_itemset)
        reward = self.compute_reward(new_itemset)
        terminal = (self.step_count >= S_max or size > self.max_itemset_size or reward == -1.0)
        self.itemset = new_itemset
        state = np.concatenate([self.itemset.astype(float), [support, max_conf, size]])
        logger.debug(f"Step {self.step_count}: Action {action}, Reward {reward}, Terminal {terminal}")
        return state, reward, terminal

# RLAR Agent
class RLAR_Agent:
    def __init__(self, state_size, action_size, env, epsilon_min):
        self.state_size = state_size
        self.action_size = action_size
        self.env = env
        self.memory = deque(maxlen=B)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = np.exp(np.log(self.epsilon_min / epsilon) / 1000) #reach epsilon_min after 1000 steps
        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.update_target_model()
        self.replay_success_count = 0
        self.replay_error_count = 0

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, terminal):
        self.memory.append((state, action, reward, next_state, terminal))

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            action = np.random.randint(self.action_size)
            logger.debug(f"Random action: {action}")
            return action
        state = np.expand_dims(state, axis=0)
        q_values = self.model(state)
        action = np.argmax(q_values[0])
        logger.debug(f"Greedy action: {action}, Q-values: {q_values[0][:5]}...")
        return action

    def replay(self):
        if len(self.memory) < B_m:
            return
        minibatch = random.sample(self.memory, B_m)
        states = np.array([item[0] for item in minibatch])
        actions = np.array([item[1] for item in minibatch])
        rewards = np.array([item[2] for item in minibatch])
        next_states = np.array([item[3] for item in minibatch])
        terminals = np.array([item[4] for item in minibatch])

        try:
            with tf.GradientTape() as tape:
                q_values = self.model(states)
                q_values_selected = tf.reduce_sum(q_values * tf.one_hot(actions, self.action_size), axis=1)
                target_q_values = self.target_model(next_states)
                target_values = rewards + self.gamma * tf.reduce_max(target_q_values, axis=1) * (1 - terminals)
                loss = tf.reduce_mean(tf.square(target_values - q_values_selected))
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            self.replay_success_count += 1
            if self.replay_success_count % 100 == 0:
                logger.info(f"Successful replays: {self.replay_success_count}, Errors: {self.replay_error_count}")
        except Exception as e:
            self.replay_error_count += 1
            logger.error(f"Error in replay #{self.replay_success_count + self.replay_error_count}: {e}")
            if self.replay_error_count % 100 == 0:
                logger.info(f"Replay errors: {self.replay_error_count}, Successes: {self.replay_success_count}")

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

# Experiment parameters
min_supp_values = [0.1, 0.2, 0.3, 0.4, 0.5]
min_conf = 0.5
m_max = 10 # Maximum size of itemsets
E_max = 1000 # Number of episodes
S_max = 100 # 500, maximum steps per episode
B = 10000 # Memory buffer size
B_m = 32 # Minibatch size
gamma = 0.99 # Discount factor
epsilon = 1.0   # initial exploration rate
epsilon_min = 0.1  # Minimum exploration rate
F = 1000 # Frequency of target model updates
k = 3   # Number of top actions to consider
state_size = num_items + 3  # bit-vector + support, max_conf, size
action_size = num_items
logger.info(f"State size: {state_size}, Action size: {action_size}")

# Initialize lists to store metrics
rlar_runtimes, rlar_memory, rlar_num_rules, rlar_num_itemsets = [], [], [], []

for min_supp in tqdm(min_supp_values, desc="Processing min_supp values"):
    start_time = time.time()
    start_memory = get_memory_usage()
    logger.info(f"Starting RLAR experiment for min_supp={min_supp}")

    try:
        # Initialize environment and agent
        env = RLAR_Environment(binary_matrix, min_supp, min_conf, m_max)
        agent = RLAR_Agent(state_size, action_size, env, epsilon_min)

        # Training with cumulative reward tracking
        episodes_completed = 0
        episode_rewards = []  # Store cumulative rewards per episode
        for episode in tqdm(range(E_max), desc=f"Training RLAR min_supp={min_supp}"):
            state = env.reset()
            episode_reward = 0  # Track reward for current episode
            for step in range(S_max):
                action = agent.choose_action(state)
                next_state, reward, terminal = env.step(action)
                episode_reward += reward  # Accumulate reward
                agent.remember(state, action, reward, next_state, terminal)
                agent.replay()
                state = next_state
                if terminal:
                    break
            agent.decay_epsilon()
            episodes_completed += 1
            episode_rewards.append(episode_reward)  # Store episode reward
            if (episode + 1) % F == 0:
                agent.update_target_model()
        logger.info(f"Completed {episodes_completed} episodes for min_supp={min_supp}")

        # plot cumulative rewards over episodes
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(episode_rewards) + 1), episode_rewards, label='Cumulative Reward')
        plt.xlabel('Episode')
        plt.ylabel('Cumulative Reward')
        plt.title(f'Cumulative Reward over Episodes (min_supp={min_supp})')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f"reward_plot_min_supp_{min_supp}.png"))
        plt.close()

        # Rules extraction
        rlar_rules = []
        rlar_frequent_itemsets = []
        trajectories = [(env.reset(), np.zeros(num_items, dtype=bool))]
        step = 0
        while step < S_max and trajectories:
            new_trajectories = []
            for state, itemset in trajectories:
                state_tensor = np.expand_dims(state, axis=0)
                q_values = agent.model(state_tensor).numpy()[0]
                top_k_actions = np.argsort(q_values)[-k:]
                for action in top_k_actions:
                    next_state, _, terminal = env.step(action)
                    next_itemset = env.itemset.copy()
                    if np.sum(next_itemset) <= m_max:
                        support = env.compute_support(next_itemset)
                        if support >= min_supp:
                            itemset_indices = tuple(np.where(next_itemset)[0])
                            if itemset_indices not in rlar_frequent_itemsets:
                                rlar_frequent_itemsets.append(itemset_indices)
                                logger.debug(f"Found frequent itemset: {itemset_indices}, Support: {support}")
                            indices = np.where(next_itemset)[0]
                            for r in range(1, len(indices)):
                                for antecedent in combinations(indices, r):
                                    antecedent = set(antecedent)
                                    consequent = set(indices) - antecedent
                                    p_A = env.compute_support(np.array([1 if i in antecedent else 0 for i in range(num_items)]))
                                    if p_A > 0:
                                        conf = support / p_A
                                        if conf >= min_conf:
                                            p_B = env.compute_support(np.array([1 if i in consequent else 0 for i in range(num_items)]))
                                            lift = conf / p_B if p_B > 0 else 0
                                            rule = (antecedent, consequent, support, conf, lift)
                                            if rule not in rlar_rules:
                                                rlar_rules.append(rule)
                                                logger.debug(f"Found rule: {antecedent} -> {consequent}, Conf: {conf}, Lift: {lift}")
                            if not terminal:
                                new_trajectories.append((next_state, next_itemset))
                    env.itemset = itemset  # Restore itemset for next action
            trajectories = new_trajectories
            step += 1

        end_time = time.time()
        end_memory = get_memory_usage()

        rlar_runtimes.append(end_time - start_time)
        rlar_memory.append(max(end_memory - start_memory, 0))
        rlar_num_rules.append(len(rlar_rules))
        rlar_num_itemsets.append(len(rlar_frequent_itemsets))

        # Save RLAR summary table
        rlar_summary_data = []
        for i, (antecedent, consequent, support, conf, lift) in enumerate(rlar_rules, 1):
            antecedent_items = ', '.join([f'item_{idx}' for idx in sorted(list(antecedent))])
            consequent_items = ', '.join([f'item_{idx}' for idx in sorted(list(consequent))])
            rlar_summary_data.append({
                '#': i,
                'Premises': antecedent_items,
                'Conclusion': consequent_items,
                'Support': support,
                'Confidence': conf,
                'Lift': lift
            })
        rlar_summary_df = pd.DataFrame(rlar_summary_data)
        if not rlar_summary_df.empty:
            rlar_summary_df.to_csv(os.path.join(output_dir, f"rlar_summary_min_supp_{min_supp}.csv"), index=False)
        else:
            logger.warning(f"No RLAR rules found for min_supp={min_supp}. Skipping summary save.")

        # Save metrics
        metrics_data = {
            'Min_Supp': [min_supp],
            'Algorithm': ['RLAR'],
            'Runtime (s)': [rlar_runtimes[-1]],
            'Memory Usage (MB)': [rlar_memory[-1]],
            'Number of Frequent Itemsets': [rlar_num_itemsets[-1]],
            'Number of Rules': [rlar_num_rules[-1]]
        }
        metrics_df = pd.DataFrame(metrics_data)
        metrics_df.to_csv(os.path.join(output_dir, f"metrics_min_supp_{min_supp}.csv"), index=False)

        # Analysis of top rules with a specific RHS item
        def get_top_rules_rlar(rules, j, n=10):
            filtered_rules = [rule for rule in rules if len(rule[1]) == 1 and list(rule[1])[0] == j]
            sorted_rules = sorted(filtered_rules, key=lambda x: x[3], reverse=True)  # Sort by confidence
            return sorted_rules[:n]

        RHS_item_no = 2
        top_n = 10
        top_rlar_rules = get_top_rules_rlar(rlar_rules, RHS_item_no, top_n)
        rlar_analysis_data = []
        if top_rlar_rules:
            for rule in top_rlar_rules:
                antecedent, consequent, support, conf, lift = rule
                full_itemset = sorted(list(antecedent) + list(consequent))
                supporting_indices = [i for i, t in enumerate(transactions) if set(full_itemset).issubset(set(t))]
                support_count = len(supporting_indices)
                example_indices = supporting_indices[:3] if len(supporting_indices) >= 3 else supporting_indices
                rlar_analysis_data.append({
                    'Rule': f"{sorted(list(antecedent))} -> {sorted(list(consequent))}",
                    'Confidence': conf,
                    'Support (prob)': support,
                    'Support Count': support_count,
                    'Example Transaction Indices': example_indices,
                    'Example Transactions': [transactions[xx] for xx in example_indices]
                })
            rlar_analysis_df = pd.DataFrame(rlar_analysis_data)
            rlar_analysis_df.to_csv(os.path.join(output_dir, f"rlar_analysis_min_supp_{min_supp}.csv"), index=False)
        else:
            logger.warning(f"No RLAR rules found with item {RHS_item_no} on RHS for min_supp={min_supp}")
    except Exception as e:
        logger.error(f"Error in RLAR experiment for min_supp={min_supp}: {e}")
        continue

# Save final summary of metrics
if rlar_runtimes:
    final_metrics_data = {
        'Min_Supp': min_supp_values[:len(rlar_runtimes)],
        'RLAR_Runtime (s)': rlar_runtimes,
        'RLAR_Memory (MB)': rlar_memory,
        'RLAR_Frequent_Itemsets': rlar_num_itemsets,
        'RLAR_Rules': rlar_num_rules
    }
    try:
        final_metrics_df = pd.DataFrame(final_metrics_data)
        final_metrics_df.to_csv(os.path.join(output_dir, "final_metrics_summary_rlar.csv"), index=False)
    except ValueError as e:
        logger.error(f"Error creating final metrics DataFrame: {e}")
else:
    logger.warning("No metrics collected. Skipping final metrics summary save.")

# Print final metrics
logger.info(f"Number of frequent itemsets (RLAR): {rlar_num_itemsets}")
logger.info(f"Number of rules (RLAR): {rlar_num_rules}")
