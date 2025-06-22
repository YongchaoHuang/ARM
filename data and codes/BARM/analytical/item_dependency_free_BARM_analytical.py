import numpy as np
import pandas as pd
import os
import time
import psutil
from itertools import combinations
from tqdm import tqdm
import matplotlib.pyplot as plt

# fix random seed for reproducibility
np.random.seed(111)  # set seed for NumPy's random number generator

# Define the output directory
output_dir = "~/BARM/results"
os.makedirs(output_dir, exist_ok=True)

# Load the dataset
transactions_df = pd.read_csv(os.path.join(output_dir, "synthetic_transactions.csv"))
transactions = [eval(t) for t in transactions_df['Transaction']]  # Parse string representation
print(f'Transactions 0-5: \n {transactions[:5]}')

# Get number of items (assuming feature_matrix.csv defines the item set)
X_df = pd.read_csv(os.path.join(output_dir, "feature_matrix.csv"))
num_items = X_df.shape[0]

# Convert transactions to binary matrix
def transactions_to_binary_matrix(transactions, num_items):
    binary_matrix = np.zeros((len(transactions), num_items), dtype=bool)
    for i, t in enumerate(transactions):
        binary_matrix[i, t] = True
    return binary_matrix

binary_matrix = transactions_to_binary_matrix(transactions, num_items)
T_df = pd.DataFrame(binary_matrix, columns=[f'item_{i}' for i in range(num_items)])

# Analytical posterior sampling
def sample_posterior_analytical(binary_matrix, alpha, beta, S):
    """
    Compute analytical Beta posterior and sample S values for each item's presence probability.
    Parameters:
        binary_matrix: (N x M) binary matrix of transactions
        alpha, beta: (M,) arrays of Beta prior parameters
        S: Number of samples
    Returns:
        p_samples: (S x M) array of sampled probabilities
    """
    N = binary_matrix.shape[0]  # Number of transactions
    n_i = np.sum(binary_matrix, axis=0)  # Count of transactions where item i is present
    alpha_post = alpha + n_i  # Posterior alpha: alpha_i + n_i
    beta_post = beta + (N - n_i)  # Posterior beta: beta_i + (N - n_i)
    
    # Sample S values from Beta posterior for each item
    p_samples = np.zeros((S, num_items))
    for i in range(num_items):
        p_samples[:, i] = np.random.beta(alpha_post[i], beta_post[i], size=S)
    
    return p_samples

# Co-occurrence probability estimation
def cooccurrence_prob(itemset, p_samples, S):
    """
    Estimate co-occurrence probability for an itemset using posterior samples.
    Parameters:
        itemset: Tuple of item indices
        p_samples: (S x M) array of posterior samples
        S: Number of samples
    Returns:
        prob: Estimated co-occurrence probability
    """
    item_indices = list(itemset)
    prob = np.mean(np.prod(p_samples[:, item_indices], axis=1))
    return prob

# Function to get current memory usage
def get_memory_usage():
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024  # Convert bytes to MB
    return round(memory_mb, 3)

# BARM algorithm parameters
min_prob_values = [0.1, 0.2, 0.3, 0.4, 0.5]
min_conf = 0.5
S = 1000  # Number of posterior samples for probability estimation
alpha = np.ones(num_items)  # Uniform Beta(1,1) prior
beta = np.ones(num_items)

# Initialize lists to store metrics
barm_runtimes, barm_memory, barm_num_rules, barm_num_itemsets = [], [], [], []

for min_prob in tqdm(min_prob_values, desc="Processing min_prob values"):
    start_time = time.time()
    start_memory = get_memory_usage()

    # Analytical posterior sampling
    p_samples = sample_posterior_analytical(binary_matrix, alpha, beta, S)

    # Rule mining
    barm_rules = []
    barm_frequent_itemsets = []
    max_itemset_size = min(num_items + 1, num_items + 1)  # Full range of itemset sizes

    for m in range(2, max_itemset_size):
        itemsets = list(combinations(range(num_items), m))
        for itemset in tqdm(itemsets, desc=f"Itemsets size {m}"):
            p_I = cooccurrence_prob(itemset, p_samples, S)
            if p_I >= min_prob:
                barm_frequent_itemsets.append(itemset)
                # Rule generation from frequent itemset
                for r in range(1, m):
                    for antecedent_tuple in combinations(itemset, r):
                        antecedent = set(antecedent_tuple)
                        consequent = set(itemset) - antecedent
                        p_A = cooccurrence_prob(tuple(antecedent), p_samples, S)
                        p_B = cooccurrence_prob(tuple(consequent), p_samples, S)
                        if p_A > 0:
                            conf = p_I / p_A
                            if conf >= min_conf:
                                lift = p_I / (p_A * p_B) if p_B > 0 else 0
                                rule = (antecedent, consequent, p_I, conf, lift)
                                barm_rules.append(rule)

    end_time = time.time()
    end_memory = get_memory_usage()

    barm_runtimes.append(end_time - start_time)
    barm_memory.append(max(end_memory - start_memory, 0))
    barm_num_rules.append(len(barm_rules))
    barm_num_itemsets.append(len(barm_frequent_itemsets))

    # Save BARM summary table
    barm_summary_data = []
    for i, (antecedent, consequent, support, conf, lift) in enumerate(barm_rules, 1):
        antecedent_items = ', '.join([f'item_{idx}' for idx in sorted(list(antecedent))])
        consequent_items = ', '.join([f'item_{idx}' for idx in sorted(list(consequent))])
        barm_summary_data.append({
            '#': i,
            'Premises': antecedent_items,
            'Conclusion': consequent_items,
            'Support': support,
            'Confidence': conf,
            'Lift': lift
        })
    barm_summary_df = pd.DataFrame(barm_summary_data)
    if not barm_summary_df.empty:
        barm_summary_df.to_csv(os.path.join(output_dir, f"barm_summary_min_prob_{min_prob}.csv"), index=False)
    else:
        print(f"No BARM rules found for min_prob={min_prob}. Skipping summary save.")

    # Save metrics
    metrics_data = {
        'Min_Prob': [min_prob],
        'Algorithm': ['BARM'],
        'Runtime (s)': [barm_runtimes[-1]],
        'Memory Usage (MB)': [barm_memory[-1]],
        'Number of Frequent Itemsets': [barm_num_itemsets[-1]],
        'Number of Rules': [barm_num_rules[-1]]
    }
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.to_csv(os.path.join(output_dir, f"metrics_min_prob_{min_prob}.csv"), index=False)

    # Analysis of top rules with a specific RHS item
    def get_top_rules_barm(rules, j, n=10):
        filtered_rules = [rule for rule in rules if len(rule[1]) == 1 and list(rule[1])[0] == j]
        sorted_rules = sorted(filtered_rules, key=lambda x: x[3], reverse=True)  # Sort by confidence
        return sorted_rules[:n]

    RHS_item_no = 2  # Item index for RHS analysis
    top_n = 10
    top_barm_rules = get_top_rules_barm(barm_rules, RHS_item_no, top_n)
    barm_analysis_data = []
    if top_barm_rules:
        for rule in top_barm_rules:
            antecedent, consequent, support, conf, lift = rule
            full_itemset = sorted(list(antecedent) + list(consequent))
            supporting_indices = [i for i, t in enumerate(transactions) if set(full_itemset).issubset(set(t))]
            support_count = len(supporting_indices)
            example_indices = supporting_indices[:3] if len(supporting_indices) >= 3 else supporting_indices
            barm_analysis_data.append({
                'Rule': f"{sorted(list(antecedent))} -> {sorted(list(consequent))}",
                'Confidence': conf,
                'Support (prob)': support,
                'Support Count': support_count,
                'Example Transaction Indices': example_indices,
                'Example Transactions': [transactions[xx] for xx in example_indices]
            })
        barm_analysis_df = pd.DataFrame(barm_analysis_data)
        barm_analysis_df.to_csv(os.path.join(output_dir, f"barm_analysis_min_prob_{min_prob}.csv"), index=False)
    else:
        print(f"No BARM rules found with item {RHS_item_no} on RHS for min_prob={min_prob}.")

# Save final summary of metrics
final_metrics_data = {
    'Min_Prob': min_prob_values,
    'BARM_Runtime (s)': barm_runtimes,
    'BARM_Memory (MB)': barm_memory,
    'BARM_Frequent_Itemsets': barm_num_itemsets,
    'BARM_Rules': barm_num_rules
}
final_metrics_df = pd.DataFrame(final_metrics_data)
final_metrics_df.to_csv(os.path.join(output_dir, "final_metrics_summary_barm.csv"), index=False)

# Print final metrics
print("\nNumber of frequent itemsets (BARM):", barm_num_itemsets)
print("Number of rules (BARM):", barm_num_rules)
