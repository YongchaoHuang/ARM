import numpy as np
import pandas as pd
import os
import time
import psutil
from itertools import combinations
from tqdm import tqdm
import math
import matplotlib.pyplot as plt

# Fix random seed for reproducibility
np.random.seed(111)

# Define the output directory
output_dir = "~/MABARM/results"
os.makedirs(output_dir, exist_ok=True)

# Load the dataset
transactions_df = pd.read_csv(os.path.join(output_dir, "synthetic_transactions.csv"))
transactions = [eval(t) for t in transactions_df['Transaction']]
print(f'Transactions 0-5: \n {transactions[:5]}')

# Get number of items
X_df = pd.read_csv(os.path.join(output_dir, "feature_matrix.csv"))
num_items = X_df.shape[0]

# Convert transactions to binary matrix
def transactions_to_binary_matrix(transactions, num_items):
    """
    Convert transactions to a binary matrix.
    Parameters:
        transactions: List of transaction sets
        num_items: Number of items
    Returns:
        binary_matrix: (N x M) binary matrix of transactions
    """
    binary_matrix = np.zeros((len(transactions), num_items), dtype=bool)
    for i, t in enumerate(transactions):
        binary_matrix[i, t] = True
    return binary_matrix

binary_matrix = transactions_to_binary_matrix(transactions, num_items)
T_df = pd.DataFrame(binary_matrix, columns=[f'item_{i}' for i in range(num_items)])

# Co-occurrence probability estimation (empirical frequency)
def cooccurrence_prob(itemset, binary_matrix):
    """
    Estimate co-occurrence probability for an itemset using empirical frequency.
    Parameters:
        itemset: Tuple of item indices
        binary_matrix: (N x M) binary matrix of transactions
    Returns:
        prob: Empirical co-occurrence probability
    """
    N = binary_matrix.shape[0]
    item_indices = list(itemset)
    present = np.all(binary_matrix[:, item_indices], axis=1)
    prob = np.sum(present) / N
    return prob

# Function to get current memory usage
def get_memory_usage():
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024  # Convert bytes to MB
    return round(memory_mb, 3)

# MAB-ARM algorithm
def mab_arm(binary_matrix, min_prob, min_conf, T_max, m_min, m_max):
    """
    Multi-Armed Bandit Association Rule Mining (MAB-ARM).
    Parameters:
        binary_matrix: (N x M) binary matrix of transactions
        min_prob: Minimum support threshold
        min_conf: Minimum confidence threshold
        T_max: Maximum number of itemset evaluations
        m_min: Minimum itemset size
        m_max: Maximum itemset size
    Returns:
        rules: List of (antecedent, consequent, support, confidence, lift) tuples
        frequent_itemsets: List of frequent itemsets
        n_I: Dictionary of evaluation counts for all itemsets
    """
    N = binary_matrix.shape[0]
    M = binary_matrix.shape[1]
    
    # Initialize candidate itemsets
    candidate_itemsets = []
    for m in range(m_min, m_max + 1):
        candidate_itemsets.extend(combinations(range(M), m))
    candidate_itemsets = [tuple(itemset) for itemset in candidate_itemsets]
    
    # Initialize counters and estimates
    n_I = {itemset: 0 for itemset in candidate_itemsets}  # Number of evaluations
    p_I = {itemset: 0.0 for itemset in candidate_itemsets}  # Estimated probability
    
    # Store frequent itemsets and rules
    frequent_itemsets = []
    rules = []
    
    # Main MAB loop
    for t in range(1, T_max + 1):
        # Compute UCB scores for each I ∈ ℐ
        ucb_scores = {}
        for itemset in candidate_itemsets:
            if n_I[itemset] == 0:
                ucb_scores[itemset] = float('inf')
            else:
                exploration = math.sqrt(2 * math.log(t) / n_I[itemset])
                ucb_scores[itemset] = p_I[itemset] + exploration
        
        # Select itemset with highest UCB score
        I_star = max(ucb_scores, key=ucb_scores.get)
        
        # Update evaluation count and estimate probability
        n_I[I_star] += 1
        p_I[I_star] = cooccurrence_prob(I_star, binary_matrix)
        
        # Associative probability update for supersets within candidate itemsets
        I_star_set = set(I_star)
        for itemset in candidate_itemsets:
            if I_star_set.issubset(itemset) and len(itemset) > len(I_star):
                if p_I[itemset] < p_I[I_star]:
                    p_I[itemset] = p_I[I_star]  # Update underestimated probabilities
        
        # Check if itemset is frequent
        if p_I[I_star] > min_prob:
            if I_star not in frequent_itemsets:
                frequent_itemsets.append(I_star)
            
            # Generate rules
            m = len(I_star)
            for r in range(1, m):
                for antecedent_tuple in combinations(I_star, r):
                    antecedent = set(antecedent_tuple)
                    consequent = set(I_star) - antecedent
                    p_A = cooccurrence_prob(tuple(antecedent), binary_matrix)
                    p_B = cooccurrence_prob(tuple(consequent), binary_matrix)
                    if p_A > 0:
                        conf = p_I[I_star] / p_A
                        if conf > min_conf:
                            lift = p_I[I_star] / (p_A * p_B) if p_B > 0 else 0
                            rule = (antecedent, consequent, p_I[I_star], conf, lift)
                            rules.append(rule)
        
        # Optionally prune ℐ: remove itemsets with low UCB_I(t) or p_I after sufficient evaluations
        if t % 100 == 0:  # Prune every 100 iterations
            min_ucb = min(ucb_scores.values())
            candidate_itemsets = [I for I in candidate_itemsets if ucb_scores[I] > min_ucb or n_I[I] < 10]
    
    return rules, frequent_itemsets, n_I

# Experiment parameters
min_prob_values = [0.1, 0.2, 0.3, 0.4, 0.5]
min_conf = 0.5
T_max = 2**num_items  # originally 1000, maximum itemset evaluations
m_min = 2  # Minimum itemset size
m_max = 10  # Maximum itemset size (to manage complexity)

# Initialize lists to store metrics
mabarm_runtimes, mabarm_memory, mabarm_num_rules, mabarm_num_itemsets = [], [], [], []

for min_prob in tqdm(min_prob_values, desc="Processing min_prob values"):
    start_time = time.time()
    start_memory = get_memory_usage()
    
    # Run MAB-ARM
    mabarm_rules, mabarm_frequent_itemsets, n_I = mab_arm(
        binary_matrix, min_prob, min_conf, T_max, m_min, m_max
    )
    
    end_time = time.time()
    end_memory = get_memory_usage()
    
    # Store metrics
    mabarm_runtimes.append(end_time - start_time)
    mabarm_memory.append(max(end_memory - start_memory, 0))
    mabarm_num_rules.append(len(mabarm_rules))
    mabarm_num_itemsets.append(len(mabarm_frequent_itemsets))
    
    # Save rule summary
    mabarm_summary_data = []
    for i, (antecedent, consequent, support, conf, lift) in enumerate(mabarm_rules, 1):
        antecedent_items = ', '.join([f'item_{idx}' for idx in sorted(list(antecedent))])
        consequent_items = ', '.join([f'item_{idx}' for idx in sorted(list(consequent))])
        mabarm_summary_data.append({
            '#': i,
            'Premises': antecedent_items,
            'Conclusion': consequent_items,
            'Support': support,
            'Confidence': conf,
            'Lift': lift
        })
    mabarm_summary_df = pd.DataFrame(mabarm_summary_data)
    if not mabarm_summary_df.empty:
        mabarm_summary_df.to_csv(os.path.join(output_dir, f"mabarm_summary_min_prob_{min_prob}.csv"), index=False)
    else:
        print(f"No MAB-ARM rules found for min_prob={min_prob}. Skipping summary save.")
    
    # Save metrics for this min_prob
    metrics_data = {
        'Min_Prob': [min_prob],
        'Algorithm': ['MAB-ARM'],
        'Runtime (s)': [mabarm_runtimes[-1]],
        'Memory Usage (MB)': [mabarm_memory[-1]],
        'Number of Frequent Itemsets': [mabarm_num_itemsets[-1]],
        'Number of Rules': [mabarm_num_rules[-1]]
    }
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.to_csv(os.path.join(output_dir, f"metrics_min_prob_{min_prob}.csv"), index=False)
    
    # Analysis of top rules with item 2 as RHS
    def get_top_rules_mabarm(rules, j, n=10):
        filtered_rules = [rule for rule in rules if len(rule[1]) == 1 and list(rule[1])[0] == j]
        sorted_rules = sorted(filtered_rules, key=lambda x: x[3], reverse=True)  # Sort by confidence
        return sorted_rules[:n]
    
    RHS_item_no = 2
    top_n = 10
    top_mabarm_rules = get_top_rules_mabarm(mabarm_rules, RHS_item_no, top_n)
    mabarm_analysis_data = []
    if top_mabarm_rules:
        for rule in top_mabarm_rules:
            antecedent, consequent, support, conf, lift = rule
            full_itemset = sorted(list(antecedent) + list(consequent))
            supporting_indices = [i for i, t in enumerate(transactions) if set(full_itemset).issubset(set(t))]
            support_count = len(supporting_indices)
            example_indices = supporting_indices[:3] if len(supporting_indices) >= 3 else supporting_indices
            mabarm_analysis_data.append({
                'Rule': f"{sorted(list(antecedent))} -> {sorted(list(consequent))}",
                'Confidence': conf,
                'Support (prob)': support,
                'Support Count': support_count,
                'Example Transaction Indices': example_indices,
                'Example Transactions': [transactions[xx] for xx in example_indices]
            })
        mabarm_analysis_df = pd.DataFrame(mabarm_analysis_data)
        mabarm_analysis_df.to_csv(os.path.join(output_dir, f"mabarm_analysis_min_prob_{min_prob}.csv"), index=False)
    else:
        print(f"No MAB-ARM rules found with item {RHS_item_no} on RHS for min_prob={min_prob}.")

# Save final summary of metrics
final_metrics_data = {
    'Min_Prob': min_prob_values,
    'MABARM_Runtime (s)': mabarm_runtimes,
    'MABARM_Memory (MB)': mabarm_memory,
    'MABARM_Frequent_Itemsets': mabarm_num_itemsets,
    'MABARM_Rules': mabarm_num_rules
}
final_metrics_df = pd.DataFrame(final_metrics_data)
final_metrics_df.to_csv(os.path.join(output_dir, "final_metrics_summary_mabarm.csv"), index=False)

# Print final metrics
print("\nNumber of frequent itemsets (MAB-ARM):", mabarm_num_itemsets)
print("Number of rules (MAB-ARM):", mabarm_num_rules)


# Plot the counters n_I
plt.figure(figsize=(12, 6))
itemsets = list(n_I.keys())
counts = list(n_I.values())
# plt.bar(range(len(counts)), counts, tick_label=[str(i) for i in itemsets])
plt.bar(range(len(counts)), counts)
plt.xlabel('Itemset indices')
plt.ylabel('Number of Evaluations ($n_I$)')
plt.title('Evaluation Counts for Itemsets')
# plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "n_I_evaluation_counts.png"))
plt.show()
