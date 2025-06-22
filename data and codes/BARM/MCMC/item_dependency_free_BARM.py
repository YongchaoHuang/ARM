import numpy as np
import pandas as pd
import os
import time
import psutil
# from google.colab import drive
import pymc as pm
print(pm.__version__)
import matplotlib.pyplot as plt
from itertools import combinations
from tqdm import tqdm
import pytensor.tensor as at  # Import pytensor for symbolic operations

# # Mount Google Drive
# drive.mount('/content/drive')

# Define the output directory in Google Drive
output_dir = "~/BARM/results"
os.makedirs(output_dir, exist_ok=True)

# Load the saved dataset
X_df = pd.read_csv(os.path.join(output_dir, "feature_matrix.csv"))
X = X_df.values  # Convert to numpy array
num_items = X.shape[0]
num_features = X.shape[1]

transactions_df = pd.read_csv(os.path.join(output_dir, "synthetic_transactions.csv"))
transactions = [eval(t) for t in transactions_df['Transaction']]  # Parse string representation
print(f'Transactions 0-5: \n {transactions[:5]}')

# Convert transactions to binary matrix
def transactions_to_binary_matrix(transactions, num_items):
    binary_matrix = np.zeros((len(transactions), num_items), dtype=bool)
    for i, t in enumerate(transactions):
        binary_matrix[i, t] = True
    return binary_matrix

binary_matrix = transactions_to_binary_matrix(transactions, num_items)
T_df = pd.DataFrame(binary_matrix, columns=[f'item_{i}' for i in range(num_items)])

# MCMC sampling for posterior inference (simplified without correlation)
def sample_posterior(transactions, alpha, beta, S_MCMC):
    with pm.Model() as model:
        # Priors
        p = pm.Beta('p', alpha=alpha, beta=beta, shape=num_items)

        # Likelihood (independent Bernoulli)
        obs = pm.Bernoulli('obs', p=p, observed=binary_matrix)

        # MCMC sampling
        trace = pm.sample(S_MCMC, tune=1000, return_inferencedata=False, target_accept=0.9)

    # Extract sampled values as numpy arrays
    return trace['p']

# Co-occurrence probability estimation (simplified without correlation)
def cooccurrence_prob(itemset, p_samples, S):
    m = len(itemset)
    prob = 0.0
    item_indices = list(itemset)

    # Calculate the joint probability for the itemset
    for s in range(S):
        p_s = p_samples[s]
        joint_prob = np.prod(p_s[item_indices])
        prob += joint_prob
    return prob / S

# Function to get current memory usage
def get_memory_usage():
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024  # Convert bytes to MB
    return round(memory_mb, 3)

# Run BARM with varying min_prob thresholds
min_prob_values = [0.1, 0.2, 0.3, 0.4, 0.5]
min_conf = 0.5
S_MCMC = 1000  # Number of MCMC samples
S = S_MCMC  # Number of posterior samples for probability estimation
alpha = np.ones(num_items)  # Uniform prior
beta = np.ones(num_items)

# Initialize lists to store metrics
barm_runtimes, barm_memory, barm_num_rules, barm_num_itemsets = [], [], [], []

for min_prob in tqdm(min_prob_values, desc="Processing min_prob values"):
    start_time = time.time()
    start_memory = get_memory_usage()

    # MCMC sampling
    p_samples = sample_posterior(transactions, alpha, beta, S_MCMC)

    # Rule mining
    barm_rules = []
    barm_frequent_itemsets = []
    max_itemset_size = min(num_items + 1, num_items + 1)  # If needed, limit the maximum itemset size
    print(f"Maximum itemset size considered: {max_itemset_size - 1}")

    for m in range(2, max_itemset_size):  # Iterate up to the maximum size
        print(f"Processing itemsets of size {m}...")
        # Use combinations on range(num_items) which are integers
        itemsets = list(combinations(range(num_items), m))
        print(f"Generated {len(itemsets)} itemsets of size {m}.")
        for itemset in tqdm(itemsets, desc=f"Itemsets size {m}"):  # Add progress bar for itemsets
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
                        if p_A > 0 and p_B > 0:
                            conf = p_I / p_A
                            if conf >= min_conf:
                                # Calculate lift
                                lift = p_I / (p_A * p_B) if p_A * p_B != 0 else 0
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
        # Convert sets back to sorted lists for consistent output
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
    # Handle case where no rules are found
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
        # Filter rules where the consequent set contains exactly one item, which is j
        filtered_rules = [rule for rule in rules if len(rule[1]) == 1 and list(rule[1])[0] == j]
        sorted_rules = sorted(filtered_rules, key=lambda x: x[3], reverse=True)  # Sort by confidence
        return sorted_rules[:n]

    RHS_item_no = 2  # Index of the item appearing on RHS
    top_n = 10
    # Pass the potentially empty list of rules to the function
    top_barm_rules = get_top_rules_barm(barm_rules, RHS_item_no, top_n)
    barm_analysis_data = []
    print(f"\n Top {top_n} BARM rules with item {RHS_item_no} on RHS (min_prob={min_prob}):")
    if top_barm_rules:  # Only process if there are top rules
        for rule in top_barm_rules:
            antecedent, consequent, support, conf, lift = rule
            # Convert sets back to sorted lists for display and transaction checking
            full_itemset = sorted(list(antecedent) + list(consequent))
            # Check for full itemset in original transactions
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
