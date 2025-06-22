import numpy as np
from scipy.stats import multivariate_normal
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
from itertools import combinations
import time
import psutil
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
import pandas as pd
import io
import os
from tqdm import tqdm

np.random.seed(111)

# Define the output directory
output_dir = "~/Synthetic datasets/Synthetic 1"
os.makedirs(output_dir, exist_ok=True)

# Load the saved dataset
# Load feature matrix X
X_df = pd.read_csv(os.path.join(output_dir, "feature_matrix.csv"))
X = X_df.values  # Convert to numpy array
num_items = X.shape[0]
num_features = X.shape[1]

# Load transactions
transactions_df = pd.read_csv(os.path.join(output_dir, "synthetic_transactions.csv"))
transactions = [eval(t) for t in transactions_df['Transaction']]  # Parse string representation of lists back to lists
print(f'Transactions 0-5: \n {transactions[:5]}')

# Convert transactions to binary matrix for Apriori, FP-Growth, and Eclat (First Script)
def transactions_to_binary_matrix(transactions, num_items):
    binary_matrix = np.zeros((len(transactions), num_items), dtype=bool)
    for i, t in enumerate(transactions):
        binary_matrix[i, t] = True
    return binary_matrix

binary_matrix = transactions_to_binary_matrix(transactions, num_items)
T_df = pd.DataFrame(binary_matrix, columns=[f'item_{i}' for i in range(num_items)])
items = [f'item_{i}' for i in range(num_items)]  # Define items for Eclat

pd.DataFrame(binary_matrix, columns=[f'item_{i}' for i in range(num_items)]).to_csv(os.path.join(output_dir, "synthetic_transactions_binaryMatrix.csv"), index=False)

# Shifted RBF kernel for GPAR
def shifted_rbf_kernel(X, length_scale, d0):
    """
    Shifted RBF kernel: k(x_i, x_j) = exp(-(||x_i - x_j|| - d0)^2 / (2 * length_scale^2))
    Parameters:
    - X: Feature matrix (M x d)
    - length_scale: Length scale parameter (ell)
    - d0: Optimal distance hyperparameter
    Returns:
    - Kernel matrix K (M x M)
    """
    # Compute pairwise Euclidean distances (not squared yet)
    dists = squareform(pdist(X, 'euclidean'))
    # Shifted distance: (||x_i - x_j|| - d0)^2
    shifted_dists = (dists - d0) ** 2
    # Compute the kernel matrix
    K = np.exp(-shifted_dists / (2 * length_scale ** 2))
    return K

def make_psd(K, epsilon=1e-6):
    """
    Convert a non-PSD kernel matrix to PSD using eigen-decomposition.
    Parameters:
    - K: Kernel matrix (M x M)
    - epsilon: Small positive value to ensure numerical stability
    Returns:
    - K_psd: PSD kernel matrix (M x M)
    """
    # Perform eigen-decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(K)
    # Retain only positive eigenvalues
    positive_indices = eigenvalues > 0
    eigenvalues_positive = eigenvalues[positive_indices]
    eigenvectors_positive = eigenvectors[:, positive_indices]
    # Add small epsilon to positive eigenvalues for numerical stability
    eigenvalues_positive = eigenvalues_positive + epsilon
    # Reconstruct PSD kernel matrix: K_psd = Q * Lambda * Q^T
    K_psd = eigenvectors_positive @ np.diag(eigenvalues_positive) @ eigenvectors_positive.T
    # Ensure symmetry and numerical stability
    K_psd = (K_psd + K_psd.T) / 2
    return K_psd

# GPAR: Log-likelihood for GP parameter estimation (Updated to optimize both length_scale and d0)
def neg_log_likelihood(params, X, transactions):
    length_scale, d0 = params  # Unpack length_scale and d0
    # Use shifted RBF kernel with d0 and make PSD
    K = shifted_rbf_kernel(X, length_scale, d0)
    K = make_psd(K)  # Ensure the kernel matrix is PSD
    K = K + 1e-6 * np.eye(num_items)  # Add small diagonal for numerical stability
    log_lik = 0
    for t in transactions:
        z_mean = np.zeros(num_items)
        z_sample = np.zeros(num_items)
        z_sample[t] = 1  # Approximate: set z_i > 0 as 1 for simplicity
        log_lik += multivariate_normal.logpdf(z_sample, mean=z_mean, cov=K)
    return -log_lik

# GPAR: Co-occurrence probability (Second Script)
def monte_carlo_prob(K, itemset, S=100):
    samples = np.random.normal(0, 1, (S, len(itemset)))
    item_indices = list(itemset)
    K_subset = K[np.ix_(item_indices, item_indices)]
    L = np.linalg.cholesky(K_subset + np.eye(len(itemset)) * 1e-6)
    z = np.dot(L, samples.T).T
    z_binary = (z > 0).astype(int)
    prob = np.mean(np.all(z_binary == 1, axis=1))
    return prob

def compute_lift(K, antecedent, consequent, p_I):
    p_A = monte_carlo_prob(K, antecedent)
    p_B = monte_carlo_prob(K, consequent)
    if p_A * p_B == 0:
        return 0
    return p_I / (p_A * p_B)

# Eclat Implementation (Second Script)
def eclat(transactions, min_support, items):
    vertical_db = {item: set() for item in items}
    for tid, row in transactions.iterrows():
        for item in items:
            if row[item] == 1:
                vertical_db[item].add(tid)

    N = len(transactions)
    min_count = min_support * N
    frequent_itemsets = []
    itemsets_dict = {}

    for item, tids in vertical_db.items():
        support = len(tids)
        if support >= min_count:
            frequent_itemsets.append((frozenset([item]), support / N))
            itemsets_dict[frozenset([item])] = tids

    def eclat_recursive(prefix, items, min_count):
        nonlocal frequent_itemsets, itemsets_dict
        for i, item1 in enumerate(items):
            new_prefix = prefix | frozenset([item1])
            tids1 = itemsets_dict[frozenset([item1])]
            new_items = []
            new_tids = {}

            for item2 in items[i+1:]:
                tids2 = itemsets_dict[frozenset([item2])]
                intersection = tids1 & tids2
                if len(intersection) >= min_count:
                    new_items.append(item2)
                    new_tids[item2] = intersection

            support = len(tids1) / N if len(new_prefix) == 1 else len(itemsets_dict[new_prefix]) / N
            frequent_itemsets.append((new_prefix, support))
            for item, tids in new_tids.items():
                itemsets_dict[new_prefix | frozenset([item])] = tids

            if new_items:
                eclat_recursive(new_prefix, new_items, min_count)

    initial_items = [item for item, tids in vertical_db.items() if len(tids) >= min_count]
    eclat_recursive(frozenset(), initial_items, min_count)

    freq_df = pd.DataFrame(frequent_itemsets, columns=['itemsets', 'support'])
    freq_df['itemsets'] = freq_df['itemsets'].apply(lambda x: set(x))
    return freq_df

# Function to get current memory usage (Second Script)
def get_memory_usage():
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024  # Convert bytes to MB
    return round(memory_mb, 3)  # Round to 3 decimal places

# Run all algorithms with varying min_support thresholds (Second Script)
min_support_values = [0.1, 0.2, 0.3, 0.4, 0.5]
min_conf = 0.5

# Initialize lists to store metrics (Second Script)
gpar_runtimes, gpar_memory, gpar_num_rules, gpar_num_itemsets = [], [], [], []
apriori_runtimes, apriori_memory, apriori_num_rules, apriori_num_itemsets = [], [], [], []
fpgrowth_runtimes, fpgrowth_memory, fpgrowth_num_rules, fpgrowth_num_itemsets = [], [], [], []
eclat_runtimes, eclat_memory, eclat_num_rules, eclat_num_itemsets = [], [], [], []

# Optimize length scale and d0 for GPAR once
initial_params = [1.0, 1.0]  # [length_scale, d0]
bounds = [(0.1, 100.0), (0.0, 100.0)]  # Bounds for length_scale and d0
result = minimize(neg_log_likelihood, initial_params, args=(X, transactions), bounds=bounds)
length_scale, d0 = result.x
print(f"Optimized length_scale: {length_scale}, Optimized d0: {d0}")
# Compute the kernel matrix with the shifted RBF kernel and make it PSD
K = shifted_rbf_kernel(X, length_scale, d0)
K = make_psd(K)

for min_support in tqdm(min_support_values, desc="Processing min_support values"):
    # GPAR (Adapted from Second Script, preserving First Script's style)
    start_time = time.time()
    start_memory = get_memory_usage()

    gpar_rules = []
    gpar_frequent_itemsets = []
    m = 2
    while True:
        itemsets = list(combinations(range(num_items), m))
        found_rule = False
        for itemset in itemsets:
            p_I = monte_carlo_prob(K, itemset)
            if p_I >= min_support:
                gpar_frequent_itemsets.append(itemset)
                for r in range(1, m):
                    for antecedent in combinations(itemset, r):
                        antecedent = set(antecedent)
                        consequent = set(itemset) - antecedent
                        p_A = monte_carlo_prob(K, antecedent)
                        if p_A > 0:
                            conf = p_I / p_A
                            if conf >= min_conf:
                                lift = compute_lift(K, antecedent, consequent, p_I)
                                rule = (antecedent, consequent, p_I, conf, lift)
                                gpar_rules.append(rule)
                                found_rule = True
        if not found_rule:
            break
        m += 1

    end_time = time.time()
    end_memory = get_memory_usage()

    gpar_runtimes.append(end_time - start_time)
    gpar_memory.append(max(end_memory - start_memory, 0))
    gpar_num_rules.append(len(gpar_rules))
    gpar_num_itemsets.append(len(gpar_frequent_itemsets))

    # Save GPAR summary table for this min_support (Second Script)
    gpar_summary_data = []
    for i, (antecedent, consequent, support, conf, lift) in enumerate(gpar_rules, 1):
        antecedent_items = ', '.join([f'item_{idx}' for idx in antecedent])
        consequent_items = ', '.join([f'item_{idx}' for idx in consequent])
        gpar_summary_data.append({
            '#': i,
            'Premises': antecedent_items,
            'Conclusion': consequent_items,
            'Support': support,
            'Confidence': conf,
            'Lift': lift
        })
    gpar_summary_df = pd.DataFrame(gpar_summary_data)
    gpar_summary_df.to_csv(os.path.join(output_dir, f"gpar_summary_min_support_{min_support}.csv"), index=False)

    # Apriori (Second Script, preserving First Script's style)
    start_time = time.time()
    start_memory = get_memory_usage()

    apriori_freq = apriori(T_df, min_support=min_support, use_colnames=True)
    apriori_rules = association_rules(apriori_freq, metric="confidence", min_threshold=min_conf)

    end_time = time.time()
    end_memory = get_memory_usage()

    apriori_runtimes.append(end_time - start_time)
    apriori_memory.append(max(end_memory - start_memory, 0))
    apriori_num_rules.append(len(apriori_rules))
    apriori_num_itemsets.append(len(apriori_freq))

    # Save Apriori summary table for this min_support
    apriori_summary_data = []
    for i, row in apriori_rules.iterrows():
        antecedent = ', '.join(row['antecedents'])
        consequent = ', '.join(row['consequents'])
        support = row['support']
        conf = row['confidence']
        lift = row['lift']
        apriori_summary_data.append({
            '#': i + 1,
            'Premises': antecedent,
            'Conclusion': consequent,
            'Support': support,
            'Confidence': conf,
            'Lift': lift
        })
    apriori_summary_df = pd.DataFrame(apriori_summary_data)
    apriori_summary_df.to_csv(os.path.join(output_dir, f"apriori_summary_min_support_{min_support}.csv"), index=False)

    # FP-Growth (Second Script)
    start_time = time.time()
    start_memory = get_memory_usage()

    fpgrowth_freq = fpgrowth(T_df, min_support=min_support, use_colnames=True)
    fpgrowth_rules = association_rules(fpgrowth_freq, metric="confidence", min_threshold=min_conf)

    end_time = time.time()
    end_memory = get_memory_usage()

    fpgrowth_runtimes.append(end_time - start_time)
    fpgrowth_memory.append(max(end_memory - start_memory, 0))
    fpgrowth_num_rules.append(len(fpgrowth_rules))
    fpgrowth_num_itemsets.append(len(fpgrowth_freq))

    # Save FP-Growth summary table for this min_support
    fpgrowth_summary_data = []
    for i, row in fpgrowth_rules.iterrows():
        antecedent = ', '.join(row['antecedents'])
        consequent = ', '.join(row['consequents'])
        support = row['support']
        conf = row['confidence']
        lift = row['lift']
        fpgrowth_summary_data.append({
            '#': i + 1,
            'Premises': antecedent,
            'Conclusion': consequent,
            'Support': support,
            'Confidence': conf,
            'Lift': lift
        })
    fpgrowth_summary_df = pd.DataFrame(fpgrowth_summary_data)
    fpgrowth_summary_df.to_csv(os.path.join(output_dir, f"fpgrowth_summary_min_support_{min_support}.csv"), index=False)

    # Eclat (Second Script)
    start_time = time.time()
    start_memory = get_memory_usage()

    eclat_freq = eclat(T_df, min_support, items)
    eclat_rules = association_rules(eclat_freq, metric="confidence", min_threshold=min_conf)

    end_time = time.time()
    end_memory = get_memory_usage()

    eclat_runtimes.append(end_time - start_time)
    eclat_memory.append(max(end_memory - start_memory, 0))
    eclat_num_rules.append(len(eclat_rules))
    eclat_num_itemsets.append(len(eclat_freq))

    # Save Eclat summary table for this min_support
    eclat_summary_data = []
    for i, row in eclat_rules.iterrows():
        antecedent = ', '.join(row['antecedents'])
        consequent = ', '.join(row['consequents'])
        support = row['support']
        conf = row['confidence']
        lift = row['lift']
        eclat_summary_data.append({
            '#': i + 1,
            'Premises': antecedent,
            'Conclusion': consequent,
            'Support': support,
            'Confidence': conf,
            'Lift': lift
        })
    eclat_summary_df = pd.DataFrame(eclat_summary_data)
    eclat_summary_df.to_csv(os.path.join(output_dir, f"eclat_summary_min_support_{min_support}.csv"), index=False)

    # Save key numbers (metrics) for this iteration (Second Script)
    metrics_data = {
        'Min_Support': [min_support] * 4,
        'Algorithm': ['GPAR', 'Apriori', 'FP-Growth', 'Eclat'],
        'Runtime (s)': [gpar_runtimes[-1], apriori_runtimes[-1], fpgrowth_runtimes[-1], eclat_runtimes[-1]],
        'Memory Usage (MB)': [gpar_memory[-1], apriori_memory[-1], fpgrowth_memory[-1], eclat_memory[-1]],
        'Number of Frequent Itemsets': [gpar_num_itemsets[-1], apriori_num_itemsets[-1], fpgrowth_num_itemsets[-1], eclat_num_itemsets[-1]],
        'Number of Rules': [gpar_num_rules[-1], apriori_num_rules[-1], fpgrowth_num_rules[-1], eclat_num_rules[-1]]
    }
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.to_csv(os.path.join(output_dir, f"metrics_min_support_{min_support}.csv"), index=False)

    # Plotting and saving figures for this iteration (Second Script)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(min_support_values[:len(gpar_runtimes)], gpar_runtimes, marker='o', label='GPAR')
    plt.plot(min_support_values[:len(apriori_runtimes)], apriori_runtimes, marker='s', label='Apriori')
    plt.plot(min_support_values[:len(fpgrowth_runtimes)], fpgrowth_runtimes, marker='^', label='FP-Growth')
    plt.plot(min_support_values[:len(eclat_runtimes)], eclat_runtimes, marker='d', label='Eclat')
    plt.xlabel('Minimum support threshold')
    plt.ylabel('Runtime (s)')
    plt.title('Runtime over minimum support threshold')
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(min_support_values[:len(gpar_memory)], gpar_memory, marker='o', label='GPAR')
    plt.plot(min_support_values[:len(apriori_memory)], apriori_memory, marker='s', label='Apriori')
    plt.plot(min_support_values[:len(fpgrowth_memory)], fpgrowth_memory, marker='^', label='FP-Growth')
    plt.plot(min_support_values[:len(eclat_memory)], eclat_memory, marker='d', label='Eclat')
    plt.xlabel('Minimum support threshold')
    plt.ylabel('Memory Usage (MB)')
    plt.title('Memory usage over minimum support threshold')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"performance_metrics_min_support_{min_support}.png"))
    plt.close()

    ##### Analysis of Results (First Script, Extended to All Algorithms) #####
    #### Extract the top rules that contain certain item on the RHS
    def get_top_rules_gpar(rules, j, n=10):
        filtered_rules = [rule for rule in rules if list(rule[1])[0] == j]
        sorted_rules = sorted(filtered_rules, key=lambda x: x[3], reverse=True)  # Sort by confidence
        return sorted_rules[:n]

    def get_top_rules_apriori(rules_df, j, n=10):
        filtered_rules = rules_df[rules_df['consequents'].apply(lambda x: list(x) == [f'item_{j}'])]
        sorted_rules = filtered_rules.sort_values(by='confidence', ascending=False)
        return sorted_rules.head(n)

    def get_supporting_transactions(itemset, transactions):
        itemset_set = set(itemset)
        supporting_indices = [i for i, t in enumerate(transactions) if itemset_set.issubset(set(t))]
        return supporting_indices

    RHS_item_no = 2  # Index of the item appearing on RHS
    top_n = 10  # Number of top rules

    # GPAR Analysis
    top_gpar_rules = get_top_rules_gpar(gpar_rules, RHS_item_no, top_n)
    gpar_analysis_data = []
    print(f"\n Top {top_n} GPAR rules with item {RHS_item_no} on RHS (min_support={min_support}):")
    for rule in top_gpar_rules:
        antecedent, consequent, support, conf, lift = rule
        full_itemset = list(antecedent) + list(consequent)
        supporting_indices = get_supporting_transactions(full_itemset, transactions)
        support_count = len(supporting_indices)
        example_indices = supporting_indices[:3] if len(supporting_indices) >= 3 else supporting_indices
        gpar_analysis_data.append({
            'Rule': f"{list(antecedent)} -> {list(consequent)}",
            'Confidence': conf,
            'Support (prob)': support,
            'Support Count': support_count,
            'Example Transaction Indices': example_indices,
            'Example Transactions': [transactions[xx] for xx in example_indices]
        })
    gpar_analysis_df = pd.DataFrame(gpar_analysis_data)
    gpar_analysis_df.to_csv(os.path.join(output_dir, f"gpar_analysis_min_support_{min_support}.csv"), index=False)

    # Apriori Analysis
    top_apriori_rules = get_top_rules_apriori(apriori_rules, RHS_item_no, top_n)
    apriori_analysis_data = []
    print(f"\n Top {top_n} Apriori rules with item {RHS_item_no} on RHS (min_support={min_support}):")
    for idx, rule in top_apriori_rules.iterrows():
        full_itemset = list(rule['antecedents']) + list(rule['consequents'])
        full_itemset = [int(item.split('_')[1]) for item in full_itemset]
        supporting_indices = get_supporting_transactions(full_itemset, transactions)
        support_count = len(supporting_indices)
        example_indices = supporting_indices[:3] if len(supporting_indices) >= 3 else supporting_indices
        apriori_analysis_data.append({
            'Rule': f"{list(rule['antecedents'])} -> {list(rule['consequents'])}",
            'Confidence': rule['confidence'],
            'Support': rule['support'],
            'Support Count': support_count,
            'Example Transaction Indices': example_indices,
            'Example Transactions': [transactions[xx] for xx in example_indices]
        })
    apriori_analysis_df = pd.DataFrame(apriori_analysis_data)
    apriori_analysis_df.to_csv(os.path.join(output_dir, f"apriori_analysis_min_support_{min_support}.csv"), index=False)

    # FP-Growth Analysis
    top_fpgrowth_rules = get_top_rules_apriori(fpgrowth_rules, RHS_item_no, top_n)
    fpgrowth_analysis_data = []
    print(f"\n Top {top_n} FP-Growth rules with item {RHS_item_no} on RHS (min_support={min_support}):")
    for idx, rule in top_fpgrowth_rules.iterrows():
        full_itemset = list(rule['antecedents']) + list(rule['consequents'])
        full_itemset = [int(item.split('_')[1]) for item in full_itemset]
        supporting_indices = get_supporting_transactions(full_itemset, transactions)
        support_count = len(supporting_indices)
        example_indices = supporting_indices[:3] if len(supporting_indices) >= 3 else supporting_indices
        fpgrowth_analysis_data.append({
            'Rule': f"{list(rule['antecedents'])} -> {list(rule['consequents'])}",
            'Confidence': rule['confidence'],
            'Support': rule['support'],
            'Support Count': support_count,
            'Example Transaction Indices': example_indices,
            'Example Transactions': [transactions[xx] for xx in example_indices]
        })
    fpgrowth_analysis_df = pd.DataFrame(fpgrowth_analysis_data)
    fpgrowth_analysis_df.to_csv(os.path.join(output_dir, f"fpgrowth_analysis_min_support_{min_support}.csv"), index=False)

    # Eclat Analysis
    top_eclat_rules = get_top_rules_apriori(eclat_rules, RHS_item_no, top_n)
    eclat_analysis_data = []
    print(f"\n Top {top_n} Eclat rules with item {RHS_item_no} on RHS (min_support={min_support}):")
    for idx, rule in top_eclat_rules.iterrows():
        full_itemset = list(rule['antecedents']) + list(rule['consequents'])
        full_itemset = [int(item.split('_')[1]) for item in full_itemset]
        supporting_indices = get_supporting_transactions(full_itemset, transactions)
        support_count = len(supporting_indices)
        example_indices = supporting_indices[:3] if len(supporting_indices) >= 3 else supporting_indices
        eclat_analysis_data.append({
            'Rule': f"{list(rule['antecedents'])} -> {list(rule['consequents'])}",
            'Confidence': rule['confidence'],
            'Support': rule['support'],
            'Support Count': support_count,
            'Example Transaction Indices': example_indices,
            'Example Transactions': [transactions[xx] for xx in example_indices]
        })
    eclat_analysis_df = pd.DataFrame(eclat_analysis_data)
    eclat_analysis_df.to_csv(os.path.join(output_dir, f"eclat_analysis_min_support_{min_support}.csv"), index=False)

# Save final summary of metrics (Second Script)
final_metrics_data = {
    'Min_Support': min_support_values,
    'GPAR_Runtime (s)': gpar_runtimes,
    'Apriori_Runtime (s)': apriori_runtimes,
    'FP-Growth_Runtime (s)': fpgrowth_runtimes,
    'Eclat_Runtime (s)': eclat_runtimes,
    'GPAR_Memory (MB)': gpar_memory,
    'Apriori_Memory (MB)': apriori_memory,
    'FP-Growth_Memory (MB)': fpgrowth_memory,
    'Eclat_Memory (MB)': eclat_memory,
    'GPAR_Frequent_Itemsets': gpar_num_itemsets,
    'Apriori_Frequent_Itemsets': apriori_num_itemsets,
    'FP-Growth_Frequent_Itemsets': fpgrowth_num_itemsets,
    'Eclat_Frequent_Itemsets': eclat_num_itemsets,
    'GPAR_Rules': gpar_num_rules,
    'Apriori_Rules': apriori_num_rules,
    'FP-Growth_Rules': fpgrowth_num_rules,
    'Eclat_Rules': eclat_num_rules
}
final_metrics_df = pd.DataFrame(final_metrics_data)
final_metrics_df.to_csv(os.path.join(output_dir, "final_metrics_summary.csv"), index=False)

# Print final metrics (Second Script)
print("\nNumber of frequent itemsets:")
print(f"GPAR: {gpar_num_itemsets}")
print(f"Apriori: {apriori_num_itemsets}")
print(f"FP-Growth: {fpgrowth_num_itemsets}")
print(f"Eclat: {eclat_num_itemsets}")

print("\nNumber of rules:")
print(f"GPAR: {gpar_num_rules}")
print(f"Apriori: {apriori_num_rules}")
print(f"FP-Growth: {fpgrowth_num_rules}")
print(f"Eclat: {eclat_num_rules}")
