import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize
from itertools import combinations
import time
import psutil
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
import io
import os

np.random.seed(111)

# Define the output directory
output_dir = "~/UK Accidents dataset"
os.makedirs(output_dir, exist_ok=True)

# Function to get current memory usage
def get_memory_usage():
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024  # Convert bytes to MB
    return round(memory_mb, 3)  # Round to 3 decimal places

# Load the data
data = pd.read_csv(output_dir + "/Accidents_fatal_serious.csv")
# data = data.head(100)

# Preprocess the data
# Check for missing columns
item_columns = ['Accident_Severity', 'Road_Type', 'Weather_Conditions', 'Light_Conditions',
                'Road_Surface_Conditions', 'Junction_Detail', 'Urban_or_Rural_Area']
feature_columns = ['Longitude', 'Latitude', 'Speed_limit', 'Number_of_Vehicles',
                   'Day_of_Week', 'Hour_of_Day', 'Junction_Control', 'Pedestrian_Crossing-Human_Control',
                   'Pedestrian_Crossing-Physical_Facilities', '1st_Road_Class']

# Check for missing values
if data[item_columns + [col for col in feature_columns if col != 'Hour_of_Day']].isna().any().any():
    print("Warning: Missing values detected. Dropping rows with missing values.")
    data = data.dropna(subset=item_columns + [col for col in feature_columns if col != 'Hour_of_Day'])

# Update the format for the Time column to match the dataset's HH:MM:SS format
data['Hour_of_Day'] = pd.to_datetime(data['Time'], format='%H:%M:%S').dt.hour

# Reset the index to ensure contiguous indices starting from 0
data = data.reset_index(drop=True)

# Define items
items = []
item_to_id = {}
item_id = 0
for col in item_columns:
    unique_values = data[col].unique()
    for val in unique_values:
        item_name = f"{col}={val}"
        items.append(item_name)
        item_to_id[item_name] = item_id
        item_id += 1

M = len(items)
print(f"Number of items: {M}")
print(f"Items: {items}")

# Construct feature matrix X (M x d) for GPAR
d = len(feature_columns)
X = np.zeros((M, d))
for i, item in enumerate(items):
    col, val = item.split('=')
    val = float(val)
    mask = data[col] == val
    if mask.sum() > 0:
        X[i] = data.loc[mask, feature_columns].mean().values
    else:
        X[i] = 0

# Construct transaction matrix T (N x M)
N = len(data)
T = np.zeros((N, M), dtype=int)
for idx, row in data.iterrows():
    for col in item_columns:
        val = row[col]
        item_name = f"{col}={val}"
        item_idx = item_to_id[item_name]
        T[idx, item_idx] = 1

# Convert T to a DataFrame for Apriori, FP-Growth, and Eclat
T_df = pd.DataFrame(T, columns=items)

# GPAR Implementation
def rbf_kernel(X, length_scale):
    sq_dists = squareform(pdist(X, 'sqeuclidean'))
    return np.exp(-sq_dists / (2 * length_scale ** 2))

def neg_log_likelihood(length_scale, X, T):
    K = rbf_kernel(X, length_scale)
    K += np.eye(M) * 1e-6
    L = np.linalg.cholesky(K)
    log_det = 2 * np.sum(np.log(np.diag(L)))
    alpha = np.linalg.solve(K, T.T @ T)
    quad_term = np.trace(T.T @ T @ alpha)
    return 0.5 * (log_det + quad_term)

def monte_carlo_prob(T, itemset, S=100):
    samples = np.random.normal(0, 1, (S, len(itemset)))
    item_indices = list(itemset)
    K_subset = K[np.ix_(item_indices, item_indices)]
    L = np.linalg.cholesky(K_subset + np.eye(len(itemset)) * 1e-6)
    z = np.dot(L, samples.T).T
    z_binary = (z > 0).astype(int)
    prob = np.mean(np.all(z_binary == 1, axis=1))
    return prob

def compute_lift(T, antecedent, consequent, p_I):
    p_A = monte_carlo_prob(T, antecedent)
    p_B = monte_carlo_prob(T, consequent)
    if p_A * p_B == 0:
        return 0
    return p_I / (p_A * p_B)

# Eclat Implementation
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

            # Correct support calculation for new_prefix
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

# Run all algorithms with varying min_support thresholds
min_support_values = [0.1, 0.2, 0.3, 0.4, 0.5]
min_conf = 0.5

# Initialize lists to store metrics
gpar_runtimes, gpar_memory, gpar_num_rules, gpar_num_itemsets = [], [], [], []
apriori_runtimes, apriori_memory, apriori_num_rules, apriori_num_itemsets = [], [], [], []
fpgrowth_runtimes, fpgrowth_memory, fpgrowth_num_rules, fpgrowth_num_itemsets = [], [], [], []
eclat_runtimes, eclat_memory, eclat_num_rules, eclat_num_itemsets = [], [], [], []

# Optimize length scale for GPAR once
initial_length_scale = 1.0
result = minimize(neg_log_likelihood, initial_length_scale, args=(X, T), bounds=[(0.1, 100.0)])
length_scale = result.x[0]
K = rbf_kernel(X, length_scale)

print(f"Optimized length scale: {length_scale}")
