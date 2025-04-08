import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Compute EER for each user from a CSV of genuine/impostor scores.")
parser.add_argument("input_csv", help="Path to input CSV file containing columns: gen_user, test_user, window_idx, energy_sum")
parser.add_argument("output_csv", help="Path to output CSV file to store per-user EER values")
parser.add_argument("--plot_dir", help="Directory to save FAR/FRR and ROC plot images", default=".")
args = parser.parse_args()

input_path = args.input_csv
output_path = args.output_csv
plot_dir = args.plot_dir

# Create the output directory for plots if it does not exist
os.makedirs(plot_dir, exist_ok=True)

# Read the CSV data into a pandas DataFrame
print(f"Reading input data from {input_path}...")
try:
    df = pd.read_csv(input_path)
except Exception as e:
    print(f"Error reading CSV file: {e}")
    exit(1)

# Verify that required columns are present
required_cols = {'gen_user', 'test_user', 'window_idx', 'energy_sum'}
if not required_cols.issubset(df.columns):
    print("Input CSV is missing one of the required columns: gen_user, test_user, window_idx, energy_sum")
    exit(1)

# Prepare a list to collect EER results
eer_results = []

# Group data by each genuine user
for user, group in df.groupby('gen_user'):
    print(f"\nProcessing user '{user}'...")

    # Separate genuine and impostor attempts for this user
    genuine_scores = group[group['test_user'] == user]['energy_sum'].values
    impostor_scores = group[group['test_user'] != user]['energy_sum'].values

    # If no impostor or no genuine samples for this user, skip or assign EER appropriately
    if len(genuine_scores) == 0 or len(impostor_scores) == 0:
        print(f"  Skipping user {user}: insufficient data for EER (genuine={len(genuine_scores)}, impostor={len(impostor_scores)})")
        continue

    # Sort the scores for threshold sweep
    genuine_scores.sort()
    impostor_scores.sort()

    # Initialize lists to store FAR and FRR values for various thresholds
    far_list = []
    frr_list = []
    thr_list = []

    # We will iterate through candidate threshold values.
    # A good set of thresholds is all unique score values from both genuine and impostor sets.
    all_scores = np.concatenate([genuine_scores, impostor_scores])
    unique_scores = np.unique(all_scores)

    # To ensure we consider thresholds that yield 0% and 100% error rates,
    # we also include values just below the minimum score and just above the maximum score.
    # This accounts for the extremes (all rejected vs. all accepted).
    min_score = unique_scores[0]
    max_score = unique_scores[-1]
    # Use a tiny epsilon to go just outside the range
    epsilon = 1e-6
    thresholds = np.concatenate(([min_score - epsilon], unique_scores, [max_score + epsilon]))

    # Compute FAR and FRR for each threshold
    # Note: We consider "accept" if energy_sum <= threshold (assuming lower scores indicate more likely genuine).
    total_genuine = len(genuine_scores)
    total_impostor = len(impostor_scores)
    for thr in thresholds:
        # Number of genuine scores <= thr (accepted genuine)
        num_gen_accept = np.searchsorted(genuine_scores, thr, side='right')
        # Number of impostor scores <= thr (accepted impostors)
        num_imp_accept = np.searchsorted(impostor_scores, thr, side='right')

        # Compute rates
        far = num_imp_accept / float(total_impostor)   # False Accept Rate
        frr = 1 - (num_gen_accept / float(total_genuine))  # False Reject Rate (1 - True Accept Rate)

        far_list.append(far)
        frr_list.append(frr)
        thr_list.append(thr)

    far_arr = np.array(far_list)
    frr_arr = np.array(frr_list)
    thr_arr = np.array(thr_list)

    # Calculate EER: find the point where |FAR - FRR| is minimal
    abs_diffs = np.abs(far_arr - frr_arr)
    eer_index = np.argmin(abs_diffs)
    eer_value = (far_arr[eer_index] + frr_arr[eer_index]) / 2.0  # average FAR and FRR at this point
    eer_threshold = thr_arr[eer_index]

    print(f"  - Found EER ≈ {eer_value:.4f} (approximately {eer_value*100:.2f}%) for user {user} at threshold {eer_threshold:.4f}")
    eer_results.append({'gen_user': user, 'EER': eer_value})

    # Plot FAR and FRR as functions of threshold for this user
    plt.figure(figsize=(6, 5))
    plt.plot(thr_arr, far_arr, label="FAR (False Accept Rate)", color='orange')
    plt.plot(thr_arr, frr_arr, label="FRR (False Reject Rate)", color='red')
    # Mark the EER point on the FAR/FRR plot
    plt.axvline(eer_threshold, color='gray', linestyle='--', label=f"EER Threshold = {eer_threshold:.3f}")
    plt.axhline(eer_value, color='gray', linestyle='--')
    plt.scatter([eer_threshold], [eer_value], color='black', zorder=5, label=f"EER = {eer_value*100:.2f}%")
    plt.title(f"User {user} - FAR and FRR vs Threshold")
    plt.xlabel("Decision Threshold")
    plt.ylabel("Error Rate")
    plt.ylim([0, 1])
    plt.legend(loc="best")
    # Save the FAR/FRR plot
    plot_path = os.path.join(plot_dir, f"user_{user}_FAR_FRR.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"  - Saved FAR/FRR curve plot to {plot_path}")

    # Plot ROC curve (TPR vs FPR) for this user
    # True Positive Rate (TPR) = 1 - FRR, False Positive Rate (FPR) = FAR
    tpr_arr = 1 - frr_arr
    fpr_arr = far_arr
    plt.figure(figsize=(6, 5))
    plt.plot(fpr_arr, tpr_arr, label="ROC Curve", color='orange')
    # Plot the diagonal line for reference (chance line)
    plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
    # Mark the EER point on the ROC curve
    eer_fpr = far_arr[eer_index]
    eer_tpr = tpr_arr[eer_index]
    plt.scatter([eer_fpr], [eer_tpr], color='black', zorder=5, label=f"EER Point = {eer_value*100:.2f}%")
    plt.title(f"User {user} - ROC Curve")
    plt.xlabel("False Positive Rate (FAR)")
    plt.ylabel("True Positive Rate (1 - FRR)")
    plt.xlim([0, 1]); plt.ylim([0, 1])
    plt.legend(loc="lower right")
    # Save the ROC plot
    roc_path = os.path.join(plot_dir, f"user_{user}_ROC.png")
    plt.savefig(roc_path)
    plt.close()
    print(f"  - Saved ROC curve plot to {roc_path}")

# Save all EER results to a CSV file
print(f"\nSaving per-user EER results to {output_path}...")
eer_df = pd.DataFrame(eer_results)
eer_df.to_csv(output_path, index=False)
print("Done. EER computation complete for all users.")
