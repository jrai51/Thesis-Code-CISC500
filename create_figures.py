import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("results/MEMTO_results.csv")
df['attempt_type'] = df.apply(lambda row: 'genuine' if row['gen_user']==row['test_user'] else 'impostor', axis=1)
df['attempt_type'] = pd.Categorical(df['attempt_type'], categories=['genuine','impostor'], ordered=True)

# Set plot style for consistency and clarity
sns.set_theme(style="whitegrid", palette="muted")

# 1. Scatter plot of energy_sum per user (colored by attempt type)
plt.figure(figsize=(8,6))
sns.stripplot(x="gen_user", y="anomaly_percentage", hue="attempt_type", data=df, dodge=False, jitter=True, hue_order=["genuine","impostor"])
plt.xlabel("User")
plt.ylabel("Anomaly Score (energy_sum)")
plt.legend(title="Attempt Type", loc="upper right")
plt.title("Energy Sum Anomaly Scores per User")
plt.tight_layout()
plt.savefig("energy_score_scatter.png", dpi=300)
plt.close()

# 2. Box-and-whisker plot comparing genuine vs impostor distributions per user
plt.figure(figsize=(10,6))
sns.boxplot(x="gen_user", y="anomaly_percentage", hue="attempt_type", data=df, hue_order=["genuine","impostor"])
plt.xlabel("User")
plt.ylabel("Anomaly Score (energy_sum)")
plt.legend(title="Attempt Type", loc="upper right")
plt.title("Distribution of Energy Sum Scores: Genuine vs Impostor")
plt.tight_layout()
plt.savefig("energy_score_boxplot.png", dpi=300)
plt.close()

# 3. Overlaid histogram of energy_sum for genuine vs impostor attempts
plt.figure(figsize=(8,6))
genuine_scores = df[df['attempt_type']=="genuine"]['anomaly_percentage']
impostor_scores = df[df['attempt_type']=="impostor"]['anomaly_percentage']
plt.hist(genuine_scores, bins=30, alpha=0.5, label='Genuine', color='C0')
plt.hist(impostor_scores, bins=30, alpha=0.5, label='Impostor', color='C1')
plt.xlabel("Anomaly Score (anomaly_percentage)")
plt.ylabel("Frequency")
plt.legend(title="Attempt Type")
plt.title("Histogram of Energy Sum Scores: Genuine vs Impostor")
plt.tight_layout()
plt.savefig("energy_score_histogram.png", dpi=300)
plt.close()

# 4. Summary table of descriptive stats per user (genuine vs impostor)
stats = df.groupby(['gen_user','attempt_type'])['anomaly_percentage'].agg(['mean', 'median', 'std', 'min', 'max'])
stats.reset_index().to_csv("energy_score_summary_stats.csv", index=False)
