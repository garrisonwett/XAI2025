import pandas as pd
import os

file_path = "research_data/phase2_ablation_results.csv"

if not os.path.exists(file_path):
    print("[ERROR] No data file found at:", file_path)
else:
    df = pd.read_csv(file_path)
    
    # 1. Count Unique Run IDs (How many trials actually finished and saved?)
    run_ids = df['run_id'].unique()
    print(f"\n[INFO] Total Completed Trials Found: {len(run_ids)}")
    print("-" * 40)
    
    # 2. Group by Experiment to see breakdown
    # We parse the string to ignore the trial number for grouping
    # Assumes format "ExperimentName_trial_X"
    if not df.empty:
        df['exp_group'] = df['run_id'].apply(lambda x: x.split('_trial_')[0])
        
        counts = df.groupby('exp_group')['run_id'].nunique()
        print("Trial Counts per Experiment:")
        print(counts)
        
        print("-" * 40)
        # 3. Check for Identical Scores (The Variance=0 Bug)
        # Get the final fitness for each run to see if they are different
        final_scores = df.groupby('run_id')['best_fitness'].last()
        print("\n[DATA] Final Scores per Trial:")
        print(final_scores)
        
        # Check variance automatically
        print("\n[ANALYSIS] Variance Check:")
        for group in counts.index:
            # Filter scores belonging to this group
            group_scores = final_scores[final_scores.index.str.contains(group)]
            
            if len(group_scores) < 2:
                print(f" -> {group}: Not enough data to calculate variance (N={len(group_scores)})")
            elif group_scores.std() == 0:
                print(f" -> {group}: ZERO VARIANCE DETECTED. All trials have identical scores.")
                print("    (This causes NaN in T-Tests. Check random seeding.)")
            else:
                print(f" -> {group}: Variance OK (Std Dev: {group_scores.std():.4f})")
    else:
        print("[WARNING] CSV file is empty.")