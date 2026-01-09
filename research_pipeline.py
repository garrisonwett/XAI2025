import os
import time
import pandas as pd
import numpy as np
import itertools
import pickle
import shutil
from datetime import datetime

# --- Import your existing modules ---
# We use 'algorithms' to access the GA logic and config
import algorithms
# We need to pass the controller class to the GA
from redone_controller import FuzzyController

# -----------------------------------------------------------------------------
# GLOBAL SETTINGS
# -----------------------------------------------------------------------------
DATA_DIR = "research_data"
CHECKPOINT_DIR = "checkpoints"

# specific random seeds for reproducibility across experiments
SEEDS = [42, 101, 999] 

def setup_directories():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)

def save_experiment_data(experiment_name, run_id, history, config, best_agent):
    """
    Saves the fitness history to CSV and the best agent to a pickle.
    """
    # 1. Save Fitness History
    df = pd.DataFrame({
        "generation": range(len(history)),
        "best_fitness": history,
        "run_id": run_id,
        "experiment": experiment_name
    })
    
    # Append to master CSV
    csv_path = os.path.join(DATA_DIR, f"{experiment_name}_results.csv")
    write_header = not os.path.exists(csv_path)
    df.to_csv(csv_path, mode='a', header=write_header, index=False)
    
    # 2. Save Best Agent
    agent_path = os.path.join(DATA_DIR, f"{experiment_name}_{run_id}_best.pkl")
    with open(agent_path, "wb") as f:
        pickle.dump(best_agent, f)

# -----------------------------------------------------------------------------
# PHASE 1: HYPERPARAMETER TUNING
# -----------------------------------------------------------------------------
def run_hyperparameter_tuning():
    print("\n" + "="*60)
    print("PHASE 1: HYPERPARAMETER TUNING")
    print("="*60)

    # Define a small grid of parameters to test
    # Kept small for demonstration. Expand lists for deeper research.
    tuning_grid = {
        "popsize": [20, 40],
        "struct_mut_start": [0.3, 0.7],
        "mf_mut_start": [0.3, 0.7]
    }
    
    # Generate all combinations
    keys, values = zip(*tuning_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    best_config_score = -float('inf')
    best_config = None

    for i, params in enumerate(combinations):
        run_id = f"tune_{i}"
        print(f"\n--- Tuning Run {i+1}/{len(combinations)}: {params} ---")
        
        # 1. Get Default Config & Update with Grid Params
        cfg = algorithms.get_ga_config()
        cfg["controller_callback"] = FuzzyController
        
        # Override settings for tuning
        cfg["popsize"] = params["popsize"]
        cfg["struct_mut_prob_start"] = params["struct_mut_start"]
        cfg["mf_mut_rate_start"] = params["mf_mut_start"]
        
        # Reduce generations/time for tuning phase to save time
        cfg["generations"] = 100  # Short runs to see convergence speed
        cfg["max_hours"] = 3   # 180 mins max per run
        
        # 2. Run GA
        # Note: algorithms.run_ga returns (best_ind, history)
        best_ind, history = algorithms.run_ga(cfg)
        
        # 3. Save Data
        save_experiment_data("phase1_tuning", run_id, history, cfg, best_ind)
        
        # 4. Track Best
        final_score = history[-1] if history else -999
        if final_score > best_config_score:
            best_config_score = final_score
            best_config = params

    print(f"\n>>> Tuning Complete. Best Score: {best_config_score}")
    print(f">>> Best Config: {best_config}")
    
    # Save best config to file for Phase 2 to use
    with open(os.path.join(DATA_DIR, "best_hyperparams.pkl"), "wb") as f:
        pickle.dump(best_config, f)
        
    return best_config

# -----------------------------------------------------------------------------
# PHASE 2: ABLATION STUDY (The Core Research)
# -----------------------------------------------------------------------------
def run_ablation_study(best_params=None):
    print("\n" + "="*60)
    print("PHASE 2: ABLATION / COMPARATIVE STUDY")
    print("="*60)

    # Load best params if not provided
    if best_params is None:
        try:
            with open(os.path.join(DATA_DIR, "best_hyperparams.pkl"), "rb") as f:
                best_params = pickle.load(f)
        except:
            print("No tuning data found, using defaults.")
            best_params = {"popsize": 40, "struct_mut_start": 0.6, "mf_mut_start": 0.5}

    # Define the Experiments
    # 1. Simultaneous (Proposed): Optimizes Structure AND Params
    # 2. Fixed Structure: Optimizes Params ONLY (Structure mutation = 0)
    # 3. Fixed Params: Optimizes Structure ONLY (Param mutation = 0) - Optional, often performs poorly
    
    experiments = [
        ("Simultaneous_Proposed",  True, True),
        ("Fixed_Structure_Baseline", False, True),
        # ("Fixed_Params_Baseline", True, False) # Uncomment if you want to test structure-only
    ]
    
    # Run multiple trials (seeds) for statistical significance
    # Ideally 5-10 runs, using 2 here for time
    NUM_TRIALS = 3 
    GENERATIONS = 100 # Longer runs for final results

    for exp_name, do_struct, do_params in experiments:
        for trial in range(NUM_TRIALS):
            run_id = f"{exp_name}_trial_{trial}"
            print(f"\n--- Running {exp_name} | Trial {trial+1}/{NUM_TRIALS} ---")
            
            # 1. Setup Config
            cfg = algorithms.get_ga_config()
            cfg["controller_callback"] = FuzzyController
            cfg["generations"] = GENERATIONS
            cfg["max_hours"] = 2.0 
            
            # Apply Best Params from Phase 1
            cfg["popsize"] = best_params.get("popsize", 40)
            
            # --- APPLY ABLATION SETTINGS ---
            
            if do_struct:
                # Use tuned values
                cfg["struct_mut_prob_start"] = best_params.get("struct_mut_start", 0.6)
                cfg["struct_mut_prob_end"] = 0.1
            else:
                # Disable structure mutation
                cfg["struct_mut_prob_start"] = 0.0
                cfg["struct_mut_prob_end"] = 0.0
                
            if do_params:
                # Use tuned values
                cfg["mf_mut_rate_start"] = best_params.get("mf_mut_start", 0.5)
                cfg["mf_mut_rate_end"] = 0.05
                cfg["rule_mut_rate_start"] = best_params.get("mf_mut_start", 0.5) # sync rule with mf
                cfg["rule_mut_rate_end"] = 0.05
            else:
                # Disable param mutation
                cfg["mf_mut_rate_start"] = 0.0
                cfg["mf_mut_rate_end"] = 0.0
                cfg["rule_mut_rate_start"] = 0.0
                cfg["rule_mut_rate_end"] = 0.0

            # 2. Run GA
            # (In a real scientific paper, you might set np.random.seed here using SEEDS[trial])
            best_ind, history = algorithms.run_ga(cfg)
            
            # 3. Save Data
            save_experiment_data("phase2_ablation", run_id, history, cfg, best_ind)

# -----------------------------------------------------------------------------
# MAIN ENTRY POINT
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    setup_directories()
    
    # Step 1: Tune Parameters
    # (Comment this out if you've already done it and just want to run Phase 2)
    best_config = run_hyperparameter_tuning()
    
    # Step 2: Compare Algorithms
    run_ablation_study(best_config)
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print(f"Data saved to: {os.path.abspath(DATA_DIR)}")
    print("="*60)