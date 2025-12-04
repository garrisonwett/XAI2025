from algorithms import get_ga_config, run_ga, save_chromosome
from redone_controller import FuzzyController

if __name__ == "__main__":

    cfg = get_ga_config()

    cfg["controller_callback"] = lambda chrom: FuzzyController(chrom)

    best, history = run_ga(cfg)

    save_chromosome(best, "best_kessler_fuzzy.pkl")

    print("Training complete.")
