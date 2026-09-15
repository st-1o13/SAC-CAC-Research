# Auto-extracted from Proposal_Ver1.ipynb.ipynb.
# Do not hand-edit the extracted logic; this file is a verbatim relocation,
# not a rewrite. Only import statements were added/adjusted for standalone execution.

# NUM_RUNS = 100 for the Proposal Ver.1 production run. Results are saved to
# runs_proposal_ver1/ (via save_dir) so they never mix with or overwrite the
# existing runs/, runs_cac/, or runs_verify_*/ directories.

from train import run_experiment

NUM_RUNS = 100
SAVE_DIR = "runs_proposal_ver1"
for run_id in range(1, NUM_RUNS + 1):
    run_experiment(run_id, save_dir=SAVE_DIR)
