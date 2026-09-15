# Auto-extracted from Proposal_Ver1.ipynb.ipynb.
# Do not hand-edit the extracted logic; this file is a verbatim relocation,
# not a rewrite. Only import statements were added/adjusted for standalone execution.

# NOTE: NUM_RUNS is set to 1 here (not 10, unlike the notebook's current
# cell) deliberately, for the run_id=1 import/dry-run verification step
# requested for this migration. This is NOT a change to the experimental
# condition -- it only controls how many times this driver script calls
# run_experiment() in this verification pass. See the migration report.

from train import run_experiment

NUM_RUNS = 1
for run_id in range(1, NUM_RUNS + 1):
    run_experiment(run_id)
