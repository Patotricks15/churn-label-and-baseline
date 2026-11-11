MODEL_VERSION = "CH01 v0.1"

# Churn window in days: a customer churns if no purchase occurs in [snapshot, snapshot + H_DAYS).
H_DAYS = 90

# Fixed snapshot dates. Train snapshots precede test snapshots in time.
TRAIN_SNAPSHOTS = ["2010-03-01", "2010-06-01", "2010-09-01", "2010-12-01"]
TEST_SNAPSHOTS = ["2011-03-01", "2011-06-01", "2011-09-01"]

# Sensitivity check on the churn window (see README).
H_SENSITIVITY_DAYS = [30, 60, 90]

RANDOM_STATE = 42
