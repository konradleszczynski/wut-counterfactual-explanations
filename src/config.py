"""Project-wide constants, paths, and configuration."""

from pathlib import Path

# ==============================================================================
# Reproducibility
# ==============================================================================
RANDOM_SEED: int = 42

# ==============================================================================
# Paths (relative to repository root)
# ==============================================================================
ROOT_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = ROOT_DIR / "data"
MODELS_DIR: Path = ROOT_DIR / "models"
NOTEBOOKS_DIR: Path = ROOT_DIR / "notebooks"

# ==============================================================================
# Dataset files (expected after Kaggle download)
# ==============================================================================
APPLICATION_TRAIN_FILE: Path = DATA_DIR / "application_train.csv"
APPLICATION_TEST_FILE: Path = DATA_DIR / "application_test.csv"
BUREAU_FILE: Path = DATA_DIR / "bureau.csv"
BUREAU_BALANCE_FILE: Path = DATA_DIR / "bureau_balance.csv"
PREVIOUS_APPLICATION_FILE: Path = DATA_DIR / "previous_application.csv"
POS_CASH_BALANCE_FILE: Path = DATA_DIR / "POS_CASH_balance.csv"
INSTALLMENTS_PAYMENTS_FILE: Path = DATA_DIR / "installments_payments.csv"
CREDIT_CARD_BALANCE_FILE: Path = DATA_DIR / "credit_card_balance.csv"

# ==============================================================================
# Modeling
# ==============================================================================
TARGET_COLUMN: str = "TARGET"
TEST_SIZE: float = 0.2
N_FEATURES_TO_SELECT: int = 20  # Select 15-20 features (adjust as needed)

# ==============================================================================
# Counterfactual Analysis
# ==============================================================================
N_COUNTERFACTUAL_EXAMPLES: int = 10  # Select 10-15 interesting test examples
