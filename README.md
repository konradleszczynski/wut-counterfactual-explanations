# Counterfactual Explanation in Allegro Pay

**Warsaw University of Technology — Final Project (Spring 2026)**

> Build a credit risk classifier on the Home Credit Default Risk dataset, then use counterfactual explanations to understand *why* the model decides and *what would need to change* for a different outcome.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Team Structure](#team-structure)
- [Dataset](#dataset)
- [Environment Setup](#environment-setup)
- [Repository Structure](#repository-structure)
- [Implementation Roadmap](#implementation-roadmap)
  - [Phase 1 — Data Preparation](#phase-1--data-preparation)
  - [Phase 2 — Feature Engineering](#phase-2--feature-engineering)
  - [Phase 3 — Model Training & Validation](#phase-3--model-training--validation)
  - [Phase 4 — Counterfactual Analysis](#phase-4--counterfactual-analysis)
  - [Phase 5 — Evaluation & Benchmarking](#phase-5--evaluation--benchmarking)

- [Timeline](#timeline)
- [Deliverables](#deliverables)
- [AI Usage Policy](#ai-usage-policy)
- [Communication](#communication)

---

## Project Overview

This project has two core objectives:

1. **Predictive Modeling** — Develop a robust binary classification model that predicts whether a loan applicant will default on their credit (`TARGET = 1`) or not (`TARGET = 0`).

2. **Counterfactual Explanations (CE)** — For selected test instances, generate *counterfactual explanations* that answer: *"What minimal changes to this applicant's profile would flip the model's decision?"* This makes black-box predictions interpretable and actionable.

The final deliverable is a reproducible codebase and a presentation synthesizing modeling results and counterfactual insights for stakeholders.

---

## Team Structure

- Teams of **2 students**.
- Both team members must understand and be able to explain all code (see [AI Usage Policy](#ai-usage-policy)).

---

## Dataset

**Source:** [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/data) (Kaggle)

The dataset contains:

| File | Description |
|------|-------------|
| `application_train.csv` | Static application data with the binary `TARGET` |
| `application_test.csv` | Application data without `TARGET` (for Kaggle submission format) |
| `bureau.csv` | Client's previous credits from other financial institutions |
| `bureau_balance.csv` | Monthly balance snapshots of bureau credits |
| `previous_application.csv` | Previous Home Credit loan applications |
| `POS_CASH_balance.csv` | Monthly POS/cash loan balance snapshots |
| `installments_payments.csv` | Payment history for previous loans |
| `credit_card_balance.csv` | Monthly credit card balance snapshots |

### Download Instructions

The starter notebook (`notebooks/00_starter_notebook.ipynb`) includes a cell that **automatically downloads and extracts** the dataset using the Kaggle API. Just make sure you have the API token set up first:

1. Place your Kaggle API token at `~/.kaggle/kaggle.json` (download from [Kaggle Account Settings](https://www.kaggle.com/settings)).

2. Run the download cell in the notebook, or manually:
   ```bash
   kaggle competitions download -c home-credit-default-risk -p data/
   unzip data/home-credit-default-risk.zip -d data/
   ```

3. Verify the files exist in `data/`. These files are **gitignored** and must not be committed.

---

## Environment Setup

This project uses **[uv](https://docs.astral.sh/uv/)** for Python dependency management and virtual environments.

### Quick Start

```bash
# 1. Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone your fork
git clone git@github.com:<your-username>/wut-counterfactual-explanations.git
cd wut-counterfactual-explanations

# 3. Create the virtual environment and install all dependencies
uv sync

# 4. Activate the environment
source .venv/bin/activate

# 5. Register the Jupyter kernel
python -m ipykernel install --user --name wut-ce --display-name "WUT-CE"

# 6. Launch Jupyter
jupyter notebook
```

### Adding New Dependencies

```bash
uv add <package-name>        # Add to [project.dependencies]
uv add --group dev <package>  # Add to [dependency-groups.dev]
```

> **Do not** use `pip install` directly — always go through `uv` so that `pyproject.toml` and `uv.lock` stay in sync.

---

## Repository Structure

```
wut-counterfactual-explanations/
├── .gitignore                          # Python + project-specific ignores
├── .python-version                     # Python 3.11 (used by uv)
├── pyproject.toml                      # Dependencies & project metadata
├── README.md                           # This file
│
├── data/                               # Raw data (gitignored — download from Kaggle)
│   └── .gitkeep
│
├── models/                             # Serialized model artifacts (.pkl)
│   └── .gitkeep                        #   -> These ARE committed to the repo
│
├── notebooks/                          # Jupyter notebooks
│   └── 00_starter_notebook.ipynb       # Starter notebook with guided TODOs
│
└── src/                                # Reusable Python modules (extend as needed)
    ├── __init__.py
    └── config.py                       # Constants, paths, random seed
```

> **Note:** You are expected to create your own modules in `src/` (e.g., feature engineering, model training, counterfactual analysis) as your project evolves.

---

## Implementation Roadmap

### Phase 1 — Data Preparation

- Download and explore the Home Credit dataset.
- Understand the table relationships (join keys: `SK_ID_CURR`, `SK_ID_BUREAU`).
- Handle missing values, encode categoricals, and perform initial EDA.
- **Checkpoint deliverable** (14.04): Data downloaded and accessible.

### Phase 2 — Feature Engineering

- Aggregate supplementary tables (bureau, previous applications, balances) at the `SK_ID_CURR` level.
- Engineer new features with **semantic, human-readable names** (e.g., `income_to_credit_ratio`, not `feat_42`).
- Document the business logic behind each feature in code comments.

### Phase 3 — Model Training & Validation

- **Dataset partitioning:** Use `RANDOM_SEED = 42` for all train/test splits (defined in `src/config.py`).
- **Feature selection:** Select **15–20 features** using one of:
  - [Boruta](https://github.com/scikit-learn-contrib/boruta_py) (recommended)
  - Best Subset Selection (BSS) via `sklearn.feature_selection`
- **Model training:** Use tree-based classifiers:
  - **LightGBM** (recommended primary)
  - XGBoost or CatBoost (alternative/comparison)
  - You have flexibility to try other approaches, but at least one tree-based model is required.
- **Decision threshold:** Map predicted probabilities (PD) to a **credit rating masterscale** (AAA → D). Choose a rating grade as your decision boundary such that at least ~10% of applicants are classified as defaults. Visualize the PD distribution overlaid with rating thresholds.
- **Validation:** Report ROC-AUC, precision, recall, F1, and confusion matrix — computed at your chosen threshold.
- **SHAP analysis:** Produce a SHAP summary plot and provide a brief discussion of how the model behaves globally (which features matter most and in which direction).
- **Serialization:** Save the final model as a `.pkl` file in `models/`.

### Phase 4 — Counterfactual Analysis

1. **Select 10–15 interesting test examples** — Choose instances near the decision boundary, true positives, false positives/negatives, or high-confidence edge cases.

2. **Generate counterfactual explanations using two packages:**

   | Package | Purpose | Documentation |
   |---------|---------|---------------|
   | **DiCE** | Diverse Counterfactual Explanations | [github.com/interpretml/DiCE](https://github.com/interpretml/DiCE) |
   | **Alibi Explain** | Counterfactual with prototypes | [docs.seldon.io/projects/alibi](https://docs.seldon.io/projects/alibi) |

3. **Compare and contrast** the explanations from both methods.

### Phase 5 — Evaluation & Benchmarking

| Tool | Purpose | Documentation |
|------|---------|---------------|
| **DALEX** | Ceteris Paribus / What-If analysis profiles, feature importance | [dalex.drwhy.ai](https://dalex.drwhy.ai/) |

- **Compare DiCE vs Alibi:** Analyze differences in the counterfactuals produced by each method — sparsity, plausibility, diversity, and feature overlap.
- Use DALEX to create Ceteris Paribus (What-If) profiles for the selected examples, showing how single-feature changes affect predictions.
- Compute permutation-based feature importance to understand global model behavior.
- Synthesize insights: Which features are most influential? Are the counterfactuals realistic and actionable for a credit applicant?

---

## Timeline

| Date | Milestone | Description |
|------|-----------|-------------|
| **30.03** | Project Launch | Repository available, teams formed |
| **14.04** | Checkpoint | Data downloaded, initial model prepared |
| **27.04** | Code Submission | Final code committed to the repository |
| **30.04** | Presentation | Results presented to stakeholders |

> The final presentation will **not** be stored in this repo — upload it to the dedicated Google Drive space.

---

## Deliverables

- [ ] **Reproducible code** — Notebooks and/or scripts that can be re-run to reproduce all results.
- [ ] **In-code documentation** — Detailed comments explaining logic, assumptions, and design choices.
- [ ] **Environment specification** — `pyproject.toml` with pinned dependencies (managed via `uv`).
- [ ] **Trained model** — Serialized `.pkl` file in `models/` (committed to the repository).
- [ ] **Counterfactual analysis** — Explanations from DiCE + Alibi, evaluated with DALEX.
- [ ] **Final presentation** — Uploaded to Google Drive (not in this repo).

---

## AI Usage Policy

AI tools (e.g., ChatGPT, GitHub Copilot, Claude) **are permitted** as project aids, subject to the following rules:

1. **Code accountability** — Both team members must fully understand every line of code. You will be **interviewed** to verify comprehension.
2. **Documentation** — Explicitly document which AI tools were used and how, either in your notebooks or in a dedicated section of your code.
3. **No shortcuts** — AI can help you write code faster, but it cannot replace understanding. Use it to learn, not to avoid learning.

---

## Communication

All project communication happens on the **Slack workspace: `WUT-AllegroPay-CounterFactual`**.

| Channel | Purpose |
|---------|---------|
| `#general` | Announcements, timeline updates, general questions |
| `#data` | Technical questions about the dataset, modeling, or CE packages |

**Rules:**
- Ask questions early — do not wait until the deadline.
- **Do not share solutions** in open channels. Use DMs for team-specific discussions.
- Tag the instructor for urgent questions.

---

*Good luck! Focus on understanding, not just results.*
