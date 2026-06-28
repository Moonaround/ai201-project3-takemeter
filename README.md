# TakeMeter: Basketball Quality Discourse Classifier

This system evaluates and maps text metrics from community discourse using fine-tuned semantic transformers compared to large language model baselines.

## Custom Taxonomy Layout
* **`tactical_analysis`**: Strategic, deep, data-backed breakdowns.
* **`narrative_take`**: Subjective, non-empirical narrative claims.
* **`fan_reaction`**: Short, high-sentiment community expressions.

## Community Context
- **Platform:** Reddit
- **Target community:** r/nba
- **Why this community:** NBA subreddit features high-volume discourse with clear quality distinctions. Posts range from analytical breakdowns (cap-space math, playoff mechanics) to hot takes (player comparisons, legacy debates) to raw game-thread reactions (hype, frustration). These distinctions are meaningful to community members and reflect broader discourse norms.

## Quantitative Architecture
* **Fine-Tuned Architecture:** `distilbert-base-uncased`
* **Base Model:** Hugging Face pre-trained DistilBERT (uncased, 6 layers, 768 hidden dims)
* **Baseline Engine:** `llama-3.3-70b-versatile` via Groq Cloud API
* **Hyperparameters Chosen:** Learning Rate = `2e-5`, Epochs = `4`, Batch Size = `16`.
* **Training approach:** Supervised fine-tuning on labeled training set (70% of ~225 examples) with validation on 15% held-out validation set. Test set (15%) reserved for final evaluation and baseline comparison.

## Dataset
- **Total examples:** ~225 labeled posts and comments (target: 75 per label class)
- **Data split:** 70% training / 15% validation / 15% test (handled by Colab notebook)
- **Source:** Reddit r/nba subreddit, collected June 2026 via public JSON API
- **Text sources:** Post titles, post bodies (selftext), and top-level comments from high-engagement posts
- **Labeling method:** Groq Llama 3.3 70B pre-labeling (zero-shot with label definitions) followed by mandatory human review and correction
- **Label distribution:** [To be filled after data collection completes — run `collect_and_label.py` script]
- **Difficult annotation cases:** [To be documented during manual review pass]

## Setup and Reproduction

### Install dependencies
```bash
pip install groq requests
```

### Data collection
```bash
export GROQ_API_KEY="gsk_your_api_key_here"
python collect_and_label.py
```

The script:
- Fetches posts from r/nba hot, top, and rising feeds via Reddit's public JSON API
- Pre-labels candidates using Groq Llama 3.3
- Outputs `dataset.csv` with ~225 examples balanced across three labels
- Expects manual review of pre-labeled examples before training (see next step)

### Manual review before training
1. Open `dataset.csv` in Google Sheets
2. Review all rows labeled "REVIEW" or "FAILED" — assign correct labels manually
3. Spot-check ~20 random rows to assess pre-labeling quality
4. Correct any misclassified examples
5. Delete `source` and `notes` columns
6. Save as CSV (not xlsx)

### Fine-tuning and evaluation (in Google Colab)
1. Open the [TakeMeter Colab notebook](https://colab.research.google.com/drive/1ilOny04QwR6CRUYLKvFycwzDsQLdPypI?usp=sharing)
2. Set runtime to T4 GPU: Runtime → Change runtime type → T4 GPU
3. Add Groq API key: Click 🔑 icon (Secrets) → add `GROQ_API_KEY` → enable notebook access
4. Upload `dataset.csv` via Files panel
5. Run sections in order: 1 (load data) → 2 (split and tokenize) → 5 (Groq baseline) → 3 (fine-tune) → 4 (evaluate) → 6 (comparison)
6. Download `evaluation_results.json` and `confusion_matrix.png`

---

## Performance Evaluation Report

### Model Accuracies
* **Fine-Tuned DistilBERT Accuracy:** [To be filled after training]
* **Groq Llama 3.3 70B Accuracy:** [To be filled after training]
* **Macro F1 (DistilBERT):** [To be filled after training]
* **Macro F1 (Groq Llama):** [To be filled after training]

### Per-Class Metrics

| Class | DistilBERT Precision | DistilBERT Recall | DistilBERT F1 | Groq Llama Precision | Groq Llama Recall | Groq Llama F1 |
|-------|----------------------|-------------------|---------------|----------------------|-------------------|---------------|
| tactical_analysis | [To fill] | [To fill] | [To fill] | [To fill] | [To fill] | [To fill] |
| narrative_take | [To fill] | [To fill] | [To fill] | [To fill] | [To fill] | [To fill] |
| fan_reaction | [To fill] | [To fill] | [To fill] | [To fill] | [To fill] | [To fill] |

### Confusion Matrix (DistilBERT)

| Predicted → | tactical_analysis | narrative_take | fan_reaction |
|Actual ↓|---|---|---|
| **tactical_analysis** | [To fill] | [To fill] | [To fill] |
| **narrative_take** | [To fill] | [To fill] | [To fill] |
| **fan_reaction** | [To fill] | [To fill] | [To fill] |

### Misclassified Examples and Analysis

**Example 1:** [To be filled with actual misclassified text from test set]
- Ground truth: [label]
- Model prediction: [label]
- Analysis: [Why the model failed — focus on which labels were confused and what linguistic or contextual factors made the distinction hard]

**Example 2:** [To be filled]
- Ground truth: [label]
- Model prediction: [label]
- Analysis: [Explanation of failure mode]

**Example 3:** [To be filled]
- Ground truth: [label]
- Model prediction: [label]
- Analysis: [Explanation of failure mode]

### Sample Classifications (DistilBERT)

| Text | Predicted Label | Confidence | Notes |
|------|---|---|---|
| [Example 1] | [label] | [conf]% | [Explanation for correct predictions] |
| [Example 2] | [label] | [conf]% | |
| [Example 3] | [label] | [conf]% | |
| [Example 4] | [label] | [conf]% | |
| [Example 5] | [label] | [conf]% | |

### Reflection: Intended vs. Learned Behavior

[To be filled after evaluation: Describe one way the model's learned decision boundary diverged from the intended label definitions. For example, did it overfit to post length? Did it struggle with sarcasm? Did the `narrative_take` / `tactical_analysis` boundary prove harder than expected?]

### Spec Reflection

**How the spec helped:** [One specific way the spec structure guided implementation decisions]

**How implementation diverged from spec:** [One reason the approach changed from the spec, and why]

---

## AI Usage

### AI Tool Application 1: Label Stress-Testing
- **Action:** Provided label definitions to Claude and requested boundary-case posts between `narrative_take` and `tactical_analysis`.
- **Output:** Claude generated posts using the single-stat edge case.
- **What I changed/verified:** Reviewed output against definitions. Single-stat posts with emotional framing are `narrative_take`, not `tactical_analysis`. Boundary rule refined and finalized before annotation.

### AI Tool Application 2: Data Collection Pre-Labeling
- **Action:** Groq Llama 3.3 70B pre-labeled all 225 candidate posts and comments using `collect_and_label.py` with zero-shot classification prompt.
- **Output:** Pre-labeled dataset with labels: "tactical_analysis", "narrative_take", "fan_reaction", or "REVIEW"/"FAILED" for unparseable responses.
- **What I changed/verified:** Mandatory human review pass for all "REVIEW" and "FAILED" rows. Spot-checked ~20 random rows. Corrected misclassifications before uploading to Colab. All AI-assisted rows disclosed.

### AI Tool Application 3: Failure Pattern Analysis (Post-Training)
- **Action:** [To be filled after fine-tuning: pasted list of misclassified test examples into Claude, asked for pattern identification]
- **Output:** [Claude's identified patterns: e.g., "The model confuses narrative_take and tactical_analysis when posts cite a single stat in an accusatory tone"]
- **What I changed/verified:** [Verified patterns manually by re-reading examples and consulting confusion matrix]
