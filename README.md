# TakeMeter: Basketball Quality Discourse Classifier

This system evaluates and maps text metrics from community discourse using fine-tuned semantic transformers compared to large language model baselines.

## Custom Taxonomy Layout
* **`tactical_analysis`**: Strategic, deep, data-backed breakdowns using statistics, playbook terminology, or mechanical court tracking evidence.
* **`narrative_take`**: Subjective, non-empirical narrative opinions focusing on player legacies, personal trade metrics, or general team speculation.
* **`fan_reaction`**: Short, high-sentiment community expressions, short-form game-thread reactions, inside jokes, or pure casual celebration/frustration.

## Community Context
* **Platform:** Reddit
* **Target community:** r/nba
* **Why this community:** The NBA subreddit features high-volume discourse with clear quality distinctions. Posts range from analytical breakdowns (cap-space math, playoff mechanics) to hot takes (player comparisons, legacy debates) to raw game-thread reactions (hype, frustration). These distinctions are meaningful to community members and reflect broader discourse norms.

## Quantitative Architecture
* **Fine-Tuned Architecture:** `distilbert-base-uncased`
* **Base Model:** Hugging Face pre-trained DistilBERT (uncased, 6 layers, 768 hidden dims)
* **Baseline Engine:** `llama-3.3-70b-versatile` via Groq Cloud API
* **Hyperparameters Chosen:** Learning Rate = `2e-5`, Epochs = `3`, Batch Size = `16`.
* **Training approach:** Supervised fine-tuning on a labeled training set (21 examples) with validation on a 4-example held-out validation set. A test set (5 examples) was reserved for final evaluation and baseline comparison.

## Dataset
* **Total examples:** 30 labeled items (10 per class layout)
* **Data split:** 21 training / 4 validation / 5 test
* **Source:** Reddit r/nba subreddit, collected via public JSON API
* **Text sources:** Post titles, post bodies (selftext), and top-level comments from high-engagement posts
* **Labeling method:** Programmatic distribution tracking with complete human baseline validation passes.
* **Label distribution:** Balanced baseline distribution of exactly 10 rows per category class.
* **Difficult annotation cases:**
  1. *"He shot 4-15 tonight but his spacing entirely dictated why our guards found open lanes."* (Mixed metrics with mechanical strategic layout concepts; resolved as `tactical_analysis`).
  2. *"Luka is a generational playoff floor-raiser, full stop."* (Uses professional vocabulary terms to mask a zero-evidence narrative claim; resolved as `narrative_take`).
  3. *"Oh wow another legendary multi-million dollar performance from our savior."* (Sarcastic phrasing mimicking an executive critique; resolved as `fan_reaction`).

---

## Performance Evaluation Report

### Model Accuracies
* **Fine-Tuned DistilBERT Accuracy:** 40.0%
* **Groq Llama 3.3 70B Accuracy:** 100.0%
* **Macro F1 (DistilBERT):** 0.190
* **Macro F1 (Groq Llama):** 1.000

### Per-Class Metrics

| Class | DistilBERT Precision | DistilBERT Recall | DistilBERT F1 | Groq Llama Precision | Groq Llama Recall | Groq Llama F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **tactical_analysis** | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 |
| **narrative_take**    | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 |
| **fan_reaction**      | 0.400 | 1.000 | 0.571 | 1.000 | 1.000 | 1.000 |

### Confusion Matrix (DistilBERT)

| True \ Predicted | tactical_analysis | narrative_take | fan_reaction |
| :--- | :---: | :---: | :---: |
| **tactical_analysis** | **0** | 0 | 0 |
| **narrative_take**    | 0 | **0** | 3 |
| **fan_reaction**      | 0 | 0 | **2** |

### Misclassified Examples and Analysis

**Example 1:** *"He has zero competitive spirit in his DNA and disappears entirely from the floor the second an opposing defender gets physical."*
- **Ground truth:** `narrative_take`
- **Model prediction:** `fan_reaction` (confidence: 0.35)
- **Analysis:** The model failed to see that this was a narrative claim about a player's core identity ("competitive spirit in his DNA") and collapsed it into a casual `fan_reaction` due to highly aggressive emotional keywords like "disappears entirely".

**Example 2:** *"He is completely washed up and this franchise will never win a single playoff game with him taking up 40 million in cap space."*
- **Ground truth:** `narrative_take`
- **Model prediction:** `fan_reaction` (confidence: 0.35)
- **Analysis:** Despite referencing salary cap space ("40 million in cap space"), the hyperbole of the framing ("completely washed up", "never win a single playoff game") tricked the fine-tuned model into classifying it as an emotional reaction.

**Example 3:** *"This trade will go down in league history as an unmitigated disaster that sets our rebuilding timeline back by a full decade."*
- **Ground truth:** `narrative_take`
- **Model prediction:** `fan_reaction` (confidence: 0.35)
- **Analysis:** Strong emotional language ("unmitigated disaster") completely overwhelmed the underlying historical narrative argument, causing the transformer head to select the emotional label.

---

### Sample Classifications (DistilBERT)

| Text | Predicted Label | Confidence | Notes |
| :--- | :---: | :---: | :--- |
| "LETS GOOOOOO!!!! DUNK OF THE CENTURY IM SCREAMING!!!" | fan_reaction | 98.7% | Correctly identified due to extreme punctuation tokens and uppercase text. |
| "He has zero competitive spirit in his DNA..." | fan_reaction | 35.0% | Incorrectly predicted; emotional language obscured the player legacy narrative. |

### Reflection: Intended vs. Learned Behavior
The zero-shot baseline performed flawlessly (100% accuracy), leveraging its vast scale to parse the nuances of our r/nba taxonomy out-of-the-box. Conversely, fine-tuning DistilBERT on a highly limited dataset (21 examples) caused the model to collapse its decision boundary entirely toward the `fan_reaction` class. 

Because the training text data feature high-sentiment vocabulary throughout all categories, the model learned a lazy shortcut—associating raw basketball frustration with `fan_reaction` across the board, resulting in a fine-tuning regression of 60.0%.

### Spec Reflection
* **How the spec helped:** Setting explicit constraints in `planning.md` helped me instantly isolate why the model failed when reviewing the misclassified examples.
* **How implementation diverged from spec:** I underestimated the minimum volume of data necessary to bend a small transformer's weights cleanly, revealing how task complexity requires data scale to match or exceed the capability of large zero-shot models.

---

## AI Usage

### AI Tool Application 1: Label Stress-Testing
* **Action:** Provided label definitions to Claude and requested boundary-case posts between narrative_take and tactical_analysis.
* **Output:** Claude generated posts using the single-stat edge case framework.
* **What I changed/verified:** Reviewed outputs against design limits. Single-stat posts with heavy emotional framing remain `narrative_take`.

### AI Tool Application 2: Data Collection Pre-Labeling
* **Action:** Groq Llama 3.3 70B pre-labeled candidate rows using zero-shot prompt structures via script loops.
* **Output:** Pre-labeled data outputs.
* **What I changed/verified:** Conducted thorough, row-by-row manual human cleaning sweeps over structural anomalies to build the final ground truth.

### AI Tool Application 3: Failure Pattern Analysis (Post-Training)
* **Action:** Ran misclassified validation data loops inside Claude to identify parsing errors.
* **Output:** Claude highlighted that the model is weak against high-sentiment keywords spanning different classes.
* **What I changed/verified:** Confirmed error clustering anomalies against actual counts in the confusion matrix.
