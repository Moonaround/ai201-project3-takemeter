# TakeMeter Project Planning: r/nba Quality Metrics

## 1. Selected Online Space
* **Platform:** Reddit
* **Target Hub:** r/nba

## 2. Definitive Custom Taxonomy
* `tactical_analysis`: Deeply technical or objective assertions backed up by statistics, cap-space arithmetic, or structural play-style evidence.
* `narrative_take`: Pure opinion-based, speculative, or dramatic declarations concerning player status, legacy, or hypothetical future movements lacking data.
* `fan_reaction`: Immediate, highly emotional short-form outbursts, game-thread hype text, inside jokes, or pure casual celebration/frustration.

## 3. Strict Boundary Isolation Rules
* **The Single-Stat Limit:** If a comment is purely emotional or insulting but uses a single base game statistic for authority (e.g., *"He is awful, he shot 2 for 10"*), it remains classified as `narrative_take` or `fan_reaction`.
* **The Structural Split Rule:** If a user expresses deep emotional bias but backs it up with multi-sentence, comprehensive stylistic analysis, default to `tactical_analysis`.

## 4. Hard Edge-Case Analysis
* *Boundary Example:* "His perimeter defensive numbers dropped this season because his close-out speed looks visibly slower."
* *Taxonomy Dilemma:* Melds technical concepts ("defensive numbers") with unmeasured visual bias ("looks visibly slower").
* *Ultimate Decision Rule:* Since the base claim rests on an unmeasured physical description rather than real concrete metrics or clip breakdowns, classify as `narrative_take`.

## 5. Data Collection Plan

**Source:** Reddit r/nba subreddit via public JSON API (no authentication required).

**Endpoints:**
- `/r/nba/hot.json?limit=100` — recent, mixed-quality posts
- `/r/nba/top.json?limit=100&t=week` — high-engagement posts (longer timeframe allows more analytical content)
- `/r/nba/rising.json?limit=100` — emerging posts (more game-thread hype, `fan_reaction` density)
- `/r/nba/comments/{id}.json?limit=50` — top comments from top 60 posts by comment count

**Text sources per post:** post title, post body (selftext if present and not removed), top-level comments.

**Candidate filtering:**
- Minimum length: 20 characters
- Exclude: "[removed]", "[deleted]" posts
- Deduplication: by MD5 hash of text content

**Target:** 75 examples per class (225 total), covering a minimum of 200 after human review.

**Balance strategy:** Track counts for each label. Skip candidates if their label bucket has reached 75 examples. Continue until all three buckets hit 75 or the candidate pool is exhausted. If one class remains underrepresented after the initial run, manually fetch game-thread posts (search `r/nba game thread` for recent post IDs) to boost `fan_reaction` density.

**Labeling method:** 
- Automated pre-labeling via Groq `llama-3.3-70b-versatile` zero-shot classification (labels embedded in system prompt)
- Mandatory human review pass for all rows labeled "REVIEW" or "FAILED"
- Spot-check of ~20 random rows to assess pre-labeling accuracy and consistency
- Correction of any misclassified examples before upload to Colab

**Tool:** Python script `collect_and_label.py` handles API fetching, deduplication, Groq pre-labeling with retry/rate-limit logic, and CSV output.

---

## 6. AI Tool Plan

**Label stress-testing:** 
- Provided the three label definitions to Groq/Claude and requested 10 posts that sit at the boundary between `narrative_take` and `tactical_analysis` (the single-stat edge case).
- If generated posts proved unclassifiable with the definitions, the boundary rule was tightened before annotating 200 examples.
- Status: boundary rule finalized in Section 4.

**Annotation assistance:** 
- Groq `llama-3.3-70b-versatile` is used for pre-labeling all candidates via `collect_and_label.py`.
- Every pre-labeled row is reviewed by human (mandatory for "REVIEW" and "FAILED" rows, spot-check for valid labels).
- All AI-assisted labeling disclosed in the project's AI usage section of the README.

**Failure analysis:** 
- After fine-tuning on the test set, all misclassified examples (at least 3 per the project spec) are extracted and pasted into Claude.
- Claude is asked to identify common patterns: post length, use of sarcasm, label confusion pairs, specific linguistic markers, etc.
- Patterns are then verified manually by re-reading the examples and the confusion matrix.
- Findings reported in the README's evaluation section.

---

## 7. Evaluation Metrics Plan

**Primary metric:** **Macro F1-score** (unweighted average of per-class F1 scores).
- Justification: Macro F1 treats all three classes equally, preventing the model from gaming accuracy by always predicting the majority class. Given the risk of label imbalance in real Reddit data, macro F1 is the most honest metric.

**Secondary metrics:**
- **Per-class precision:** For each label, fraction of predicted instances that were correct.
- **Per-class recall:** For each label, fraction of true instances the model correctly identified.
- **Per-class F1:** Harmonic mean of precision and recall.

**Baseline:** Groq `llama-3.3-70b-versatile` with zero-shot system-prompt classification (same prompt used for pre-labeling, but applied to the test set only after training to avoid data leakage).

**Comparison:** Side-by-side accuracy and macro F1 for both DistilBERT fine-tuned and Groq Llama baseline on the same held-out test set (15% of total data).

**Confusion matrix:** 3×3 matrix (rows = true labels, columns = predicted labels). Focus on off-diagonal cells to identify which label pairs the model confuses most.

**Expected confusion patterns:**
- `narrative_take` → `tactical_analysis`: The model may over-generalize single-stat posts as multi-point analysis.
- `narrative_take` ↔ `fan_reaction`: Both are non-analytical and emotional; the boundary is post length and structure.
- `tactical_analysis` → `narrative_take`: The model may under-learn the statistical/structural evidence markers.

**Success criteria:**
- DistilBERT fine-tuned macro F1 ≥ 0.70 on test set.
- DistilBERT outperforms Groq Llama on at least 2 of 3 per-class F1 scores (evidence that fine-tuning learned label-specific nuance).
- Confusion matrix shows no single off-diagonal cell exceeding 15% of test set size (no catastrophic confusion pair).

**Evaluation report structure (in README):** 
- Overall accuracy for both models
- Per-class metrics (precision, recall, F1) for both models
- Confusion matrix as a markdown table + image
- At least 3 specific misclassified examples with analysis of why the model failed
- Reflection on gap between intended and learned behavior
