#!/usr/bin/env python3
"""
TakeMeter: Collect and pre-label r/nba posts for discourse quality classification.

Fetches posts from r/nba via Reddit's public JSON API, pre-labels with Groq,
and outputs a balanced dataset.csv with 75 examples per label class.
"""

import os
import csv
import json
import time
import random
import hashlib
import requests
from groq import Groq

# Constants
GROQ_MODEL = "llama-3.3-70b-versatile"
TARGET_PER_CLASS = 75
MIN_TEXT_LENGTH = 20
REDDIT_HEADERS = {"User-Agent": "TakeMeter/1.0 (academic project; japjamunpan@gmail.com)"}
VALID_LABELS = {"tactical_analysis", "narrative_take", "fan_reaction"}

GROQ_SYSTEM_PROMPT = """You are a text classifier for basketball discourse on Reddit's r/nba community.
Classify text into exactly one of three labels:

  tactical_analysis — deeply technical assertions backed by stats, cap-space
    arithmetic, or structural play-style evidence (multi-sentence reasoning).
  narrative_take — pure opinion, speculation, or dramatic declarations about
    player status, legacy, or hypotheticals. May cite a single stat for
    emotional emphasis but lacks multi-point analytical structure.
  fan_reaction — immediate emotional outbursts, game-thread hype, inside jokes,
    short celebration/frustration. Usually under 2 sentences and data-free.

Rules:
- A single stat used for emotional emphasis → narrative_take or fan_reaction (not tactical_analysis)
- Multi-sentence structural breakdown even with emotional framing → tactical_analysis
- Reply with ONLY the label name. No punctuation, no explanation."""


def fetch_posts(feed: str, time_filter: str = None) -> list:
    """Fetch posts from a given r/nba feed endpoint."""
    url = f"https://www.reddit.com/r/nba/{feed}.json"
    params = {"limit": 100}
    if time_filter:
        params["t"] = time_filter

    try:
        response = requests.get(url, headers=REDDIT_HEADERS, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        posts = []
        for item in data.get("data", {}).get("children", []):
            post = item.get("data", {})
            if post:
                posts.append({
                    "id": post.get("id"),
                    "title": post.get("title", ""),
                    "selftext": post.get("selftext", ""),
                    "num_comments": post.get("num_comments", 0),
                    "url": post.get("url", "")
                })
        return posts
    except Exception as e:
        print(f"Error fetching {feed}: {e}")
        return []


def fetch_comments(post_id: str, limit: int = 50) -> list:
    """Fetch top-level comments from a post."""
    url = f"https://www.reddit.com/r/nba/comments/{post_id}.json"
    try:
        response = requests.get(url, headers=REDDIT_HEADERS, params={"limit": limit}, timeout=10)
        response.raise_for_status()
        data = response.json()
        comments = []
        if isinstance(data, list) and len(data) > 1:
            comments_data = data[1].get("data", {}).get("children", [])
            for item in comments_data:
                comment = item.get("data", {})
                if comment and comment.get("body") and comment["body"] not in ("[removed]", "[deleted]"):
                    comments.append(comment["body"])
        return comments
    except Exception as e:
        print(f"Error fetching comments for {post_id}: {e}")
        return []


def build_candidate_pool(posts: list) -> list:
    """
    Extract titles, selftexts, and comments from posts into a deduplicated candidate pool.
    """
    seen_hashes = set()
    candidates = []

    # Extract titles and selftexts
    for post in posts:
        title = post.get("title", "").strip()
        selftext = post.get("selftext", "").strip()

        if title and len(title) >= MIN_TEXT_LENGTH:
            h = hashlib.md5(title.encode()).hexdigest()
            if h not in seen_hashes:
                seen_hashes.add(h)
                candidates.append((title, "title", post.get("id")))

        if selftext and selftext not in ("[removed]", "[deleted]") and len(selftext) >= MIN_TEXT_LENGTH:
            h = hashlib.md5(selftext.encode()).hexdigest()
            if h not in seen_hashes:
                seen_hashes.add(h)
                candidates.append((selftext, "selftext", post.get("id")))

    # Extract top comments from top posts
    top_posts = sorted(posts, key=lambda p: p.get("num_comments", 0), reverse=True)[:60]
    for post in top_posts:
        comments = fetch_comments(post.get("id"), limit=50)
        for comment in comments:
            comment = comment.strip()
            if len(comment) >= MIN_TEXT_LENGTH:
                h = hashlib.md5(comment.encode()).hexdigest()
                if h not in seen_hashes:
                    seen_hashes.add(h)
                    candidates.append((comment, "comment", post.get("id")))
        time.sleep(0.5)  # Mild delay between comment fetches

    return [c[0] for c in candidates]


def classify_with_groq(text: str, client: Groq, retries: int = 4) -> tuple:
    """
    Classify text using Groq API with exponential backoff retry logic.
    Returns (label, notes) tuple.
    """
    for attempt in range(retries):
        try:
            completion = client.chat.completions.create(
                model=GROQ_MODEL,
                max_tokens=50,
                messages=[
                    {"role": "system", "content": GROQ_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f'Classify this Reddit r/nba text:\n"""\n{text}\n"""'
                    }
                ]
            )
            response = completion.choices[0].message.content.strip().lower()

            # Parse response: look for any valid label substring
            for label in VALID_LABELS:
                if label in response:
                    return (label, "")

            # If no valid label found, mark for review
            return ("REVIEW", f"groq_response: {response[:50]}")

        except Exception as e:
            err_str = str(e)
            if "rate_limit" in err_str.lower() or "429" in err_str:
                if attempt < retries - 1:
                    wait_time = 2 ** (attempt + 1)
                    print(f"  Rate limited. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                return ("FAILED", "rate_limit_exceeded")
            return ("FAILED", err_str[:50])
    return ("FAILED", "max_retries_exceeded")


def load_existing_csv(path: str) -> tuple:
    """
    Load existing dataset.csv and return (set of seen texts, list of rows).
    Validates labels to skip artifacts like '/content/sample_data'.
    """
    seen_texts = set()
    rows = []

    if not os.path.exists(path):
        return seen_texts, rows

    try:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Skip rows with invalid labels (like the /content/sample_data artifact)
                if row.get("label") in VALID_LABELS:
                    text = row.get("text", "").strip()
                    if text:
                        h = hashlib.md5(text.encode()).hexdigest()
                        seen_texts.add(h)
                        rows.append(row)
    except Exception as e:
        print(f"Warning: could not load existing CSV: {e}")

    return seen_texts, rows


def save_csv(rows: list, path: str) -> None:
    """Save rows to CSV file."""
    if not rows:
        print("No rows to save.")
        return

    fieldnames = ["text", "label", "source", "notes"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Saved {len(rows)} rows to {path}")


def main():
    # Initialize Groq client
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY environment variable not set.")
        print("Set it with: export GROQ_API_KEY='your_key_here'")
        return

    client = Groq(api_key=api_key)
    print("Initialized Groq client.")

    # Fetch posts from three feeds
    print("\nFetching r/nba posts from hot, top, and rising feeds...")
    hot_posts = fetch_posts("hot")
    time.sleep(1)
    top_posts = fetch_posts("top", time_filter="week")
    time.sleep(1)
    rising_posts = fetch_posts("rising")
    print(f"  Hot: {len(hot_posts)} posts")
    print(f"  Top (week): {len(top_posts)} posts")
    print(f"  Rising: {len(rising_posts)} posts")

    all_posts = hot_posts + top_posts + rising_posts

    # Deduplicate by post ID
    seen_ids = set()
    unique_posts = []
    for post in all_posts:
        post_id = post.get("id")
        if post_id and post_id not in seen_ids:
            seen_ids.add(post_id)
            unique_posts.append(post)
    print(f"  Total unique posts: {len(unique_posts)}")

    # Build candidate pool
    print("\nBuilding candidate pool (titles, selftexts, comments)...")
    candidates = build_candidate_pool(unique_posts)
    print(f"  Total candidates: {len(candidates)}")

    # Shuffle for diversity
    random.seed(42)
    random.shuffle(candidates)

    # Load existing CSV to avoid re-labeling
    csv_path = "dataset.csv"
    seen_hashes, existing_rows = load_existing_csv(csv_path)
    print(f"  Loaded {len(existing_rows)} existing labeled examples")

    # Initialize tracking
    new_rows = []
    counts = {"tactical_analysis": 0, "narrative_take": 0, "fan_reaction": 0}

    # Restore counts from existing rows
    for row in existing_rows:
        label = row.get("label")
        if label in counts:
            counts[label] += 1
    print(f"  Current counts: {counts}")

    # Classify candidates
    print("\nClassifying candidates with Groq...")
    processed = 0
    for candidate in candidates:
        # Skip if already seen
        h = hashlib.md5(candidate.encode()).hexdigest()
        if h in seen_hashes:
            continue

        # Classify with Groq
        label, notes = classify_with_groq(candidate, client)
        processed += 1

        # Balance check: skip if this label's bucket is full
        if label in counts and counts[label] >= TARGET_PER_CLASS:
            continue

        # Add row
        row = {
            "text": candidate,
            "label": label,
            "source": "reddit_r_nba",
            "notes": notes
        }
        new_rows.append(row)
        seen_hashes.add(h)

        if label in counts:
            counts[label] += 1

        # Progress logging
        if processed % 25 == 0:
            print(f"[Progress] Processed: {processed} | Labeled: {len(new_rows)} | "
                  f"tactical_analysis: {counts['tactical_analysis']} | "
                  f"narrative_take: {counts['narrative_take']} | "
                  f"fan_reaction: {counts['fan_reaction']}")

        # Sleep between API calls
        time.sleep(1)

        # Exit when balanced
        if all(v >= TARGET_PER_CLASS for v in counts.values()):
            print(f"\nAll label buckets reached target ({TARGET_PER_CLASS} examples each).")
            break

    # Combine existing and new rows
    all_rows = existing_rows + new_rows

    # Save to CSV
    print(f"\nFinal counts: {counts}")
    print(f"Total labeled examples: {len(all_rows)}")
    save_csv(all_rows, csv_path)

    # Balance check
    if any(v > 70 for v in counts.values()):
        print("\nWarning: One label exceeds 70% of dataset. Consider collecting more examples for underrepresented classes.")

    # Next steps
    print("\nNext steps:")
    print("1. Open dataset.csv in Google Sheets")
    print("2. Review all rows labeled 'REVIEW' or 'FAILED'")
    print("3. Spot-check ~20 random rows")
    print("4. Delete 'source' and 'notes' columns before uploading to Colab")
    print("5. Upload to Colab and run training pipeline")


if __name__ == "__main__":
    main()
