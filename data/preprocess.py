"""
preprocess.py
-------------
SENTRY Project - Data Preprocessing Pipeline
Author: Yewon Joung

Purpose:
    - Load and clean the Davidson hate speech dataset
    - Encode labels (hate->0, offensive->1, neutral->2)
    - Construct tweet sets (group K tweets together)
    - Split into Train / Validation / Test (70/15/15)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# ── 1. Load and Clean Data ────────────────────────────────────────────────

def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """
    Load a CSV file and apply basic cleaning steps.

    Args:
        filepath: path to labeled_data.csv

    Returns:
        cleaned DataFrame with columns: tweet, label
    """
    df = pd.read_csv(filepath)

    # Keep only relevant columns from the Davidson dataset
    # Original columns: 'tweet', 'class' (0=hate, 1=offensive, 2=neither)
    df = df[['tweet', 'class']].copy()
    df.columns = ['tweet', 'label']

    # Remove rows with missing values
    df.dropna(subset=['tweet', 'label'], inplace=True)

    # Remove duplicate tweets
    df.drop_duplicates(subset=['tweet'], inplace=True)

    # Strip leading/trailing whitespace
    df['tweet'] = df['tweet'].str.strip()

    # Remove empty strings
    df = df[df['tweet'] != '']

    print(f"[Preprocessing Complete] {len(df)} tweets loaded")
    print(f"  - Hate (0):      {(df['label'] == 0).sum()} ({(df['label'] == 0).mean()*100:.1f}%)")
    print(f"  - Offensive (1): {(df['label'] == 1).sum()} ({(df['label'] == 1).mean()*100:.1f}%)")
    print(f"  - Neutral (2):   {(df['label'] == 2).sum()} ({(df['label'] == 2).mean()*100:.1f}%)")

    return df.reset_index(drop=True)


# ── 2. Build Tweet Sets ───────────────────────────────────────────────────

def build_sets(df: pd.DataFrame, set_size: int = 5) -> list[dict]:
    """
    Group tweets of the same class into sets of size K.

    Args:
        df:       cleaned DataFrame
        set_size: number of tweets per set (default K=5)

    Returns:
        list of sets, where each set is {'tweets': [...], 'label': int}
    """
    sets = []

    for label in df['label'].unique():
        # Extract tweets belonging to this class
        class_tweets = df[df['label'] == label]['tweet'].tolist()

        # Group into sets of K (discard remainder)
        num_sets = len(class_tweets) // set_size
        for i in range(num_sets):
            tweet_group = class_tweets[i * set_size: (i + 1) * set_size]
            sets.append({
                'tweets': tweet_group,
                'label': int(label)
            })

    # Shuffle to avoid class-ordered batches
    np.random.seed(42)
    np.random.shuffle(sets)

    print(f"\n[Set Construction Complete] K={set_size}, total {len(sets)} sets")
    label_counts = {}
    for s in sets:
        label_counts[s['label']] = label_counts.get(s['label'], 0) + 1
    for label, count in sorted(label_counts.items()):
        label_name = {0: 'Hate', 1: 'Offensive', 2: 'Neutral'}[label]
        print(f"  - {label_name}: {count} sets")

    return sets


# ── 3. Train / Val / Test Split ───────────────────────────────────────────

def split_sets(
    sets: list[dict],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42
) -> tuple[list, list, list]:
    """
    Split the set list into train / validation / test partitions.

    Stratified splitting ensures class distribution is preserved across splits.

    Args:
        sets:         output of build_sets()
        train_ratio:  proportion for training (default 0.70)
        val_ratio:    proportion for validation (default 0.15)
        test_ratio:   proportion for testing (default 0.15)
        random_state: random seed for reproducibility

    Returns:
        (train_sets, val_sets, test_sets)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    labels = [s['label'] for s in sets]

    # First split: train vs (val + test)
    train_sets, temp_sets = train_test_split(
        sets,
        test_size=(val_ratio + test_ratio),
        stratify=labels,
        random_state=random_state
    )

    # Second split: val vs test
    temp_labels = [s['label'] for s in temp_sets]
    val_sets, test_sets = train_test_split(
        temp_sets,
        test_size=(test_ratio / (val_ratio + test_ratio)),
        stratify=temp_labels,
        random_state=random_state
    )

    print(f"\n[Split Complete]")
    print(f"  - Train: {len(train_sets)} sets (~{len(train_sets)*5} tweets)")
    print(f"  - Val:   {len(val_sets)} sets (~{len(val_sets)*5} tweets)")
    print(f"  - Test:  {len(test_sets)} sets (~{len(test_sets)*5} tweets)")

    return train_sets, val_sets, test_sets


# ── 4. Entry Point ────────────────────────────────────────────────────────

if __name__ == '__main__':
    DATA_PATH = 'data/labeled_data.csv'
    SET_SIZE  = 5

    df                          = load_and_clean_data(DATA_PATH)
    sets                        = build_sets(df, set_size=SET_SIZE)
    train_sets, val_sets, test_sets = split_sets(sets)
