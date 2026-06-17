"""
augmentation.py
---------------
SENTRY Project - Back-Translation Data Augmentation
Author: Yewon Joung

Purpose:
    The Davidson dataset is heavily imbalanced:
      - Hate speech:      5.77%  (1,430 tweets)  <- minority
      - Offensive:       77.43% (19,190 tweets)  <- majority
      - Neutral:         16.80%  (4,163 tweets)  <- minority

    To address this, we apply back-translation to minority classes (hate, neutral):
      English -> Intermediate language (French / German / Spanish) -> English
    This generates paraphrases that preserve meaning but vary in expression.
"""

import pandas as pd
from transformers import MarianMTModel, MarianTokenizer


# ── 1. Load Translation Models ────────────────────────────────────────────

# MarianMT model names for each intermediate language
TRANSLATION_MODELS = {
    'french':  ('Helsinki-NLP/opus-mt-en-fr', 'Helsinki-NLP/opus-mt-fr-en'),
    'german':  ('Helsinki-NLP/opus-mt-en-de', 'Helsinki-NLP/opus-mt-de-en'),
    'spanish': ('Helsinki-NLP/opus-mt-en-es', 'Helsinki-NLP/opus-mt-es-en'),
}


def load_translation_models(language: str) -> tuple:
    """
    Load a MarianMT forward (en->xx) and backward (xx->en) model pair.

    Args:
        language: one of 'french', 'german', 'spanish'

    Returns:
        (forward_model, forward_tokenizer, backward_model, backward_tokenizer)
    """
    assert language in TRANSLATION_MODELS, \
        f"Unsupported language: {language}. Choose from {list(TRANSLATION_MODELS.keys())}"

    fwd_name, bwd_name = TRANSLATION_MODELS[language]

    print(f"[Loading] {language} translation models...")

    fwd_tokenizer = MarianTokenizer.from_pretrained(fwd_name)
    fwd_model     = MarianMTModel.from_pretrained(fwd_name)

    bwd_tokenizer = MarianTokenizer.from_pretrained(bwd_name)
    bwd_model     = MarianMTModel.from_pretrained(bwd_name)

    print(f"[Done] {language} models loaded.")

    return fwd_model, fwd_tokenizer, bwd_model, bwd_tokenizer


# ── 2. Translation Helper ─────────────────────────────────────────────────

def translate(
    texts: list[str],
    model: MarianMTModel,
    tokenizer: MarianTokenizer,
    batch_size: int = 16,
    max_length: int = 128
) -> list[str]:
    """
    Translate a list of texts using a MarianMT model.

    Args:
        texts:      list of input strings
        model:      MarianMT model
        tokenizer:  corresponding tokenizer
        batch_size: number of sentences per batch
        max_length: max token length for generation

    Returns:
        list of translated strings
    """
    translated = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]

        inputs = tokenizer(
            batch,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=max_length
        )

        outputs = model.generate(**inputs, max_length=max_length)

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        translated.extend(decoded)

    return translated


# ── 3. Back-Translation ───────────────────────────────────────────────────

def back_translate(
    texts: list[str],
    language: str = 'french'
) -> list[str]:
    """
    Apply back-translation to a list of English texts.

    Pipeline:
        English -> Intermediate language -> English

    Args:
        texts:    list of original English tweets
        language: intermediate language ('french', 'german', 'spanish')

    Returns:
        list of back-translated English strings
    """
    fwd_model, fwd_tokenizer, bwd_model, bwd_tokenizer = load_translation_models(language)

    # Step 1: English -> Intermediate language
    print(f"[Step 1] Translating to {language}...")
    intermediate = translate(texts, fwd_model, fwd_tokenizer)

    # Step 2: Intermediate language -> English
    print(f"[Step 2] Translating back to English...")
    back_translated = translate(intermediate, bwd_model, bwd_tokenizer)

    return back_translated


# ── 4. Quality Filter ─────────────────────────────────────────────────────

def filter_augmented(
    originals: list[str],
    augmented: list[str],
    min_length: int = 5
) -> tuple[list[str], list[str]]:
    """
    Remove low-quality back-translated outputs.

    Filtering criteria:
        1. Too short (fewer than min_length characters)
        2. Identical to the original (no paraphrase effect)

    Args:
        originals:  original English tweets
        augmented:  back-translated tweets
        min_length: minimum character length to keep

    Returns:
        (filtered_originals, filtered_augmented) — paired lists
    """
    filtered_orig = []
    filtered_aug  = []

    removed = 0
    for orig, aug in zip(originals, augmented):
        if len(aug.strip()) < min_length:
            removed += 1
            continue
        if aug.strip().lower() == orig.strip().lower():
            removed += 1
            continue
        filtered_orig.append(orig)
        filtered_aug.append(aug)

    print(f"[Filter] Removed {removed} low-quality samples. "
          f"Kept {len(filtered_aug)} / {len(originals)}")

    return filtered_orig, filtered_aug


# ── 5. Full Augmentation Pipeline ────────────────────────────────────────

def augment_minority_classes(
    df: pd.DataFrame,
    multipliers: dict = {0: 5, 2: 2},
    languages: list[str] = ['french', 'german', 'spanish']
) -> pd.DataFrame:
    """
    Augment minority classes using back-translation and append to the dataset.

    Args:
        df:          cleaned DataFrame with columns ['tweet', 'label']
        multipliers: how many times to augment each class
                     e.g. {0: 5, 2: 2} means:
                       - hate (0)    augmented x5
                       - neutral (2) augmented x2
        languages:   intermediate languages to cycle through

    Returns:
        augmented DataFrame combining original + new samples
    """
    augmented_rows = []

    for label, multiplier in multipliers.items():
        label_name = {0: 'Hate', 1: 'Offensive', 2: 'Neutral'}[label]
        tweets = df[df['label'] == label]['tweet'].tolist()

        print(f"\n[Augmenting] Class: {label_name} | "
              f"Original: {len(tweets)} | Target multiplier: x{multiplier}")

        for i in range(multiplier):
            # Cycle through languages for variety
            lang = languages[i % len(languages)]

            back_translated = back_translate(tweets, language=lang)

            _, filtered = filter_augmented(tweets, back_translated)

            for aug_tweet in filtered:
                augmented_rows.append({'tweet': aug_tweet, 'label': label})

            print(f"  Round {i+1}/{multiplier} ({lang}): +{len(filtered)} samples")

    # Combine original + augmented
    aug_df = pd.DataFrame(augmented_rows)
    result_df = pd.concat([df, aug_df], ignore_index=True)

    print(f"\n[Augmentation Complete]")
    print(f"  Before: {len(df)} tweets")
    print(f"  After:  {len(result_df)} tweets (+{len(aug_df)} augmented)")
    print(f"\n  Class distribution after augmentation:")
    for lbl in sorted(result_df['label'].unique()):
        lname = {0: 'Hate', 1: 'Offensive', 2: 'Neutral'}[lbl]
        count = (result_df['label'] == lbl).sum()
        pct   = (result_df['label'] == lbl).mean() * 100
        print(f"    - {lname}: {count} ({pct:.1f}%)")

    return result_df


# ── 6. Entry Point ────────────────────────────────────────────────────────

if __name__ == '__main__':
    from preprocess import load_and_clean_data

    DATA_PATH = 'data/labeled_data.csv'

    # Load cleaned data
    df = load_and_clean_data(DATA_PATH)

    # Augment minority classes
    df_augmented = augment_minority_classes(
        df,
        multipliers={0: 5, 2: 2},          # hate x5, neutral x2
        languages=['french', 'german', 'spanish']
    )

    # Save augmented dataset
    df_augmented.to_csv('data/labeled_data_augmented.csv', index=False)
    print("\n[Saved] data/labeled_data_augmented.csv")
