"""
run_davidson.py
---------------
SENTRY Project - Davidson Dataset Experiments (Exp 1-6)
Author: Yewon Joung

Experiment Configurations:
    Exp 1 - Baseline:                   Sets + Cross Entropy + Mean Pooling
    Exp 2 - DPO:                        Sets + DPO Loss + Mean Pooling
    Exp 3 - Attention + DPO:            Sets + DPO Loss + Attention Pooling
    Exp 4 - Equivariant + Mean:         Equivariant Encoder + Mean Pooling
    Exp 5 - Equivariant + Attention:    Equivariant Encoder + Attention Pooling + DPO (BEST)
    Exp 6 - Equivariant + CE only:      Equivariant Encoder + Cross Entropy (ablation: no DPO)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import f1_score, accuracy_score, classification_report
import numpy as np


# ── 1. Configuration ──────────────────────────────────────────────────────

EXPERIMENTS = {
    'exp1': {
        'name':        'Baseline (Sets + CE + Mean Pooling)',
        'pooling':     'mean',
        'equivariant': False,
        'use_dpo':     False,
    },
    'exp2': {
        'name':        'Sets + DPO + Mean Pooling',
        'pooling':     'mean',
        'equivariant': False,
        'use_dpo':     True,
    },
    'exp3': {
        'name':        'Sets + DPO + Attention Pooling',
        'pooling':     'attention',
        'equivariant': False,
        'use_dpo':     True,
    },
    'exp4': {
        'name':        'Equivariant + Mean Pooling',
        'pooling':     'mean',
        'equivariant': True,
        'use_dpo':     True,
    },
    'exp5': {
        'name':        'Equivariant + Attention Pooling + DPO (BEST)',
        'pooling':     'attention',
        'equivariant': True,
        'use_dpo':     True,
    },
    'exp6': {
        'name':        'Equivariant + CE only (no DPO)',
        'pooling':     'mean',
        'equivariant': True,
        'use_dpo':     False,
    },
}

# Default hyperparameters (same across all experiments)
CONFIG = {
    'encoder_name':    'distilbert-base-uncased',
    'hidden_dim':      768,
    'set_size':        5,
    'num_classes':     3,
    'batch_size':      8,
    'epochs':          10,
    'lr_encoder':      2e-6,
    'lr_classifier':   2e-5,
    'lambda_ce':       1.0,
    'lambda_tweet':    0.1,
    'lambda_set':      0.5,
    'beta_dpo':        0.1,
    'max_length':      128,
    'device':          'cuda' if torch.cuda.is_available() else 'cpu',
}


# ── 2. Dataset ────────────────────────────────────────────────────────────

class TweetSetDataset(Dataset):
    """
    PyTorch Dataset that returns sets of tweets with a shared label.

    Each item is a dict:
        {
            'tweets': list of K tweet strings,
            'label':  int (0=hate, 1=offensive, 2=neutral)
        }
    """

    def __init__(self, sets: list[dict]):
        self.sets = sets

    def __len__(self):
        return len(self.sets)

    def __getitem__(self, idx):
        return self.sets[idx]


def collate_fn(batch: list[dict], tokenizer, max_length: int = 128):
    """
    Custom collate function: tokenizes all tweets in a batch of sets.

    Args:
        batch:      list of set dicts from TweetSetDataset
        tokenizer:  HuggingFace tokenizer
        max_length: max token length

    Returns:
        dict with keys: input_ids, attention_mask, labels
        shapes: (batch_size * set_size, max_length), (batch_size,)
    """
    all_tweets = []
    labels = []

    for item in batch:
        all_tweets.extend(item['tweets'])       # flatten all tweets
        labels.append(item['label'])

    encoded = tokenizer(
        all_tweets,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )

    return {
        'input_ids':      encoded['input_ids'],
        'attention_mask': encoded['attention_mask'],
        'labels':         torch.tensor(labels, dtype=torch.long)
    }


# ── 3. Model Components ───────────────────────────────────────────────────

class EquivariantLayer(nn.Module):
    """
    DeepSets-style permutation-equivariant transformation.

    For each tweet embedding e_i in a set {e_1, ..., e_K}:
        context = sum(g(e_k) for all k)        <- global set context
        e'_i    = phi(e_i, context)             <- contextualized embedding

    This ensures: reordering tweets produces correspondingly reordered outputs.
    (i.e., the model doesn't depend on tweet order)
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        # g: tweet embedding -> context vector
        self.g   = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        # phi: (tweet embedding + context) -> new embedding
        self.phi = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU()
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: (batch_size, set_size, hidden_dim)

        Returns:
            contextualized embeddings: (batch_size, set_size, hidden_dim)
        """
        # Global context: sum of g(e_i) across set dimension
        context = self.g(embeddings).sum(dim=1, keepdim=True)  # (B, 1, D)
        context = context.expand_as(embeddings)                 # (B, K, D)

        # Concatenate each tweet with global context, then project
        combined = torch.cat([embeddings, context], dim=-1)     # (B, K, 2D)
        return self.phi(combined)                               # (B, K, D)


class AttentionPooling(nn.Module):
    """
    Learns importance weights for each tweet in a set,
    then computes a weighted average as the set representation.

    alpha_i = softmax(w^T * e'_i)
    phi(S)  = sum(alpha_i * e'_i)
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: (batch_size, set_size, hidden_dim)

        Returns:
            set representation: (batch_size, hidden_dim)
        """
        scores  = self.attention(embeddings)            # (B, K, 1)
        weights = F.softmax(scores, dim=1)              # (B, K, 1)
        pooled  = (weights * embeddings).sum(dim=1)     # (B, D)
        return pooled


class SENTRYModel(nn.Module):
    """
    Full SENTRY model:
        DistilBERT encoder
        -> (optional) EquivariantLayer
        -> Mean or Attention Pooling
        -> MLP Classifier
    """

    def __init__(self, config: dict, use_equivariant: bool, pooling: str):
        super().__init__()

        self.set_size        = config['set_size']
        self.hidden_dim      = config['hidden_dim']
        self.use_equivariant = use_equivariant
        self.pooling_type    = pooling

        # Encoder
        self.encoder = AutoModel.from_pretrained(config['encoder_name'])

        # Optional: equivariant contextualization
        if use_equivariant:
            self.equivariant = EquivariantLayer(self.hidden_dim)

        # Pooling
        if pooling == 'attention':
            self.pooling = AttentionPooling(self.hidden_dim)

        # MLP Classifier: 768 -> 256 -> 128 -> num_classes
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, config['num_classes'])
        )

    def encode_tweets(self, input_ids, attention_mask):
        """
        Encode all tweets in a batch using DistilBERT.
        Returns mean-pooled token embeddings per tweet.

        Returns:
            tweet embeddings: (batch_size * set_size, hidden_dim)
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Mean pooling over token dimension
        token_embeddings = outputs.last_hidden_state            # (B*K, seq_len, D)
        mask = attention_mask.unsqueeze(-1).float()
        tweet_embeddings = (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1)
        return tweet_embeddings                                 # (B*K, D)

    def forward(self, input_ids, attention_mask):
        """
        Full forward pass.

        Args:
            input_ids:      (batch_size * set_size, seq_len)
            attention_mask: (batch_size * set_size, seq_len)

        Returns:
            logits:           (batch_size, num_classes)
            tweet_embeddings: (batch_size, set_size, hidden_dim)
            set_embeddings:   (batch_size, hidden_dim)
        """
        batch_size = input_ids.size(0) // self.set_size

        # 1. Encode tweets
        tweet_embs = self.encode_tweets(input_ids, attention_mask)  # (B*K, D)

        # 2. Reshape to (batch_size, set_size, hidden_dim)
        tweet_embs = tweet_embs.view(batch_size, self.set_size, self.hidden_dim)

        # 3. Optional: equivariant contextualization
        if self.use_equivariant:
            tweet_embs = self.equivariant(tweet_embs)

        # 4. Pooling -> set representation
        if self.pooling_type == 'attention':
            set_embs = self.pooling(tweet_embs)                     # (B, D)
        else:
            set_embs = tweet_embs.mean(dim=1)                       # (B, D)

        # 5. Classify
        logits = self.classifier(set_embs)                          # (B, num_classes)

        return logits, tweet_embs, set_embs


# ── 4. DPO Loss ───────────────────────────────────────────────────────────

def dpo_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    beta: float = 0.1
) -> torch.Tensor:
    """
    Hierarchical DPO (Direct Preference Optimization) contrastive loss.

    For each anchor embedding:
        - Positive: embedding from the SAME class
        - Negative: embedding from a DIFFERENT class

    Loss = -log(sigmoid(beta * (sim(anchor, pos) - sim(anchor, neg))))

    This pushes same-class embeddings closer and different-class embeddings apart.

    Args:
        embeddings: (batch_size, hidden_dim)  — tweet-level OR set-level
        labels:     (batch_size,)
        beta:       temperature parameter

    Returns:
        scalar loss value
    """
    losses = []

    for i in range(len(embeddings)):
        anchor = embeddings[i]
        label  = labels[i]

        # Find positive and negative indices
        pos_mask = (labels == label)
        neg_mask = (labels != label)

        pos_mask[i] = False  # exclude self

        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            continue  # skip if no valid pair

        # Randomly sample one positive and one negative
        pos_idx = pos_mask.nonzero(as_tuple=True)[0][
            torch.randint(pos_mask.sum(), (1,))
        ]
        neg_idx = neg_mask.nonzero(as_tuple=True)[0][
            torch.randint(neg_mask.sum(), (1,))
        ]

        pos_emb = embeddings[pos_idx.item()]
        neg_emb = embeddings[neg_idx.item()]

        sim_pos = F.cosine_similarity(anchor.unsqueeze(0), pos_emb.unsqueeze(0))
        sim_neg = F.cosine_similarity(anchor.unsqueeze(0), neg_emb.unsqueeze(0))

        loss = -F.logsigmoid(beta * (sim_pos - sim_neg))
        losses.append(loss)

    if len(losses) == 0:
        return torch.tensor(0.0, requires_grad=True)

    return torch.stack(losses).mean()


# ── 5. Training & Evaluation ──────────────────────────────────────────────

def train_one_epoch(model, dataloader, optimizer, config, use_dpo):
    model.train()
    total_loss = 0.0
    ce_loss_fn = nn.CrossEntropyLoss()

    for batch in dataloader:
        input_ids      = batch['input_ids'].to(config['device'])
        attention_mask = batch['attention_mask'].to(config['device'])
        labels         = batch['labels'].to(config['device'])

        optimizer.zero_grad()

        logits, tweet_embs, set_embs = model(input_ids, attention_mask)

        # Cross Entropy loss (always on)
        loss = config['lambda_ce'] * ce_loss_fn(logits, labels)

        # DPO losses (optional)
        if use_dpo:
            # Tweet-level: use mean of each set's tweet embeddings
            tweet_embs_flat = tweet_embs.mean(dim=1)  # (B, D)
            loss += config['lambda_tweet'] * dpo_loss(
                tweet_embs_flat, labels, config['beta_dpo']
            )
            # Set-level
            loss += config['lambda_set'] * dpo_loss(
                set_embs, labels, config['beta_dpo']
            )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, config):
    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids      = batch['input_ids'].to(config['device'])
            attention_mask = batch['attention_mask'].to(config['device'])
            labels         = batch['labels']

            logits, _, _ = model(input_ids, attention_mask)
            preds = logits.argmax(dim=-1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    acc      = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    report   = classification_report(
        all_labels, all_preds,
        target_names=['Hate', 'Offensive', 'Neutral']
    )

    return acc, macro_f1, report


# ── 6. Run Single Experiment ──────────────────────────────────────────────

def run_experiment(exp_key: str, train_sets, val_sets, test_sets):
    """
    Run a single experiment end-to-end.

    Args:
        exp_key:    one of 'exp1' ... 'exp6'
        train_sets: output of split_sets() — training portion
        val_sets:   output of split_sets() — validation portion
        test_sets:  output of split_sets() — test portion
    """
    exp_config = EXPERIMENTS[exp_key]
    print(f"\n{'='*60}")
    print(f"Running: {exp_config['name']}")
    print(f"{'='*60}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['encoder_name'])

    # Dataloaders
    def make_loader(sets, shuffle):
        return DataLoader(
            TweetSetDataset(sets),
            batch_size=CONFIG['batch_size'],
            shuffle=shuffle,
            collate_fn=lambda b: collate_fn(b, tokenizer, CONFIG['max_length'])
        )

    train_loader = make_loader(train_sets, shuffle=True)
    val_loader   = make_loader(val_sets,   shuffle=False)
    test_loader  = make_loader(test_sets,  shuffle=False)

    # Model
    model = SENTRYModel(
        config=CONFIG,
        use_equivariant=exp_config['equivariant'],
        pooling=exp_config['pooling']
    ).to(CONFIG['device'])

    # Optimizer: different learning rates for encoder vs classifier
    optimizer = torch.optim.AdamW([
        {'params': model.encoder.parameters(),    'lr': CONFIG['lr_encoder']},
        {'params': [p for n, p in model.named_parameters()
                    if 'encoder' not in n],        'lr': CONFIG['lr_classifier']},
    ])

    # Training loop
    best_val_f1 = 0.0
    best_model_state = None

    for epoch in range(CONFIG['epochs']):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, CONFIG, exp_config['use_dpo']
        )
        val_acc, val_f1, _ = evaluate(model, val_loader, CONFIG)

        print(f"  Epoch {epoch+1:02d}/{CONFIG['epochs']} | "
              f"Loss: {train_loss:.4f} | "
              f"Val Acc: {val_acc*100:.1f}% | "
              f"Val Macro-F1: {val_f1*100:.1f}%")

        # Save best model based on validation F1
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Evaluate best model on test set
    model.load_state_dict(best_model_state)
    test_acc, test_f1, report = evaluate(model, test_loader, CONFIG)

    print(f"\n  ── Test Results ──")
    print(f"  Accuracy:  {test_acc*100:.1f}%")
    print(f"  Macro-F1:  {test_f1*100:.1f}%")
    print(f"\n{report}")

    return {'accuracy': test_acc, 'macro_f1': test_f1, 'report': report}


# ── 7. Run All Experiments ────────────────────────────────────────────────

def run_all_experiments(train_sets, val_sets, test_sets):
    """
    Run all 6 experiments sequentially and print a summary table.
    """
    results = {}

    for exp_key in EXPERIMENTS:
        results[exp_key] = run_experiment(exp_key, train_sets, val_sets, test_sets)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Experiment':<45} {'Accuracy':>9} {'Macro-F1':>9}")
    print('-' * 60)
    for exp_key, result in results.items():
        name = EXPERIMENTS[exp_key]['name']
        print(f"{name:<45} "
              f"{result['accuracy']*100:>8.1f}% "
              f"{result['macro_f1']*100:>8.1f}%")

    return results


# ── 8. Entry Point ────────────────────────────────────────────────────────

if __name__ == '__main__':
    from preprocess import load_and_clean_data, build_sets, split_sets
    from augmentation import augment_minority_classes

    # Load and preprocess
    df = load_and_clean_data('data/labeled_data.csv')

    # Augment minority classes
    df = augment_minority_classes(df, multipliers={0: 5, 2: 2})

    # Build sets and split
    sets = build_sets(df, set_size=CONFIG['set_size'])
    train_sets, val_sets, test_sets = split_sets(sets)

    # Run all experiments
    run_all_experiments(train_sets, val_sets, test_sets)
