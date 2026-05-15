import os
import torch
import torch.nn.functional as F
import transformer_lens
from transformer_lens.model_bridge import TransformerBridge
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from sklearn.decomposition import PCA
from scipy.stats import ks_2samp
import json
import string

os.chdir("/Users/sundareswara/Documents/Code/Interpretability/")

# ─────────────────────────────────────────────────────────────
# Load prose passages
# ─────────────────────────────────────────────────────────────
with open("prose.json", "r", encoding="utf-8") as f:
    passages = json.load(f)

p            = passages[0]
human_prose  = p["human"]
gemini_prose = p["gemini"]
claude_prose = p["claude"]

# Named pairs for all analyses
PAIR_NAMES = [
    ("Human",  "Gemini"),
    ("Human",  "Claude"),
    ("Gemini", "Claude"),
]


# ─────────────────────────────────────────────────────────────
# Helper: look up prose / activations / df by label
# ─────────────────────────────────────────────────────────────
def get_prose(label):
    return {"Human": human_prose, "Gemini": gemini_prose, "Claude": claude_prose}[label]

def get_activations(label):
    return {"Human": human_act, "Gemini": gemini_act, "Claude": claude_act}[label]

def get_df(label):
    return {"Human": df_human, "Gemini": df_gemini, "Claude": df_claude}[label]

def get_tokens(label):
    return {"Human": human_tokens, "Gemini": gemini_tokens, "Claude": claude_tokens}[label]


# ── Token filtering utility ─────────────────────────────────

STOPWORDS = {
    'the','a','an','and','or','but','in','on','at','to','for',
    'of','with','by','from','as','is','was','are','were','be',
    'been','have','has','had','that','this','it','its','she',
    'he','her','his','they','their','them','we','our','i','my',
    'so','not','no','if','into','who','which','when','what',
    'how','all','up','out','him','its','just','also','such',
    'then','than','her','him','now','over','after','before',
}

def is_content_token(tok):
    """Keep only whole, non-stopword, non-punctuation words."""
    # Whole words start with a space in this bridge's token format
    if not tok.startswith(' '):
        return False                   # subword fragment — skip

    clean = tok.strip()

    if not clean:                                    return False
    if all(c in string.punctuation for c in clean): return False  # catches '--', '...', etc.
    if not any(c.isalpha() for c in clean):         return False  # catches '123', '42'
    if clean.lower() in STOPWORDS:                  return False
    return True


def content_mask(tokens):
    """Return boolean numpy array — True where token passes content filter."""
    return np.array([
        t != '<|endoftext|>' and is_content_token(t)
        for t in tokens
    ])



# ─────────────────────────────────────────────────────────────
# Analysis functions
# ─────────────────────────────────────────────────────────────
def per_token_nll_ppl(model, text, prepend_bos=True):
    """Per-token NLL and perplexity for a text under a TransformerLens model."""
    tokens       = model.to_tokens(text, prepend_bos=prepend_bos)
    with torch.no_grad():
        logits   = model(tokens)
    log_probs    = torch.log_softmax(logits, dim=-1)
    target_tokens   = tokens[:, 1:]
    pred_log_probs  = log_probs[:, :-1, :]
    token_log_probs = pred_log_probs.gather(
        dim=-1, index=target_tokens.unsqueeze(-1)
    ).squeeze(-1)
    nll       = -token_log_probs.squeeze(0)
    ppl       = torch.exp(nll)
    token_strs = model.to_str_tokens(tokens.squeeze(0))[1:]
    return pd.DataFrame({
        "token_index": list(range(len(token_strs))),
        "token":       token_strs,
        "nll":         nll.cpu().numpy(),
        "perplexity":  ppl.cpu().numpy(),
    })


def summarize_nll(df, name):
    print(f"\n{name}")
    print(f"  Mean NLL:              {df['nll'].mean():.4f}")
    print(f"  Median NLL:            {df['nll'].median():.4f}")
    print(f"  NLL std:               {df['nll'].std():.4f}")
    print(f"  Max NLL:               {df['nll'].max():.4f}")
    print(f"  Mean per-token PPL:    {df['perplexity'].mean():.2f}")
    whole_ppl = torch.exp(torch.tensor(df["nll"].mean())).item()
    print(f"  Whole-text PPL:        {whole_ppl:.4f}")


def annotate_large_spikes(df, ax, threshold=8):
    for _, row in df.iterrows():
        if row["nll"] > threshold:
            ax.text(row["token_index"], row["nll"], row["token"], fontsize=8)


def surprise_concentration(nll_values, top_frac=0.1):
    nll           = np.array(nll_values)
    sorted_nll    = np.sort(nll)[::-1]
    k             = max(1, int(len(nll) * top_frac))
    return sorted_nll[:k].sum() / nll.sum()


def cumulative_surprise_curve(df, label, ax):
    nll        = np.sort(df["nll"].values)[::-1]
    cumulative = np.cumsum(nll) / np.sum(nll)
    ax.plot(np.linspace(0, 1, len(cumulative)), cumulative, label=label)


# ─────────────────────────────────────────────────────────────
# Plotting functions (all pairwise)
# ─────────────────────────────────────────────────────────────
def plot_nll_pair(df_a, df_b, label_a, label_b):
    """Overlay NLL curves for two translations with spike annotations."""
    fig, ax = plt.subplots(figsize=(18, 5))
    ax.plot(df_a["token_index"], df_a["nll"], label=label_a, marker='o', markersize=3)
    ax.plot(df_b["token_index"], df_b["nll"], label=label_b, marker='x', markersize=3)
    annotate_large_spikes(df_a, ax)
    annotate_large_spikes(df_b, ax)
    ax.set_xlabel("Token Position")
    ax.set_ylabel("NLL")
    ax.set_title(f"Per-token NLL: {label_a} vs {label_b}")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(f"Figures/nll_{label_a.lower()}_{label_b.lower()}.png", dpi=150)
    plt.show()


def plot_nll_delta(df_a, df_b, label_a, label_b):
    """NLL difference plot; positive = A more surprising, negative = B more surprising."""
    min_len = min(len(df_a), len(df_b))
    delta   = df_a["nll"].values[:min_len] - df_b["nll"].values[:min_len]
    fig, ax = plt.subplots(figsize=(18, 4))
    ax.plot(delta, marker='o', markersize=3)
    ax.axhline(0, linestyle='--', color='gray')
    ax.set_xlabel("Token Position")
    ax.set_ylabel(f"{label_a} NLL  −  {label_b} NLL")
    ax.set_title(f"Difference in Token Surprise: {label_a} vs {label_b}\n"
                 f"(positive → {label_a} more surprising, negative → {label_b} more surprising)")
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(f"Figures/delta_{label_a.lower()}_{label_b.lower()}.png", dpi=150)
    plt.show()


def plot_surprise_concentration_pair(df_a, df_b, label_a, label_b):
    fig, ax = plt.subplots(figsize=(8, 5))
    cumulative_surprise_curve(df_a, label_a, ax)
    cumulative_surprise_curve(df_b, label_b, ax)
    ax.set_xlabel("Fraction of tokens (sorted by surprise)")
    ax.set_ylabel("Fraction of total surprise")
    ax.set_title(f"Surprise concentration: {label_a} vs {label_b}")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(f"Figures/concentration_{label_a.lower()}_{label_b.lower()}.png", dpi=150)
    plt.show()


def cosine_sim_profile(act_a, act_b, label_a, label_b, n_layers=12):
    """Mean-pooled cosine similarity at each layer."""
    sims = {}
    for layer in range(n_layers):
        ra = act_a["resid_post", layer][0].mean(dim=0)
        rb = act_b["resid_post", layer][0].mean(dim=0)
        sims[layer] = F.cosine_similarity(ra.unsqueeze(0), rb.unsqueeze(0)).item()
    print(f"\nCosine similarity (mean-pooled) — {label_a} vs {label_b}:")
    for layer, val in sims.items():
        print(f"  Layer {layer:2d}: {val:.4f}")
    return sims



def plot_cosine_sim_all_pairs(all_sims, n_layers=12):
    """Overlay cosine similarity profiles for all three pairs."""
    colors  = {"Human ↔ Gemini": "steelblue",
               "Human ↔ Claude": "tomato",
               "Gemini ↔ Claude": "green"}
    styles  = {"Human ↔ Gemini": "-",
               "Human ↔ Claude": "-",
               "Gemini ↔ Claude": "--"}
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, sims in all_sims.items():
        ax.plot(range(n_layers), list(sims.values()),
                marker='o', color=colors[label],
                linestyle=styles[label], label=label)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine Similarity (mean-pooled)")
    ax.set_title("Pairwise translation similarity by layer")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("Figures/cosine_sim_all_pairs.png", dpi=150)
    plt.show()


def pca_at_layer(act_a, act_b, label_a, label_b, layer):
    act1    = act_a["resid_post", layer][0].detach().numpy()
    act2    = act_b["resid_post", layer][0].detach().numpy()
    X       = np.vstack([act1, act2])
    labels  = [0] * len(act1) + [1] * len(act2)
    tokens  = list(get_tokens(label_a)) + list(get_tokens(label_b))
    pca     = PCA(n_components=2)
    X_2d    = pca.fit_transform(X)
    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, (color, lbl) in enumerate(zip(['steelblue', 'tomato'], [label_a, label_b])):
        mask = [i for i, l in enumerate(labels) if l == idx]
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=color, label=lbl, alpha=0.7, s=60)
        for i in mask:
            ax.annotate(tokens[i], (X_2d[i, 0], X_2d[i, 1]), fontsize=6, alpha=0.6)
    var = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({var[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({var[1]:.1%} var)")
    ax.set_title(f"Token activations in PCA space — Layer {layer}: {label_a} vs {label_b}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"Figures/pca_L{layer}_{label_a.lower()}_{label_b.lower()}.png", dpi=150)
    plt.show()



def prose_divergence(act_a, act_b, label_a, label_b, n_layers=12):
    traj1     = np.stack([act_a["resid_post", l][0].mean(0).detach().numpy()
                          for l in range(n_layers)])
    traj2     = np.stack([act_b["resid_post", l][0].mean(0).detach().numpy()
                          for l in range(n_layers)])
    diffs     = traj1 - traj2
    pca_diff  = PCA(n_components=3)
    proj      = pca_diff.fit_transform(diffs)
    fig, ax   = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(proj[:, 0], proj[:, 1], c=range(n_layers),
                    cmap='plasma', s=80, zorder=3)
    for i, (x, y) in enumerate(proj[:, :2]):
        ax.annotate(f"L{i}", (x, y), fontsize=8)
    plt.colorbar(sc, label='Layer')
    ax.set_xlabel(f"Diff-PC1 ({pca_diff.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"Diff-PC2 ({pca_diff.explained_variance_ratio_[1]:.1%})")
    ax.set_title(f"PCA of difference vectors: {label_a} vs {label_b}")
    plt.tight_layout()
    plt.savefig(f"Figures/diff_pca_{label_a.lower()}_{label_b.lower()}.png", dpi=150)
    plt.show()
    diff_norms = np.linalg.norm(diffs, axis=1)
    print(f"\nDifference magnitude by layer ({label_a} vs {label_b}):")
    for l, norm in enumerate(diff_norms):
        bar = '█' * int(norm / diff_norms.max() * 30)
        print(f"  L{l:2d}: {norm:6.2f}  {bar}")


def discriminating_axis_projection_nostopwords(act_a, act_b, label_a, label_b, layer):
    act1      = act_a["resid_post", layer][0].detach().numpy()
    act2      = act_b["resid_post", layer][0].detach().numpy()
    mean_diff = act1.mean(0) - act2.mean(0)
    axis      = mean_diff / (np.linalg.norm(mean_diff) + 1e-8)
    proj1, proj2 = act1 @ axis, act2 @ axis
    tokens_a  = get_tokens(label_a)
    tokens_b  = get_tokens(label_b)

    # Filter <|endoftext|> — it's a massive outlier that collapses the x-axis scale
    mask_a = np.array([t != '<|endoftext|>' for t in tokens_a])
    mask_b = np.array([t != '<|endoftext|>' for t in tokens_b])

    proj1_clean    = proj1[mask_a]
    proj2_clean    = proj2[mask_b]
    tokens_a_clean = [t for t, m in zip(tokens_a, mask_a) if m]
    tokens_b_clean = [t for t, m in zip(tokens_b, mask_b) if m]

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.scatter(proj1_clean, np.zeros_like(proj1_clean) + 0.1, alpha=0.7,
               color='steelblue', label=label_a, s=60)
    ax.scatter(proj2_clean, np.zeros_like(proj2_clean) - 0.1, alpha=0.7,
               color='tomato',    label=label_b, s=60)
    for x, tok in zip(proj1_clean, tokens_a_clean):
        ax.annotate(tok, (x, 0.1), fontsize=6, ha='center', va='bottom', color='steelblue')
    for x, tok in zip(proj2_clean, tokens_b_clean):
        ax.annotate(tok, (x, -0.1), fontsize=6, ha='center', va='top', color='tomato')

    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
    ax.set_yticks([])
    ax.set_xlabel(f"← {label_b} side  |  {label_a} side →")
    ax.set_title(f"Layer {layer}: tokens on maximally discriminating axis "
                 f"({label_a} vs {label_b})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"Figures/disc_axis_L{layer}_{label_a.lower()}_{label_b.lower()}.png", dpi=150)
    plt.show()





def characterize_discriminating_tokens_nostopwords(act_a, act_b,
                                        label_a, label_b,
                                        layer, top_n=15):
    act1 = act_a["resid_post", layer][0].detach().numpy()
    act2 = act_b["resid_post", layer][0].detach().numpy()

    mean_diff = act1.mean(0) - act2.mean(0)
    axis      = mean_diff / (np.linalg.norm(mean_diff) + 1e-8)

    proj1 = act1 @ axis
    proj2 = act2 @ axis

    # Build masks BEFORE filtering tokens, apply to projections too
    tokens_a_raw = list(get_tokens(label_a))
    tokens_b_raw = list(get_tokens(label_b))

    mask_a = np.array([t != '<|endoftext|>' for t in tokens_a_raw])
    mask_b = np.array([t != '<|endoftext|>' for t in tokens_b_raw])

    tokens_a = [t for t, m in zip(tokens_a_raw, mask_a) if m]
    tokens_b = [t for t, m in zip(tokens_b_raw, mask_b) if m]
    proj1    = proj1[mask_a]   # ← apply same mask to projections
    proj2    = proj2[mask_b]

    df_a = pd.DataFrame({"token": tokens_a, "projection": proj1,
                         "source": label_a})
    df_b = pd.DataFrame({"token": tokens_b, "projection": proj2,
                         "source": label_b})

    # Crossover tokens — on the "wrong" side
    crossover_a = df_a[df_a["projection"] < 0].sort_values("projection")
    crossover_b = df_b[df_b["projection"] > 0].sort_values("projection",
                                                             ascending=False)

    print(f"\n{label_a} tokens projecting onto {label_b} side:")
    print(crossover_a[["token","projection"]].head(top_n).to_string(index=False))

    print(f"\n{label_b} tokens projecting onto {label_a} side:")
    print(crossover_b[["token","projection"]].head(top_n).to_string(index=False))

    print(f"\nMost distinctively {label_a}:")
    print(df_a.sort_values("projection", ascending=False)
          .head(top_n)[["token","projection"]].to_string(index=False))

    print(f"\nMost distinctively {label_b}:")
    print(df_b.sort_values("projection")
          .head(top_n)[["token","projection"]].to_string(index=False))
    

def discriminating_axis_projection(act_a, act_b, label_a, label_b, layer):
    act1 = act_a["resid_post", layer][0].detach().numpy()
    act2 = act_b["resid_post", layer][0].detach().numpy()
    mean_diff = act1.mean(0) - act2.mean(0)
    axis = mean_diff / (np.linalg.norm(mean_diff) + 1e-8)
    proj1, proj2 = act1 @ axis, act2 @ axis

    tokens_a_raw = list(get_tokens(label_a))
    tokens_b_raw = list(get_tokens(label_b))

    mask_a = content_mask(tokens_a_raw)      # ← shared utility
    mask_b = content_mask(tokens_b_raw)

    proj1_clean    = proj1[mask_a]
    proj2_clean    = proj2[mask_b]
    tokens_a_clean = [t for t, m in zip(tokens_a_raw, mask_a) if m]
    tokens_b_clean = [t for t, m in zip(tokens_b_raw, mask_b) if m]
    
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.scatter(proj1_clean, np.zeros_like(proj1_clean) + 0.1, alpha=0.7,
               color='steelblue', label=label_a, s=60)
    ax.scatter(proj2_clean, np.zeros_like(proj2_clean) - 0.1, alpha=0.7,
               color='tomato',    label=label_b, s=60)
    for x, tok in zip(proj1_clean, tokens_a_clean):
        ax.annotate(tok, (x, 0.1), fontsize=6, ha='center', va='bottom', color='steelblue')
    for x, tok in zip(proj2_clean, tokens_b_clean):
        ax.annotate(tok, (x, -0.1), fontsize=6, ha='center', va='top', color='tomato')

    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
    ax.set_yticks([])
    ax.set_xlabel(f"← {label_b} side  |  {label_a} side →")
    ax.set_title(f"Layer {layer}: tokens on maximally discriminating axis "
                 f"({label_a} vs {label_b})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"Figures/disc_axis_L{layer}_{label_a.lower()}_{label_b.lower()}.png", dpi=150)
    plt.show()


def characterize_discriminating_tokens(act_a, act_b,
                                        label_a, label_b,
                                        layer, top_n=15):
    act1 = act_a["resid_post", layer][0].detach().numpy()
    act2 = act_b["resid_post", layer][0].detach().numpy()
    mean_diff = act1.mean(0) - act2.mean(0)
    axis = mean_diff / (np.linalg.norm(mean_diff) + 1e-8)
    proj1, proj2 = act1 @ axis, act2 @ axis

    tokens_a_raw = list(get_tokens(label_a))
    tokens_b_raw = list(get_tokens(label_b))

    mask_a = content_mask(tokens_a_raw)      # ← same utility
    mask_b = content_mask(tokens_b_raw)

    proj1 = proj1[mask_a]
    proj2 = proj2[mask_b]
    tokens_a = [t for t, m in zip(tokens_a_raw, mask_a) if m]
    tokens_b = [t for t, m in zip(tokens_b_raw, mask_b) if m]
    


    df_a = pd.DataFrame({"token": tokens_a, "projection": proj1,
                         "source": label_a})
    df_b = pd.DataFrame({"token": tokens_b, "projection": proj2,
                         "source": label_b})

    # Crossover tokens — on the "wrong" side
    crossover_a = df_a[df_a["projection"] < 0].sort_values("projection")
    crossover_b = df_b[df_b["projection"] > 0].sort_values("projection",
                                                             ascending=False)

    print(f"\n{label_a} tokens projecting onto {label_b} side:")
    print(crossover_a[["token","projection"]].head(top_n).to_string(index=False))

    print(f"\n{label_b} tokens projecting onto {label_a} side:")
    print(crossover_b[["token","projection"]].head(top_n).to_string(index=False))

    print(f"\nMost distinctively {label_a}:")
    print(df_a.sort_values("projection", ascending=False)
          .head(top_n)[["token","projection"]].to_string(index=False))

    print(f"\nMost distinctively {label_b}:")
    print(df_b.sort_values("projection")
          .head(top_n)[["token","projection"]].to_string(index=False))



# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
bridge = TransformerBridge.boot_transformers("gpt2", device="cpu")

# --- Activations ---
_, human_act  = bridge.run_with_cache(human_prose)
_, gemini_act = bridge.run_with_cache(gemini_prose)
_, claude_act = bridge.run_with_cache(claude_prose)

# --- Tokens ---
human_tokens  = bridge.to_str_tokens(human_prose)
gemini_tokens = bridge.to_str_tokens(gemini_prose)
claude_tokens = bridge.to_str_tokens(claude_prose)


# --- NLL dataframes ---
df_human  = per_token_nll_ppl(bridge, human_prose)
df_gemini = per_token_nll_ppl(bridge, gemini_prose)
df_claude = per_token_nll_ppl(bridge, claude_prose)

# ── NLL summaries ──────────────────────────────────────────
for label, df in [("Human", df_human), ("Gemini", df_gemini), ("Claude", df_claude)]:
    summarize_nll(df, label)

# ── Most surprising tokens ─────────────────────────────────
for label, df in [("Human", df_human), ("Gemini", df_gemini), ("Claude", df_claude)]:
    print(f"\nMost surprising tokens — {label}:")
    print(df.sort_values("nll", ascending=False).head(10).to_string(index=False))

# ── KS test — all three pairs ──────────────────────────────
print(f"\n{'Pair':<22} {'KS stat':>8}  {'p-value':>8}  {'mean NLL diff':>14}")
print("─" * 58)
for label_a, label_b in PAIR_NAMES:
    a, b    = get_df(label_a)["nll"].values, get_df(label_b)["nll"].values
    ks, pv  = ks_2samp(a, b)
    diff    = abs(a.mean() - b.mean())
    pair    = f"{label_a} ↔ {label_b}"
    print(f"{pair:<22} {ks:>8.4f}  {pv:>8.4f}  {diff:>14.4f}")

'''
# ── Surprise concentration ─────────────────────────────────
print("\nSurprise concentration (top 10% of tokens):")
for label, df in [("Human", df_human), ("Gemini", df_gemini), ("Claude", df_claude)]:
    print(f"  {label}: {surprise_concentration(df['nll']):.4f}")
'''    

'''
# ── Pairwise plots ─────────────────────────────────────────
for label_a, label_b in PAIR_NAMES:
    df_a, df_b = get_df(label_a), get_df(label_b)
    plot_nll_pair(df_a, df_b, label_a, label_b)
    plot_nll_delta(df_a, df_b, label_a, label_b)
    plot_surprise_concentration_pair(df_a, df_b, label_a, label_b)
'''

# ── Cosine similarity profiles ─────────────────────────────
all_sims = {}
for label_a, label_b in PAIR_NAMES:
    key          = f"{label_a} ↔ {label_b}"
    all_sims[key] = cosine_sim_profile(
        get_activations(label_a), get_activations(label_b),
        label_a, label_b
    )
plot_cosine_sim_all_pairs(all_sims)



# Layer 9 is where most of the difference lies for passage[0]
for label_a, label_b in PAIR_NAMES:
    discriminating_axis_projection(
        get_activations(label_a), get_activations(label_b),
        label_a, label_b, layer=9
    )


for label_a, label_b in PAIR_NAMES:
    characterize_discriminating_tokens(get_activations(label_a), get_activations(label_b),
        label_a, label_b, layer=9, top_n=15
        )


# ── Activation-space analyses (commented out — enable as needed) ──
'''
for label_a, label_b in PAIR_NAMES:
    act_a, act_b = get_activations(label_a), get_activations(label_b)
    prose_divergence(act_a, act_b, label_a, label_b)
    discriminating_axis_projection(act_a, act_b, label_a, label_b, layer=3)
    for layer in [0, 4, 8, 11]:
        pca_at_layer(act_a, act_b, label_a, label_b, layer)
'''


