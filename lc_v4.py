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
from scipy.stats import ks_2samp
from scipy.stats import gaussian_kde
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib import cm
from scipy.stats import shapiro, normaltest



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
# Discriminating-axis helper
# ─────────────────────────────────────────────────────────────
def discriminating_axis(label_a, label_b, layer=9):
    """Unit vector along (mean_A - mean_B) in the layer-`layer` residual stream."""
    act1 = get_activations(label_a)["resid_post", layer][0].detach().numpy()
    act2 = get_activations(label_b)["resid_post", layer][0].detach().numpy()
    mean_diff = act1.mean(0) - act2.mean(0)
    axis = mean_diff / (np.linalg.norm(mean_diff) + 1e-8)
    return act1, act2, axis


def get_projections(label_a, label_b, layer=9, center=True, return_tokens=False):
    """Projections of both translations' content tokens onto the discriminating axis.

    If center=True, both sets are shifted so that the midpoint between the two
    centroids sits at 0.  Centering is a pure translation: KS, overlap coefficient
    and Cohen's d are unchanged, but sign now means "which side of the midpoint",
    which makes the crossover logic (proj < 0 / proj > 0) literally correct.
    """
    act1, act2, axis = discriminating_axis(label_a, label_b, layer)

    tokens_a_raw = list(get_tokens(label_a))
    tokens_b_raw = list(get_tokens(label_b))
    mask_a = content_mask(tokens_a_raw)
    mask_b = content_mask(tokens_b_raw)

    proj1 = (act1 @ axis)[mask_a]
    proj2 = (act2 @ axis)[mask_b]

    if center:
        threshold = (proj1.mean() + proj2.mean()) / 2
        proj1 = proj1 - threshold
        proj2 = proj2 - threshold

    if return_tokens:
        tokens_a = [t for t, m in zip(tokens_a_raw, mask_a) if m]
        tokens_b = [t for t, m in zip(tokens_b_raw, mask_b) if m]
        return proj1, proj2, tokens_a, tokens_b

    return proj1, proj2


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



def plot_nll_stats(df_human, df_gemini, df_claude):

    translations = ['Human\n(Grossman)', 'Gemini', 'Claude']

    whole_ppl_human = torch.exp(torch.tensor(df_human["nll"].mean())).item()
    whole_ppl_gemini = torch.exp(torch.tensor(df_gemini["nll"].mean())).item()
    whole_ppl_claude = torch.exp(torch.tensor(df_claude["nll"].mean())).item()

    metrics = {
        'Whole-text PPL\n(exp of mean NLL)': [whole_ppl_human, whole_ppl_gemini, whole_ppl_claude],
        'Mean NLL':                           [df_human['nll'].mean(),  df_gemini['nll'].mean(),  df_claude['nll'].mean()],
        'NLL Std\n(surprise spread)':         [df_human['nll'].std(),  df_gemini['nll'].std(),  df_claude['nll'].std()],
    }

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    colors = ['steelblue', 'tomato', 'green']

    for ax, (metric, values) in zip(axes, metrics.items()):
        bars = ax.bar(translations, values, color=colors, alpha=0.85, width=0.5)
        ax.set_title(metric, fontsize=10)
        ax.set_ylabel("")
        # Annotate values on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01 * bar.get_height(),
                    f'{val:.2f}', ha='center', va='bottom', fontsize=9)
        ax.set_ylim(0, max(values) * 1.2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.suptitle("Token-level surprise comparison across translations\n"
                "(GPT-2 Small as analysis model)",
                fontsize=11, y=1.02)
    plt.tight_layout()
    plt.savefig("results/nll_summary_comparison.png", dpi=150, bbox_inches='tight')
    plt.show()



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
    # here, mean-pooled is over tokens, then cosine similarity between the two mean vectors
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




def discriminating_axis_projection(label_a, label_b, layer=9):
    """Strip plot of content tokens along the (centered) discriminating axis."""
    proj1, proj2, tokens_a, tokens_b = get_projections(
        label_a, label_b, layer, return_tokens=True)

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.scatter(proj1, np.zeros_like(proj1) + 0.1, alpha=0.7,
               color='steelblue', label=label_a, s=60)
    ax.scatter(proj2, np.zeros_like(proj2) - 0.1, alpha=0.7,
               color='tomato',    label=label_b, s=60)
    for x, tok in zip(proj1, tokens_a):
        ax.annotate(tok, (x, 0.1), fontsize=6, ha='center', va='bottom', color='steelblue')
    for x, tok in zip(proj2, tokens_b):
        ax.annotate(tok, (x, -0.1), fontsize=6, ha='center', va='top', color='tomato')

    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
    ax.axvline(0, color='gray', linewidth=1.0, linestyle='--', alpha=0.6)
    ax.set_yticks([])
    ax.set_xlabel(f"← {label_b} side  |  0 = midpoint |  {label_a} side →")
    ax.set_title(f"Layer {layer}: tokens on maximally discriminating axis "
                 f"({label_a} vs {label_b})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"Figures/disc_axis_L{layer}_{label_a.lower()}_{label_b.lower()}.png", dpi=150)
    plt.show()


def colored_passage_html(act_a, act_b, label_a, label_b, layer, prose_label):

    act1 = act_a["resid_post", layer][0].detach().numpy()
    act2 = act_b["resid_post", layer][0].detach().numpy()
    mean_diff = act1.mean(0) - act2.mean(0)
    axis      = mean_diff / (np.linalg.norm(mean_diff) + 1e-8)

    if prose_label == label_a:
        acts   = act1
        tokens = list(get_tokens(label_a))
    else:
        acts   = act2
        tokens = list(get_tokens(label_b))

    projs = acts @ axis

    # Content mask — same filter as the tables
    mask = content_mask(tokens)

    # Normalize using ONLY content token projections
    content_projs = projs[mask]
    p_min, p_max  = content_projs.min(), content_projs.max()

    def projection_to_color(p):
        t = (p - p_min) / (p_max - p_min + 1e-8)
        if t < 0.5:
            s = t * 2
            r = 255
            g = int(100 + 155 * s)
            b = int(100 + 155 * s)
        else:
            s = (t - 0.5) * 2
            r = int(255 - 155 * s)
            g = 255
            b = int(255 - 155 * s)
        return f"rgb({r},{g},{b})"

    lines = []
    lines.append("<!DOCTYPE html>")
    lines.append("<html><head><meta charset='utf-8'>")
    lines.append("<style>")
    lines.append("  body { font-family: Georgia, serif; font-size: 16px; "
                 "line-height: 2.2; max-width: 900px; margin: 40px auto; "
                 "padding: 0 20px; }")
    lines.append("  span { border-radius: 3px; padding: 1px 2px; }")
    lines.append("  h2   { font-size: 14px; color: #555; }")
    lines.append("  .legend { font-size: 12px; margin-bottom: 20px; }")
    lines.append("</style></head><body>")
    lines.append(f"<h1>{prose_label} Translation</h1>")
    lines.append(f"<h2>Content words colored by projection onto discriminating axis "
                 f"({label_a} vs {label_b}, Layer {layer}). "
                 f"Stopwords and subword fragments shown in gray.</h2>")
    lines.append("<p class='legend'>"
                 "<span style='background:rgb(100,200,100);padding:2px 8px'>"
                 f"Green = distinctively {label_a}</span>&nbsp;&nbsp;"
                 "<span style='background:rgb(255,100,100);padding:2px 8px'>"
                 f"Red = distinctively {label_b}</span>&nbsp;&nbsp;"
                 "<span style='background:rgb(200,200,200);padding:2px 8px'>"
                 "Gray = stopword / subword (not included in analysis)</span>"
                 "</p>")
    lines.append("<p>")

    for tok, proj, is_content in zip(tokens, projs, mask):
        if tok in ('<|endoftext|>', '<|BOS|>', '<|bos|>'):
            continue

        display = tok if not tok.startswith(' ') else ' ' + tok.strip()

        if is_content:
            color = projection_to_color(proj)
        else:
            color = "rgb(220,220,220)"   # neutral gray for stopwords/subwords

        lines.append(
            f"<span style='background-color:{color}'>{display}</span>"
        )

    lines.append("</p></body></html>")


    fname = (f"results/colored_passage_{label_a.lower()}_"
         f"{label_b.lower()}_L{layer}.html")
    with open(fname, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved: {fname}")




def characterize_discriminating_tokens(label_a, label_b, layer=9, top_n=15):
    """Most distinctive tokens per translation, and crossover tokens.

    Projections are centered so 0 is the midpoint between the two centroids —
    so `projection < 0` genuinely means "on B's side" and vice versa.
    """
    proj1, proj2, tokens_a, tokens_b = get_projections(
        label_a, label_b, layer, return_tokens=True)

    df_a = pd.DataFrame({"token": tokens_a, "projection": proj1,
                         "source": label_a})
    df_b = pd.DataFrame({"token": tokens_b, "projection": proj2,
                         "source": label_b})

    # Crossover tokens — on the "wrong" side of the midpoint (now literally 0)
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


def discriminating_axis_overlap(label_a, label_b, layer=9):
    """Crossover fractions and centroid separation (projections centered at 0)."""
    proj1, proj2 = get_projections(label_a, label_b, layer)

    # Fraction of each translation's tokens on the wrong side of the midpoint
    crossover_a = (proj1 < 0).sum() / len(proj1)
    crossover_b = (proj2 > 0).sum() / len(proj2)

    # How far apart are the centroids?
    separation = proj1.mean() - proj2.mean()

    print(f"\n{label_a} ↔ {label_b} at layer {layer}:")
    print(f"  {label_a} tokens on {label_b} side: {crossover_a:.1%}")
    print(f"  {label_b} tokens on {label_a} side: {crossover_b:.1%}")
    print(f"  Centroid separation:               {separation:.3f}")

    return crossover_a, crossover_b, separation


def discriminating_axis_ks(label_a, label_b, layer=9):
    proj1, proj2 = get_projections(label_a, label_b, layer)
    ks, pv = ks_2samp(proj1, proj2)
    return ks, pv


def cohens_d(proj1, proj2):
    pooled_std = np.sqrt((proj1.std()**2 + proj2.std()**2) / 2)
    return (proj1.mean() - proj2.mean()) / pooled_std


#overlap coefficient for the KDEs of the projections onto the discriminating axis
def overlap_coefficient(proj1, proj2, n_points=1000):
    """
    Computes the overlap coefficient between two projection distributions.
    OVL = 0: no overlap, OVL = 1: identical distributions.
    """
    x = np.linspace(
        min(proj1.min(), proj2.min()) - 2,
        max(proj1.max(), proj2.max()) + 2,
        n_points
    )
    kde1 = gaussian_kde(proj1)(x)
    kde2 = gaussian_kde(proj2)(x)

    # Normalize so each integrates to 1
    kde1 /= np.trapezoid(kde1, x)
    kde2 /= np.trapezoid(kde2, x)

    overlap = np.trapezoid(np.minimum(kde1, kde2), x)
    return overlap


def plot_all_projection_distributions(layer=9, compute_overlap=True):

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True, sharey=False)
    colors_a = ['steelblue', 'steelblue', 'tomato']
    colors_b = ['tomato',    'green',     'green' ]

    for ax, (label_a, label_b), ca, cb in zip(
            axes, PAIR_NAMES, colors_a, colors_b):

        proj1, proj2 = get_projections(label_a, label_b, layer)

        x = np.linspace(
            min(proj1.min(), proj2.min()) - 2,
            max(proj1.max(), proj2.max()) + 2, 300
        )

        kde1 = gaussian_kde(proj1)
        kde2 = gaussian_kde(proj2)

        kde1 = kde1(x)
        kde2 = kde2(x)


        ax.fill_between(x, kde1, alpha=0.35, color=ca, label=label_a)
        ax.fill_between(x, kde2, alpha=0.35, color=cb, label=label_b)
        ax.plot(x, kde1, color=ca, linewidth=1.5)
        ax.plot(x, kde2, color=cb, linewidth=1.5)

        # midpoint between centroids is now exactly 0
        ax.axvline(0, color='gray', linestyle='--', linewidth=1)

        if compute_overlap:
            ovl = overlap_coefficient(proj1, proj2)

            ax.text(0.97, 0.95, f"overlap = {ovl:.3f}",
                transform=ax.transAxes, ha='right', va='top', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='white',
                          edgecolor='gray', alpha=0.8))

        ax.set_title(f"{label_a} vs {label_b}")
        ax.set_xlabel("Centered projection onto discriminating axis")
        ax.legend(fontsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.suptitle("Token projection distributions at Layer 9 "
                 "(0 = midpoint between centroids)",
                 fontsize=11, y=1.02)
    plt.tight_layout()
    plt.savefig("results/proj_dist_all_pairs.png", dpi=150,
                bbox_inches='tight')
    plt.show()





#visualization of KS statistic on the CDFs of the projections onto the discriminating axis
def plot_cdf_comparison(label_a, label_b, layer=9):
    fig, ax = plt.subplots(figsize=(8, 5))
    proj1, proj2 = get_projections(label_a, label_b, layer)

    for proj, label, color in [(proj1, label_a, 'steelblue'),
                                (proj2, label_b, 'tomato')]:
        sorted_proj = np.sort(proj)
        cdf = np.arange(1, len(proj) + 1) / len(proj)
        ax.plot(sorted_proj, cdf, color=color, label=label, linewidth=1.5)

    # Find and annotate the maximum distance
    all_x  = np.sort(np.concatenate([proj1, proj2]))
    cdf1   = np.searchsorted(np.sort(proj1), all_x) / len(proj1)
    cdf2   = np.searchsorted(np.sort(proj2), all_x) / len(proj2)
    diffs  = np.abs(cdf1 - cdf2)
    max_idx = np.argmax(diffs)

    #for sanity, compare with scipy's ks_2samp result
    ks_stat, ks_pv = ks_2samp(proj1, proj2)
    print(f"\nKS statistic (scipy): {ks_stat:.3f}, p-value: {ks_pv:.3e}")
    print(f"Max CDF distance: {diffs[max_idx]:.3f} at x = {all_x[max_idx]:.3f}")

    ax.axvline(0, color='gray', linestyle='--', linewidth=1, alpha=0.6)
    ax.annotate('',
                xy=(all_x[max_idx], cdf2[max_idx]),
                xytext=(all_x[max_idx], cdf1[max_idx]),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
    ax.text(all_x[max_idx] + 0.5,
            (cdf1[max_idx] + cdf2[max_idx]) / 2,
            f'D = {diffs[max_idx]:.3f}', fontsize=8)

    ax.set_xlabel("Centered projection onto discriminating axis")
    ax.set_ylabel("Cumulative probability")
    ax.set_title(f"Empirical CDF comparison — {label_a} vs {label_b} (Layer {layer})")
    ax.legend(fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"results/cdf_{label_a.lower()}_{label_b.lower()}.png", dpi=150)
    plt.show()


def plot_projection_histogram(label_a, label_b, layer=9, bins=30):
    fig, ax = plt.subplots(figsize=(9, 4))

    proj1, proj2 = get_projections(label_a, label_b, layer)

    ax.hist(proj1, bins=bins, alpha=0.5, color='steelblue',
            label=label_a, density=True)
    ax.hist(proj2, bins=bins, alpha=0.5, color='tomato',
            label=label_b, density=True)

    # Overlay fitted Gaussians for visual comparison
    from scipy.stats import norm
    x = np.linspace(min(proj1.min(), proj2.min()) - 2,
                    max(proj1.max(), proj2.max()) + 2, 300)
    ax.plot(x, norm.pdf(x, proj1.mean(), proj1.std()),
            color='steelblue', linewidth=2, linestyle='--',
            label=f'{label_a} fitted Gaussian')
    ax.plot(x, norm.pdf(x, proj2.mean(), proj2.std()),
            color='tomato', linewidth=2, linestyle='--',
            label=f'{label_b} fitted Gaussian')
    ax.axvline(0, color='gray', linestyle='--', linewidth=1, alpha=0.6)

    ax.set_xlabel("Centered projection onto discriminating axis")
    ax.set_ylabel("Density")
    ax.set_title(f"Token projection distributions — {label_a} vs {label_b} "
                 f"(Layer {layer})")
    ax.legend(fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(
        f"results/hist_{label_a.lower()}_{label_b.lower()}.png", dpi=150)
    plt.show()


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


#Analysis, part 1: cosine similarity of mean-pooled activations at each layer
# ── Cosine similarity profiles ─────────────────────────────
all_sims = {}
for label_a, label_b in PAIR_NAMES:
    key          = f"{label_a} ↔ {label_b}"
    all_sims[key] = cosine_sim_profile(
        get_activations(label_a), get_activations(label_b),
        label_a, label_b
    )
plot_cosine_sim_all_pairs(all_sims)


#Analysis, part 3:
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

# ── KS test on NLL distributions — all three pairs ─────────
# (moved below the NLL dataframes, which it depends on)
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




# Layer 9 is where most of the difference lies for passage[0]
for label_a, label_b in PAIR_NAMES:
    discriminating_axis_projection(label_a, label_b, layer=9)


for label_a, label_b in PAIR_NAMES:
    characterize_discriminating_tokens(label_a, label_b, layer=9, top_n=15)


print(f"\n{'Pair':<22} {'A→B crossover':>14} {'B→A crossover':>14} "
      f"{'Separation':>12} {'KS stat':>9} {'Cohen d':>9}")
print("─" * 85)

cohend = []
for label_a, label_b in PAIR_NAMES:
    proj1, proj2 = get_projections(label_a, label_b, layer=9)

    # projections are centered, so the midpoint is exactly 0
    co_a = (proj1 < 0).sum() / len(proj1)
    co_b = (proj2 > 0).sum() / len(proj2)
    sep  = proj1.mean() - proj2.mean()

    ks, pv = ks_2samp(proj1, proj2)
    cd   = cohens_d(proj1, proj2)
    pair = f"{label_a} ↔ {label_b}"

    print(f"{pair:<22} {co_a:>13.1%} {co_b:>14.1%} "
          f"{sep:>12.3f} {ks:>9.4f} {cd:>9.3f}")
    cohend.append(cd)



pairs  = ["Human ↔ Gemini", "Human ↔ Claude", "Gemini ↔ Claude"]
colors = ['steelblue', 'tomato', 'green']

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(pairs, cohend, color=colors, alpha=0.85, width=0.5)
ax.axhline(0.8, linestyle='--', color='gray', alpha=0.6, label="Large effect threshold (d=0.8)")
for bar, val in zip(bars, cohend):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f'{val:.3f}', ha='center', va='bottom', fontsize=10)
ax.set_ylabel("Cohen's d")
ax.set_title("Separation of token projections at Layer 9\n(discriminating axis)")
ax.set_ylim(0, 1.6)
ax.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig("results/cohens_d.png", dpi=150)
plt.show()

#show the intersection of the distributions on the discriminating axis at layer 9 and display overlap if desired
plot_all_projection_distributions(layer = 9, compute_overlap=True)

#now show the most extreme tokens on the discriminating axis at layer 9 for each pair 


# Run for Human translation against both LLM pairs
colored_passage_html(human_act, gemini_act, "Human", "Gemini", 9, "Human")
colored_passage_html(human_act, claude_act, "Human", "Claude", 9, "Human")
colored_passage_html(gemini_act, claude_act, "Gemini", "Claude", 9, "Gemini")



#for KS test visualization, plot the empirical CDFs of the projections onto the discriminating axis at layer 9 for each pair
for label_a, label_b in PAIR_NAMES:
    plot_cdf_comparison(label_a, label_b)


for label_a, label_b in PAIR_NAMES:
    plot_projection_histogram(label_a, label_b)


#checks for normality of the projections onto the discriminating axis at layer 9 for each pair using D'Agostino-Pearson test
#(centering does not affect the test statistic — it is a pure shift)
for label_a, label_b in PAIR_NAMES:
    proj1, proj2 = get_projections(label_a, label_b, layer=9)

    print(f"\n{label_a} ↔ {label_b}")
    for proj, label in [(proj1, label_a), (proj2, label_b)]:
        stat, p = normaltest(proj)   # D'Agostino-Pearson test
        print(f"  {label}: stat={stat:.3f}, p={p:.4f} "
              f"{'→ Gaussian' if p > 0.05 else '→ not Gaussian'}")




# ── Activation-space analyses (commenting out for now) ──
'''
for label_a, label_b in PAIR_NAMES:
    act_a, act_b = get_activations(label_a), get_activations(label_b)
    prose_divergence(act_a, act_b, label_a, label_b)
    discriminating_axis_projection(label_a, label_b, layer=3)
    for layer in [0, 4, 8, 11]:
        pca_at_layer(act_a, act_b, label_a, label_b, layer)
'''
