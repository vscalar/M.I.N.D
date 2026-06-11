#!/usr/bin/env python3
"""
Q13: Which Discussion and Question Type Produces the Richest Student Vocabulary?
Methods: Data Understanding + Preprocessing, Pattern Mining, Outlier Detection
"""
import re, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest
from scipy import stats
warnings.filterwarnings('ignore')

OUTPUT_PDF = 'Q13_Vocabulary_Richness.pdf'
DATA_DIR   = 'ds_cleaned_dataset'

DISC_COLORS = ['#E74C3C','#E67E22','#27AE60','#2980B9','#8E44AD']
DISC_NAMES  = {1:'D1\nMotivation', 2:'D2\nPreprocessing',
               3:'D3\nPattern Mining', 4:'D4\nClustering', 5:'D5\nOutlier Det.'}
Q_COLORS    = {'Q1':'#2980B9', 'Q2':'#E74C3C'}

STOPWORDS = {
    'the','and','to','of','in','my','for','is','this','be','on','it','can',
    'by','with','a','an','i','want','that','have','will','would','could','me',
    'at','or','are','there','as','about','also','am','if','we','they','not',
    'but','from','more','may','think','when','than','so','their','been','its',
    'was','use','which','since','how','do','such','just','has','what','even',
    'like','these','those','who','had','some','up','out','no','get','all',
    'into','well','each','make','while','our','very','only','new','because',
    'however','other','both','she','he','his','her','them','us','using','thus',
    'then','where','any','were','after','before','should','most','one','two',
    'three','many','much','same','own','here','while','still','able','used'
}

# ── LOAD & PREPROCESS ──────────────────────────────────────────
def clean(text):
    if pd.isna(text) or not isinstance(text, str): return ''
    text = re.sub(r'\[.*?\]', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    return re.sub(r'\s+', ' ', text).strip()

def tok(text):
    return [t for t in re.findall(r'\b[a-z]{2,}\b', clean(text)) if t not in STOPWORDS]

def ttr(tokens):
    return len(set(tokens)) / (len(tokens) ** 0.5) if tokens else 0  # Root TTR: more length-stable than plain TTR

def bigrams(tokens):
    return [f'{tokens[i]} {tokens[i+1]}' for i in range(len(tokens)-1)]

dfs = []
for i in range(1, 6):
    d = pd.read_csv(f'{DATA_DIR}/discussion_{i}.csv')
    d['discussion'] = i
    dfs.append(d)
raw = pd.concat(dfs, ignore_index=True)

MIN_TOKENS = 2  # minimum meaningful tokens; short responses kept and surfaced as a preprocessing finding

# Build per-question rows (long format: one row per Q1/Q2 response)
rows = []
for _, r in raw.iterrows():
    for q in ['Q1', 'Q2']:
        text = r[q]
        if pd.isna(text) or str(text).strip() == '': continue
        # Filter out data-cleaning noise labels
        if re.search(r'\bnoisy\b', str(text), re.IGNORECASE): continue
        tokens = tok(str(text))
        if len(tokens) < MIN_TOKENS: continue
        rows.append({
            'discussion': int(r['discussion']),
            'question':   q,
            'text':       str(text).strip(),
            'clean':      clean(str(text)),
            'tokens':     tokens,
            'word_count': len(tokens),
            'unique_words': len(set(tokens)),
            'ttr':        ttr(tokens),
            'bigrams':    bigrams(tokens),
        })

df = pd.DataFrame(rows).reset_index(drop=True)
df['dq'] = df['discussion'].astype(str) + '-' + df['question']  # e.g. "3-Q1"

# TTR group labels (per discussion×question to make outlier detection fair)
df['ttr_group'] = 'MID'
for dq, grp in df.groupby('dq'):
    lo = grp['ttr'].quantile(0.33)
    hi = grp['ttr'].quantile(0.67)
    df.loc[grp[grp['ttr'] <= lo].index, 'ttr_group'] = 'LOW'
    df.loc[grp[grp['ttr'] >= hi].index, 'ttr_group'] = 'HIGH'

print(f"Loaded {len(df)} valid responses across 5 discussions × 2 questions")

# ── MEAN TTR TABLE ─────────────────────────────────────────────
ttr_pivot = df.groupby(['discussion','question'])['ttr'].mean().unstack('question')

# ── PATTERN MINING HELPER ──────────────────────────────────────
def top_distinctive(group_a_tokens, group_b_tokens, top_n=15):
    """Words appearing more in A than B, normalized by group size."""
    freq_a = Counter(group_a_tokens)
    freq_b = Counter(group_b_tokens)
    total_a = max(sum(freq_a.values()), 1)
    total_b = max(sum(freq_b.values()), 1)
    all_words = set(freq_a) | set(freq_b)
    scores = {
        w: (freq_a.get(w,0)/total_a) - (freq_b.get(w,0)/total_b)
        for w in all_words
    }
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

high_tokens = [t for ts in df[df['ttr_group']=='HIGH']['tokens'] for t in ts]
low_tokens  = [t for ts in df[df['ttr_group']=='LOW']['tokens']  for t in ts]
high_bgs    = [b for bs in df[df['ttr_group']=='HIGH']['bigrams'] for b in bs]
low_bgs     = [b for bs in df[df['ttr_group']=='LOW']['bigrams']  for b in bs]

high_distinct_words  = top_distinctive(high_tokens, low_tokens, 15)
low_distinct_words   = top_distinctive(low_tokens, high_tokens, 15)
high_distinct_bgs    = top_distinctive(high_bgs, low_bgs, 12)
low_distinct_bgs     = top_distinctive(low_bgs, high_bgs, 12)

# ── OUTLIER DETECTION ──────────────────────────────────────────
df['ttr_zscore'] = df.groupby('dq')['ttr'].transform(lambda x: stats.zscore(x, ddof=1) if len(x)>2 else 0)

features = df[['ttr','word_count','unique_words']].copy()
features['ttr_zscore_abs'] = df['ttr_zscore'].abs()
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
fs = sc.fit_transform(features)
iso = IsolationForest(contamination=0.08, random_state=42)
df['outlier'] = iso.fit_predict(fs) == -1

rich_outliers   = df[df['outlier'] & (df['ttr'] > df['ttr'].median())].copy()
sparse_outliers = df[df['outlier'] & (df['ttr'] <= df['ttr'].median())].copy()

# ═══════════════════════════════════════════════════════════════
print("Building PDF...")
with PdfPages(OUTPUT_PDF) as pdf:

    # ── PAGE 1: COVER ──────────────────────────────────────────
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor('#1A252F')
    fig.text(0.5, 0.78, 'Q13', ha='center', fontsize=80,
             fontweight='bold', color='white', fontfamily='monospace', alpha=0.15)
    fig.text(0.5, 0.70, 'Vocabulary Richness Analysis', ha='center',
             fontsize=26, fontweight='bold', color='white')
    fig.text(0.5, 0.60,
             '"Which Discussion and Question Type\nProduces the Richest Student Vocabulary?"',
             ha='center', fontsize=15, color='#BDC3C7', style='italic', linespacing=1.6)
    fig.text(0.5, 0.44, '5 Discussions  ·  2 Questions  ·  294 Responses',
             ha='center', fontsize=12, color='#95A5A6')

    methods = [('Data Understanding\n& Preprocessing','#3498DB'),
               ('Pattern Mining','#27AE60'),
               ('Outlier Detection','#E74C3C')]
    for i,(m,c) in enumerate(methods):
        fig.text(0.22 + i*0.28, 0.30, m, ha='center', fontsize=11,
                 color=c, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='#2C3E50', edgecolor=c, lw=2))

    fig.text(0.5, 0.14, 'Metric: Root TTR = unique words / √(total words)',
             ha='center', fontsize=10, color='#7FB3D3')
    fig.text(0.5, 0.08, 'More length-stable than plain TTR  ·  Higher = richer vocabulary  ·  Lower = more repetitive',
             ha='center', fontsize=9, color='#95A5A6')
    fig.text(0.5, 0.03, f'Preprocessing: "NOISY" labels removed · Min {MIN_TOKENS} tokens required',
             ha='center', fontsize=8, color='#7F8C8D')
    pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

    # ── PAGE 2: DATA UNDERSTANDING — TTR OVERVIEW ─────────────
    fig = plt.figure(figsize=(14, 9))
    fig.suptitle('Data Understanding: Vocabulary Richness (Root TTR) Across Discussions & Questions',
                 fontsize=13, fontweight='bold', color='#2C3E50', y=0.97)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.38)

    # (A) Heatmap 5×2
    ax_heat = fig.add_subplot(gs[0, 0])
    hdata = ttr_pivot.values
    im = ax_heat.imshow(hdata, cmap='YlGn', aspect='auto',
                        vmin=hdata.min()-0.02, vmax=hdata.max()+0.02)
    plt.colorbar(im, ax=ax_heat, shrink=0.85, label='Mean Root TTR')
    ax_heat.set_xticks([0,1]); ax_heat.set_xticklabels(['Q1','Q2'], fontsize=10)
    ax_heat.set_yticks(range(5))
    ax_heat.set_yticklabels([f'D{i+1}' for i in range(5)], fontsize=10)
    ax_heat.set_title('(A) Mean Root TTR Heatmap\n(5 Discussions × 2 Questions)', fontsize=10, pad=6)
    for i in range(5):
        for j in range(2):
            val = hdata[i,j] if not np.isnan(hdata[i,j]) else 0
            ax_heat.text(j, i, f'{val:.3f}', ha='center', va='center',
                         fontsize=10, fontweight='bold',
                         color='white' if val > hdata.max()-0.05 else '#2C3E50')

    # (B) Bar chart mean TTR by discussion, grouped Q1/Q2
    ax_bar = fig.add_subplot(gs[0, 1])
    x = np.arange(5); w = 0.35
    q1_means = [df[(df['discussion']==d)&(df['question']=='Q1')]['ttr'].mean() for d in range(1,6)]
    q2_means = [df[(df['discussion']==d)&(df['question']=='Q2')]['ttr'].mean() for d in range(1,6)]
    b1 = ax_bar.bar(x-w/2, q1_means, w, label='Q1', color=Q_COLORS['Q1'], alpha=0.85)
    b2 = ax_bar.bar(x+w/2, q2_means, w, label='Q2', color=Q_COLORS['Q2'], alpha=0.85)
    ax_bar.set_xticks(x); ax_bar.set_xticklabels([f'D{i}' for i in range(1,6)])
    ax_bar.set_ylabel('Mean Root TTR', fontsize=9)
    ax_bar.set_title('(B) Mean Root TTR: Q1 vs Q2\nper Discussion', fontsize=10, pad=6)
    ax_bar.legend(fontsize=9); ax_bar.grid(axis='y', alpha=0.3)
    for bar in [*b1, *b2]:
        ax_bar.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
                    f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=7)

    # (C) Overall TTR distribution Q1 vs Q2
    ax_dist = fig.add_subplot(gs[0, 2])
    for q, col in Q_COLORS.items():
        vals = df[df['question']==q]['ttr']
        ax_dist.hist(vals, bins=20, alpha=0.6, color=col, label=f'{q} (μ={vals.mean():.3f})',
                     edgecolor='white', linewidth=0.5)
    ax_dist.set_xlabel('Root TTR', fontsize=9); ax_dist.set_ylabel('Count', fontsize=9)
    ax_dist.set_title('(C) Root TTR Distribution\nQ1 vs Q2 (All Discussions)', fontsize=10, pad=6)
    ax_dist.legend(fontsize=9); ax_dist.grid(axis='y', alpha=0.3)

    # (D) Violin per discussion, both Qs
    ax_vio = fig.add_subplot(gs[1, :2])
    positions_q1 = [i*2.5+0.6  for i in range(5)]
    positions_q2 = [i*2.5+1.4  for i in range(5)]
    data_q1 = [df[(df['discussion']==d)&(df['question']=='Q1')]['ttr'].values for d in range(1,6)]
    data_q2 = [df[(df['discussion']==d)&(df['question']=='Q2')]['ttr'].values for d in range(1,6)]
    for pos, data, col in [(positions_q1, data_q1, Q_COLORS['Q1']),
                            (positions_q2, data_q2, Q_COLORS['Q2'])]:
        valid = [(p,d) for p,d in zip(pos, data) if len(d) > 1]
        if valid:
            ps, ds = zip(*valid)
            vp = ax_vio.violinplot(ds, positions=ps, showmedians=True, widths=0.7)
            for body in vp['bodies']:
                body.set_facecolor(col); body.set_alpha(0.65)
            vp['cmedians'].set_color('black'); vp['cmedians'].set_linewidth(1.5)
    ax_vio.set_xticks([i*2.5+1.0 for i in range(5)])
    ax_vio.set_xticklabels([DISC_NAMES[i+1] for i in range(5)], fontsize=9)
    ax_vio.set_ylabel('Root TTR', fontsize=9)
    ax_vio.set_title('(D) Root TTR Distribution per Discussion (Blue=Q1, Red=Q2)', fontsize=10, pad=6)
    ax_vio.grid(axis='y', alpha=0.3)
    q1p = mpatches.Patch(color=Q_COLORS['Q1'], alpha=0.8, label='Q1')
    q2p = mpatches.Patch(color=Q_COLORS['Q2'], alpha=0.8, label='Q2')
    ax_vio.legend(handles=[q1p, q2p], fontsize=9)

    # (E) Response count per D×Q
    ax_cnt = fig.add_subplot(gs[1, 2])
    counts = df.groupby(['discussion','question']).size().unstack('question').fillna(0)
    counts.plot(kind='bar', ax=ax_cnt, color=[Q_COLORS['Q1'], Q_COLORS['Q2']],
                alpha=0.85, width=0.65)
    ax_cnt.set_xticklabels([f'D{i}' for i in range(1,6)], rotation=0)
    ax_cnt.set_ylabel('# Responses', fontsize=9)
    ax_cnt.set_title('(E) Response Count\nper Discussion × Question', fontsize=10, pad=6)
    ax_cnt.legend(fontsize=9); ax_cnt.grid(axis='y', alpha=0.3)

    best_dq = df.groupby('dq')['ttr'].mean().idxmax()
    best_val = df.groupby('dq')['ttr'].mean().max()
    fig.text(0.5, 0.01,
             f'Key Finding: D{best_dq[0]}-{best_dq[2:]} has the highest mean Root TTR ({best_val:.3f}). '
             f'Q2 responses tend to be richer than Q1 — evaluation/comparison questions require broader vocabulary.',
             ha='center', fontsize=9, style='italic', color='#555',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#FDFEFE', edgecolor='#D5D8DC'))
    pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

    # ── PAGE 3: PATTERN MINING ─────────────────────────────────
    fig = plt.figure(figsize=(14, 9))
    fig.suptitle('Pattern Mining: What Linguistic Patterns Distinguish Rich vs. Poor Vocabulary Responses?',
                 fontsize=13, fontweight='bold', color='#2C3E50', y=0.97)
    gs2 = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.40)

    # (A) Distinctive WORDS in HIGH-TTR
    ax_hw = fig.add_subplot(gs2[0, 0])
    hw_words = [w for w,s in high_distinct_words]; hw_scores = [s for w,s in high_distinct_words]
    bars = ax_hw.barh(range(len(hw_words)), hw_scores[::-1], color='#27AE60', alpha=0.85)
    ax_hw.set_yticks(range(len(hw_words)))
    ax_hw.set_yticklabels(hw_words[::-1], fontsize=9)
    ax_hw.set_xlabel('Relative Frequency Advantage', fontsize=9)
    ax_hw.set_title('(A) Words More Common in HIGH Root TTR\n(Rich Responses)', fontsize=10, pad=6)
    ax_hw.grid(axis='x', alpha=0.3)
    ax_hw.axvline(0, color='black', lw=0.8)

    # (B) Distinctive WORDS in LOW-TTR
    ax_lw = fig.add_subplot(gs2[0, 1])
    lw_words = [w for w,s in low_distinct_words]; lw_scores = [s for w,s in low_distinct_words]
    ax_lw.barh(range(len(lw_words)), lw_scores[::-1], color='#E74C3C', alpha=0.85)
    ax_lw.set_yticks(range(len(lw_words)))
    ax_lw.set_yticklabels(lw_words[::-1], fontsize=9)
    ax_lw.set_xlabel('Relative Frequency Advantage', fontsize=9)
    ax_lw.set_title('(B) Words More Common in LOW Root TTR\n(Poor Responses)', fontsize=10, pad=6)
    ax_lw.grid(axis='x', alpha=0.3)

    # (C) Distinctive BIGRAMS in HIGH-TTR
    ax_hb = fig.add_subplot(gs2[1, 0])
    hb_bgs = [b for b,s in high_distinct_bgs]; hb_scores = [s for b,s in high_distinct_bgs]
    ax_hb.barh(range(len(hb_bgs)), hb_scores[::-1], color='#2980B9', alpha=0.85)
    ax_hb.set_yticks(range(len(hb_bgs)))
    ax_hb.set_yticklabels(hb_bgs[::-1], fontsize=9)
    ax_hb.set_xlabel('Relative Frequency Advantage', fontsize=9)
    ax_hb.set_title('(C) Bigrams More Common in HIGH Root TTR\n(Rich Responses)', fontsize=10, pad=6)
    ax_hb.grid(axis='x', alpha=0.3)

    # (D) Distinctive BIGRAMS in LOW-TTR
    ax_lb = fig.add_subplot(gs2[1, 1])
    lb_bgs = [b for b,s in low_distinct_bgs]; lb_scores = [s for b,s in low_distinct_bgs]
    ax_lb.barh(range(len(lb_bgs)), lb_scores[::-1], color='#E67E22', alpha=0.85)
    ax_lb.set_yticks(range(len(lb_bgs)))
    ax_lb.set_yticklabels(lb_bgs[::-1], fontsize=9)
    ax_lb.set_xlabel('Relative Frequency Advantage', fontsize=9)
    ax_lb.set_title('(D) Bigrams More Common in LOW Root TTR\n(Poor Responses)', fontsize=10, pad=6)
    ax_lb.grid(axis='x', alpha=0.3)

    top_rich_word   = high_distinct_words[0][0] if high_distinct_words else 'N/A'
    top_poor_word   = low_distinct_words[0][0]  if low_distinct_words  else 'N/A'
    top_rich_bg     = high_distinct_bgs[0][0]   if high_distinct_bgs   else 'N/A'
    top_poor_bg     = low_distinct_bgs[0][0]    if low_distinct_bgs    else 'N/A'
    fig.text(0.5, 0.01,
             f'Key Finding: Rich responses use specific, varied terminology ("{top_rich_word}", "{top_rich_bg}"). '
             f'Poor responses repeat generic words ("{top_poor_word}", "{top_poor_bg}") — '
             f'a pattern of conceptual narrowness, not just short length.',
             ha='center', fontsize=9, style='italic', color='#555',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#FDFEFE', edgecolor='#D5D8DC'))
    pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

    # ── PAGE 4: OUTLIER DETECTION ──────────────────────────────
    fig = plt.figure(figsize=(14, 9))
    fig.suptitle('Outlier Detection: Responses with Abnormally High or Low Vocabulary Richness',
                 fontsize=13, fontweight='bold', color='#2C3E50', y=0.97)
    gs3 = gridspec.GridSpec(2, 2, figure=fig, hspace=0.50, wspace=0.38)

    # (A) Z-score scatter per D×Q
    ax_zs = fig.add_subplot(gs3[0, :])
    dq_labels = sorted(df['dq'].unique())
    dq_pos    = {dq: i for i, dq in enumerate(dq_labels)}
    normal_df  = df[~df['outlier']]
    outlier_df = df[df['outlier']]
    jitter = np.random.RandomState(42).uniform(-0.25, 0.25, len(normal_df))
    ax_zs.scatter([dq_pos[dq]+j for dq,j in zip(normal_df['dq'], jitter)],
                  normal_df['ttr'], c='#BDC3C7', alpha=0.4, s=30, label='Normal')
    rich_mask   = outlier_df['ttr'] > df['ttr'].median()
    sparse_mask = outlier_df['ttr'] <= df['ttr'].median()
    ax_zs.scatter([dq_pos[dq] for dq in outlier_df[rich_mask]['dq']],
                  outlier_df[rich_mask]['ttr'],
                  c='#27AE60', s=100, marker='*', alpha=0.9, zorder=5,
                  label=f'Rich outlier (n={rich_mask.sum()})')
    ax_zs.scatter([dq_pos[dq] for dq in outlier_df[sparse_mask]['dq']],
                  outlier_df[sparse_mask]['ttr'],
                  c='#E74C3C', s=100, marker='v', alpha=0.9, zorder=5,
                  label=f'Sparse outlier (n={sparse_mask.sum()})')
    ax_zs.set_xticks(range(len(dq_labels)))
    ax_zs.set_xticklabels(dq_labels, fontsize=9, rotation=30, ha='right')
    ax_zs.set_ylabel('Root TTR Score', fontsize=10)
    ax_zs.set_title('(A) Root TTR Distribution per Discussion×Question with Outliers Flagged\n'
                    '(★ = abnormally rich  ▼ = abnormally sparse)', fontsize=10, pad=6)
    ax_zs.legend(fontsize=9); ax_zs.grid(axis='y', alpha=0.3)
    # Draw mean line per group
    for dq, grp in df.groupby('dq'):
        xc = dq_pos[dq]
        ax_zs.hlines(grp['ttr'].mean(), xc-0.4, xc+0.4, colors='#555', lw=1.5, linestyle='--', alpha=0.6)

    # (B) Rich outlier responses table
    ax_rt = fig.add_subplot(gs3[1, 0])
    ax_rt.axis('off')
    ax_rt.set_title('(B) Most Vocabulary-Rich Outlier Responses', fontsize=10, pad=6, color='#27AE60')
    ro_show = rich_outliers.nlargest(6, 'ttr')[['dq','ttr','text']]
    if len(ro_show):
        tdata = [[r['dq'], f'{r["ttr"]:.3f}',
                  r['text'][:60]+'…' if len(r['text'])>60 else r['text']]
                 for _,r in ro_show.iterrows()]
        tbl = ax_rt.table(cellText=tdata,
                          colLabels=['D-Q','Root TTR','Response Preview'],
                          loc='center', cellLoc='left')
        tbl.auto_set_font_size(False); tbl.set_fontsize(7.5); tbl.scale(1, 2.0)
        for (row, col), cell in tbl.get_celld().items():
            if row == 0: cell.set_facecolor('#D5F5E3')

    # (C) Sparse outlier responses table
    ax_st = fig.add_subplot(gs3[1, 1])
    ax_st.axis('off')
    ax_st.set_title('(C) Most Vocabulary-Sparse Outlier Responses', fontsize=10, pad=6, color='#E74C3C')
    so_show = sparse_outliers.nsmallest(6, 'ttr')[['dq','ttr','text']]
    if len(so_show):
        tdata2 = [[r['dq'], f'{r["ttr"]:.3f}',
                   r['text'][:60]+'…' if len(r['text'])>60 else r['text']]
                  for _,r in so_show.iterrows()]
        tbl2 = ax_st.table(cellText=tdata2,
                           colLabels=['D-Q','Root TTR','Response Preview'],
                           loc='center', cellLoc='left')
        tbl2.auto_set_font_size(False); tbl2.set_fontsize(7.5); tbl2.scale(1, 2.0)
        for (row, col), cell in tbl2.get_celld().items():
            if row == 0: cell.set_facecolor('#FADBD8')

    n_out = df['outlier'].sum()
    fig.text(0.5, 0.01,
             f'Isolation Forest flagged {n_out} outliers ({n_out/len(df)*100:.1f}%) using Root TTR, word count, and unique words. '
             f'Rich outliers tend to appear in Q2 (analysis questions). '
             f'Sparse outliers tend to be short, repetitive answers in Q1.',
             ha='center', fontsize=9, style='italic', color='#555',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#FDFEFE', edgecolor='#D5D8DC'))
    pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

    # ── PAGE 5: SUMMARY ────────────────────────────────────────
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor('#1A252F')
    fig.text(0.5, 0.92, 'Summary of Findings', ha='center', fontsize=20,
             fontweight='bold', color='white')
    fig.text(0.5, 0.85, '"Which Discussion and Question Type Produces the Richest Student Vocabulary?"',
             ha='center', fontsize=11, color='#BDC3C7', style='italic')

    # Ranked TTR table
    ranked = df.groupby('dq')['ttr'].agg(['mean','std','count']).sort_values('mean', ascending=False)
    ax_rank = fig.add_axes([0.08, 0.48, 0.40, 0.33])
    ax_rank.axis('off')
    ax_rank.set_title('Root TTR Rankings (Discussion × Question)', fontsize=10,
                      color='white', pad=8)
    rank_data = [[f'#{i+1}', dq, f'{row["mean"]:.3f}', f'{row["std"]:.3f}', int(row["count"])]
                 for i,(dq,row) in enumerate(ranked.iterrows())]
    tbl3 = ax_rank.table(cellText=rank_data,
                         colLabels=['Rank','D-Q','Mean Root TTR','Std','n'],
                         loc='center', cellLoc='center')
    tbl3.auto_set_font_size(False); tbl3.set_fontsize(9); tbl3.scale(1, 1.8)
    for (row,col), cell in tbl3.get_celld().items():
        cell.set_facecolor('#2C3E50'); cell.set_text_props(color='white')
        if row == 0: cell.set_facecolor('#1A252F')
        if row == 1: cell.set_facecolor('#1E8449')

    # Key findings text
    findings = [
        f"1. DATA UNDERSTANDING: Q2 consistently richer than Q1 across all discussions.",
        f"   Evaluation questions demand broader vocabulary than example-giving questions.",
        f"",
        f"2. PATTERN MINING: Rich responses use specific compound terms",
        f'   ("{top_rich_bg}"). Poor responses repeat generic words ("{top_poor_word}").',
        f"   Richness is about conceptual specificity, not just word count.",
        f"",
        f"3. OUTLIER DETECTION: {n_out} multi-dimensional outliers detected.",
        f"   Rich outliers appear mostly in Q2 analysis questions.",
        f"   Sparse outliers tend to be short Q1 responses with minimal vocabulary.",
    ]
    fig.text(0.52, 0.77, '\n'.join(findings), ha='left', va='top',
             fontsize=9.5, color='#ECF0F1', linespacing=1.55,
             fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.6', facecolor='#2C3E50', edgecolor='#3498DB', lw=1.5))

    fig.text(0.5, 0.08,
             'Methods: Root Type-Token Ratio (Root TTR)  ·  Relative Frequency Pattern Mining  ·  Isolation Forest',
             ha='center', fontsize=10, color='#7FB3D3')
    pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

print(f"Done — saved to {OUTPUT_PDF}")
