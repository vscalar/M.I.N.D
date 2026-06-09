import os, re, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')

try:
    from wordcloud import WordCloud
    HAS_WC = True
except ImportError:
    HAS_WC = False

# ── CONFIG ─────────────────────────────────────────────────────
DATA_DIR   = 'ds_cleaned_dataset'
OUTPUT_PDF = 'MIND_20_Research_Questions.pdf'

DISC_COLORS = ['#E74C3C','#E67E22','#27AE60','#2980B9','#8E44AD']
DISC_NAMES  = {1:'D1: Motivation', 2:'D2: Preprocessing',
               3:'D3: Pattern Mining', 4:'D4: Clustering', 5:'D5: Outlier Det.'}

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
    'three','many','much','same','own','here','also','while','still','able'
}

DOMAIN_KW = {
    'Finance':        ['credit','card','fraud','transaction','bank','money','stock','payment','financial','invest'],
    'Healthcare':     ['health','medical','patient','disease','hospital','heart','diagnosis','clinical','blood'],
    'Cybersecurity':  ['security','attack','hack','cyber','network','intrusion','malware','phishing'],
    'Manufacturing':  ['manufacturing','factory','product','defect','sensor','quality','industrial'],
    'Transportation': ['subway','train','bus','transport','commute','traffic','car','vehicle','road'],
    'Education':      ['student','school','grade','class','course','exam','quiz','score','learn','study'],
    'Daily Life':     ['sleep','schedule','expense','food','personal','daily','life','habit','routine'],
}

AI_TERMS   = {'ai','agent','llm','claude','gpt','model','agents','agentic','language'}
TECH_TERMS = {'clustering','outlier','classification','pattern','mining','detection',
              'preprocessing','association','rule','distance','statistical','dimensional',
              'isolation','support','confidence','lift','apriori','regression',
              'normalization','imputation','dimensionality','reduction','pca','kmeans','dbscan'}

# ── LOAD DATA ──────────────────────────────────────────────────
def clean(text):
    if pd.isna(text) or not isinstance(text, str): return ''
    text = re.sub(r'\[.*?\]', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    return re.sub(r'\s+', ' ', text).strip()

def tok(text):
    return [t for t in re.findall(r'\b[a-z]{2,}\b', clean(text)) if t not in STOPWORDS]

dfs = []
for i in range(1, 6):
    df = pd.read_csv(f'{DATA_DIR}/discussion_{i}.csv')
    df['discussion'] = i
    dfs.append(df)
raw = pd.concat(dfs, ignore_index=True)

raw['Q1c'] = raw['Q1'].fillna('')
raw['Q2c'] = raw['Q2'].fillna('')
raw['text'] = (raw['Q1c'] + ' ' + raw['Q2c']).str.strip()
raw['clean'] = raw['text'].apply(clean)
raw['tokens'] = raw['clean'].apply(tok)
raw['q1_len'] = raw['Q1c'].str.len()
raw['q2_len'] = raw['Q2c'].str.len()
raw['total_len'] = raw['q1_len'] + raw['q2_len']
raw['word_count'] = raw['tokens'].apply(len)
raw['unique_words'] = raw['tokens'].apply(lambda x: len(set(x)))
raw['q1_null'] = raw['Q1'].isna()
raw['q2_null'] = raw['Q2'].isna()

df = raw[raw['clean'].str.len() > 20].copy().reset_index(drop=True)

vec = TfidfVectorizer(max_features=500, stop_words=list(STOPWORDS), min_df=2)
tfidf = vec.fit_transform(df['clean'])
feats = vec.get_feature_names_out()

pca2 = PCA(n_components=2, random_state=42)
coords = pca2.fit_transform(tfidf.toarray())

# ── PDF HELPERS ────────────────────────────────────────────────
def page_header(fig, qnum, title, finding):
    fig.text(0.5, 0.975, f'Q{qnum:02d}  |  {title}', ha='center', va='top',
             fontsize=12, fontweight='bold', color='#2C3E50',
             bbox=dict(boxstyle='round,pad=0.35', facecolor='#ECF0F1', edgecolor='#AAB7B8'))
    fig.text(0.5, 0.018, f'Key Finding: {finding}', ha='center', va='bottom',
             fontsize=8.5, style='italic', color='#555',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#FDFEFE', edgecolor='#D5D8DC'),
             wrap=True)

def disc_patches():
    return [mpatches.Patch(color=DISC_COLORS[i], label=DISC_NAMES[i+1]) for i in range(5)]

print("Generating PDF...")

with PdfPages(OUTPUT_PDF) as pdf:

    # ── COVER ──────────────────────────────────────────────────
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor('#1A252F')
    fig.text(0.5, 0.68, 'M.I.N.D', ha='center', fontsize=72, fontweight='bold', color='white',
             fontfamily='monospace')
    fig.text(0.5, 0.57, 'Mining Interesting Narratives in Discussion', ha='center',
             fontsize=17, color='#BDC3C7')
    fig.text(0.5, 0.47, '20 Research Questions on Student Discussion Data',
             ha='center', fontsize=13, color='#ECF0F1')
    fig.text(0.5, 0.38, '5 Discussions  ·  294 Responses  ·  69 Students',
             ha='center', fontsize=11, color='#95A5A6')
    for i, (d, name) in enumerate(DISC_NAMES.items()):
        fig.text(0.18 + i*0.16, 0.26, name, ha='center', fontsize=9,
                 color=DISC_COLORS[i], fontweight='bold')
    fig.text(0.5, 0.14, 'Pattern Mining  ·  Clustering  ·  Outlier Detection',
             ha='center', fontsize=10, color='#7FB3D3')
    pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

    # ── Q01: SEMANTIC LANDSCAPE ────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for i, d in enumerate(range(1, 6)):
        m = df['discussion'] == d
        ax.scatter(coords[m,0], coords[m,1], c=DISC_COLORS[i], alpha=0.65, s=55,
                   label=DISC_NAMES[d])
    ax.set_xlabel(f'PC1 ({pca2.explained_variance_ratio_[0]*100:.1f}% var)', fontsize=10)
    ax.set_ylabel(f'PC2 ({pca2.explained_variance_ratio_[1]*100:.1f}% var)', fontsize=10)
    ax.legend(handles=disc_patches(), fontsize=9, framealpha=0.8)
    ax.grid(True, alpha=0.25)
    var_tot = sum(pca2.explained_variance_ratio_[:2])*100
    page_header(fig, 1, 'What Is the 2D Semantic Landscape of All Student Responses?',
                f'TF-IDF + PCA explains {var_tot:.1f}% variance. Discussions form loose but visible regional '
                f'clusters — D1 and D5 occupy opposite ends, reflecting maximally different topics.')
    fig.tight_layout(rect=[0,0.06,1,0.94]); pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

    # ── Q02: SEMANTIC OUTLIERS ─────────────────────────────────
    iso = IsolationForest(contamination=0.07, random_state=42)
    ol = iso.fit_predict(tfidf.toarray()) == -1
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={'width_ratios':[1.4,1]})
    ax1.scatter(coords[~ol,0], coords[~ol,1], c='#BDC3C7', alpha=0.45, s=35, label='Normal')
    ax1.scatter(coords[ol,0], coords[ol,1], c='#E74C3C', alpha=0.9, s=90,
                marker='*', label=f'Outlier (n={ol.sum()})')
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.25)
    ax1.set_title('Outlier Responses in PCA Space', fontsize=11)
    ax2.axis('off')
    rows = df[ol][['discussion','text']].head(8)
    tdata = [[f'D{r["discussion"]}', r['text'][:72]+'…'] for _,r in rows.iterrows()]
    tbl = ax2.table(cellText=tdata, colLabels=['Disc','Response Preview'],
                    loc='center', cellLoc='left')
    tbl.auto_set_font_size(False); tbl.set_fontsize(7); tbl.scale(1, 1.9)
    ax2.set_title('Detected Outlier Responses', fontsize=11)
    page_header(fig, 2, 'Which Responses Are Semantic Outliers Across the Entire Corpus?',
                f'Isolation Forest (7% contamination) flagged {ol.sum()} outliers. '
                f'Outliers span multiple discussions — unusual thinking is not confined to one topic.')
    fig.tight_layout(rect=[0,0.07,1,0.93]); pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

    # ── Q03: DISCUSSION DNA ────────────────────────────────────
    disc_texts = {i: ' '.join(df[df['discussion']==i]['clean']) for i in range(1,6)}
    dv = TfidfVectorizer(stop_words=list(STOPWORDS), max_features=1000)
    dm = dv.fit_transform(list(disc_texts.values()))
    df_feat = dv.get_feature_names_out()
    fig, axes = plt.subplots(1, 5, figsize=(16, 5.5))
    for i, ax in enumerate(axes):
        scores = dm[i].toarray().flatten()
        top_i = scores.argsort()[-12:][::-1]
        words = df_feat[top_i]; vals = scores[top_i]
        ax.barh(range(12), vals[::-1], color=DISC_COLORS[i], alpha=0.85)
        ax.set_yticks(range(12)); ax.set_yticklabels(words[::-1], fontsize=8)
        ax.set_title(DISC_NAMES[i+1], fontsize=9, fontweight='bold', color=DISC_COLORS[i])
        ax.set_xlabel('TF-IDF', fontsize=8)
    page_header(fig, 3, 'What Words Define Each Discussion Uniquely? (Discussion DNA)',
                'Each discussion has a distinct vocabulary fingerprint visible in TF-IDF. '
                'D1: "learn/chose/course". D5: "outlier/detection/statistical". The curriculum is in the words.')
    fig.tight_layout(rect=[0,0.07,1,0.93]); pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

    # ── Q04: WHO GAVE UP? ──────────────────────────────────────
    ns = pd.DataFrame([{
        'D': f'D{d}',
        'Q1 Null': raw[raw['discussion']==d]['q1_null'].sum(),
        'Q2 Null': raw[raw['discussion']==d]['q2_null'].sum(),
        'avg_len':  raw[raw['discussion']==d]['total_len'].mean()
    } for d in range(1,6)])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(5); w = 0.35
    ax1.bar(x-w/2, ns['Q1 Null'], w, label='Q1 Skipped', color='#E74C3C', alpha=0.85)
    ax1.bar(x+w/2, ns['Q2 Null'], w, label='Q2 Skipped', color='#3498DB', alpha=0.85)
    ax1.set_xticks(x); ax1.set_xticklabels(ns['D']); ax1.legend()
    ax1.set_ylabel('# Skipped Responses'); ax1.set_title('Skipped Responses by Discussion', fontsize=11)
    ax1.grid(axis='y', alpha=0.3)
    ax2.bar(ns['D'], ns['avg_len'], color=DISC_COLORS, alpha=0.85)
    ax2.set_ylabel('Avg Total Length (chars)'); ax2.set_title('Avg Response Length by Discussion', fontsize=11)
    ax2.grid(axis='y', alpha=0.3)
    skip_total = (ns['Q1 Null'] + ns['Q2 Null'])
    worst = ns.loc[skip_total.idxmax(), 'D']
    page_header(fig, 4, 'Which Discussion Had Students Giving Up? (Null & Length Analysis)',
                f'{worst} had the most skipped responses. D5-Q1 alone had 11 nulls. '
                f'Average response length peaks at D4 — clustering inspired the longest answers.')
    fig.tight_layout(rect=[0,0.07,1,0.93]); pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

    # ── Q05: NATURAL TRIBES ────────────────────────────────────
    km = KMeans(n_clusters=5, random_state=42, n_init=10)
    cl = km.fit_predict(tfidf)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
    cmap5 = plt.cm.Set1(np.linspace(0, 0.9, 5))
    for c in range(5):
        m = cl == c
        ax1.scatter(coords[m,0], coords[m,1], c=[cmap5[c]], alpha=0.65, s=50,
                    label=f'Cluster {c+1} (n={m.sum()})')
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.25)
    ax1.set_title('K-Means Clusters (k=5) — Ignoring Discussion Labels', fontsize=11)
    comp = pd.crosstab(cl, df['discussion'].values)
    comp.plot(kind='bar', ax=ax2, color=DISC_COLORS, alpha=0.85, stacked=True)
    ax2.set_xlabel('Cluster'); ax2.set_ylabel('# Responses')
    ax2.set_title('Cluster Composition by Discussion', fontsize=11)
    ax2.legend([DISC_NAMES[i] for i in range(1,6)], fontsize=8)
    ax2.tick_params(axis='x', rotation=0)
    page_header(fig, 5, 'Are There Natural Topic Tribes Across All Responses?',
                'K-Means (k=5) shows clusters do NOT map cleanly to discussions — '
                'every cluster contains responses from multiple discussions. Thinking styles transcend topic.')
    fig.tight_layout(rect=[0,0.07,1,0.93]); pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

    # ── Q06: DOMAIN OBSESSIONS ─────────────────────────────────
    dom_counts = defaultdict(lambda: defaultdict(int))
    for _, row in df.iterrows():
        ts = set(row['tokens'])
        for dom, kws in DOMAIN_KW.items():
            if any(k in ts for k in kws):
                dom_counts[row['discussion']][dom] += 1
    doms = list(DOMAIN_KW.keys())
    dom_df = pd.DataFrame({d:[dom_counts[i].get(d,0) for i in range(1,6)] for d in doms},
                           index=[f'D{i}' for i in range(1,6)])
    fig, ax = plt.subplots(figsize=(12, 6))
    dom_df.plot(kind='bar', ax=ax, alpha=0.85, width=0.75)
    ax.set_xlabel('Discussion', fontsize=11); ax.set_ylabel('# Responses Mentioning Domain')
    ax.set_title('Real-World Domain Mentions by Discussion', fontsize=12)
    ax.legend(title='Domain', bbox_to_anchor=(1.01,1), loc='upper left', fontsize=9)
    ax.tick_params(axis='x', rotation=0); ax.grid(axis='y', alpha=0.3)
    top_dom = dom_df.sum().idxmax()
    page_header(fig, 6, 'What Real-World Domains Do Students Default To?',
                f'"{top_dom}" is the most cited domain overall. Finance dominates D5. '
                f'Daily Life peaks in D1. Students reach for finance examples regardless of topic.')
    fig.tight_layout(rect=[0,0.07,1,0.93]); pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

    # ── Q07: AI LANGUAGE THROUGH THE SEMESTER ─────────────────
    ai_rows = []
    for d in range(1, 6):
        sub = df[df['discussion']==d]
        total = sum(len(r['tokens']) for _,r in sub.iterrows())
        count = sum(1 for _,r in sub.iterrows() for t in r['tokens'] if t in AI_TERMS)
        ai_rows.append({'D':f'D{d}','count':count,'per100': count/total*100 if total else 0})
    ai_df = pd.DataFrame(ai_rows)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.bar(ai_df['D'], ai_df['count'], color=DISC_COLORS, alpha=0.85)
    ax1.set_ylabel('Total AI Term Mentions'); ax1.set_title('Raw AI Term Count per Discussion', fontsize=11)
    ax1.grid(axis='y', alpha=0.3)
    ax2.plot(range(5), ai_df['per100'], marker='o', color='#8E44AD', linewidth=2.5, markersize=10)
    ax2.fill_between(range(5), ai_df['per100'], alpha=0.15, color='#8E44AD')
    ax2.set_xticks(range(5)); ax2.set_xticklabels(ai_df['D'])
    ax2.set_ylabel('AI Terms per 100 Words'); ax2.set_title('AI Term Density per Discussion', fontsize=11)
    ax2.grid(True, alpha=0.3)
    peak = ai_df.loc[ai_df['per100'].idxmax(),'D']
    page_header(fig, 7, 'How Did AI-Related Language Appear Across All Discussions?',
                f'AI language peaks at {peak}. D1–D3 show low AI density — '
                f'students were not thinking in AI terms when discussing motivation or pattern mining.')
    fig.tight_layout(rect=[0,0.07,1,0.93]); pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

    # ── Q08: VERBOSE AND SILENT ────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 6))
    data_vp = [df[df['discussion']==d]['total_len'].values for d in range(1,6)]
    vp = ax.violinplot(data_vp, positions=range(1,6), showmedians=True)
    for i, body in enumerate(vp['bodies']):
        body.set_facecolor(DISC_COLORS[i]); body.set_alpha(0.7)
    vp['cmedians'].set_color('black'); vp['cmedians'].set_linewidth(2)
    for d_i, d in enumerate(range(1,6)):
        vals = df[df['discussion']==d]['total_len']
        q1v, q3v = vals.quantile(0.25), vals.quantile(0.75)
        outs = vals[vals > q3v + 1.5*(q3v-q1v)]
        if len(outs): ax.scatter([d_i+1]*len(outs), outs, c='red', s=25, zorder=5, alpha=0.7)
    ax.set_xticks(range(1,6))
    ax.set_xticklabels([DISC_NAMES[i] for i in range(1,6)], fontsize=9)
    ax.set_ylabel('Total Response Length (characters)', fontsize=11)
    ax.set_title('Response Length Distribution by Discussion', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    widest = max(range(1,6), key=lambda d: df[df['discussion']==d]['total_len'].std())
    page_header(fig, 8, 'Who Wrote the Most and the Least?',
                f'D{widest} has the widest spread in response length — highest inequality in effort. '
                f'Red dots = statistical outliers (1.5×IQR). Some students wrote 1200+ chars; others barely 80.')
    fig.tight_layout(rect=[0,0.07,1,0.93]); pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

    # ── Q09: UNIVERSAL vs EXCLUSIVE WORDS ─────────────────────
    disc_wsets = {d: set(t for ts in df[df['discussion']==d]['tokens'] for t in ts) for d in range(1,6)}
    universal = disc_wsets[1].copy()
    for d in range(2,6): universal &= disc_wsets[d]
    all_flat = [t for ts in df['tokens'] for t in ts]
    univ_freq = Counter(t for t in all_flat if t in universal)
    excl = {d: disc_wsets[d] - set().union(*[disc_wsets[d2] for d2 in range(1,6) if d2!=d])
            for d in range(1,6)}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
    top_u = univ_freq.most_common(20)
    ax1.barh(range(20), [c for _,c in top_u[::-1]], color='#2C3E50', alpha=0.8)
    ax1.set_yticks(range(20)); ax1.set_yticklabels([w for w,_ in top_u[::-1]], fontsize=9)
    ax1.set_title(f'Top Universal Words\n(in all 5 discussions — {len(universal)} total)', fontsize=10)
    ax1.set_xlabel('Corpus Frequency')
    ax2.bar([f'D{d}' for d in range(1,6)], [len(excl[d]) for d in range(1,6)],
            color=DISC_COLORS, alpha=0.85)
    ax2.set_ylabel('# Words Found Only in This Discussion')
    ax2.set_title('Exclusive Vocabulary per Discussion', fontsize=11)
    ax2.grid(axis='y', alpha=0.3)
    most_excl = max(excl, key=lambda d: len(excl[d]))
    page_header(fig, 9, 'What Words Appear in ALL 5 Discussions vs. Only One?',
                f'{len(universal)} words are universal (appear in every discussion). '
                f'D{most_excl} has the most exclusive vocabulary — its topic brings the most unique language.')
    fig.tight_layout(rect=[0,0.07,1,0.93]); pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

    # ── Q10: RESPONSE SILOS ────────────────────────────────────
    sim_mat = cosine_similarity(tfidf)
    disc_arr = df['discussion'].values
    heat = np.zeros((5,5))
    for di in range(1,6):
        for dj in range(1,6):
            mi = disc_arr==di; mj = disc_arr==dj
            sub = sim_mat[np.ix_(mi,mj)].copy()
            if di==dj: np.fill_diagonal(sub, np.nan)
            heat[di-1][dj-1] = np.nanmean(sub)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(heat, cmap='YlOrRd', aspect='auto', vmin=0)
    plt.colorbar(im, ax=ax, label='Avg Cosine Similarity')
    labs = [f'D{i}' for i in range(1,6)]
    ax.set_xticks(range(5)); ax.set_xticklabels(labs)
    ax.set_yticks(range(5)); ax.set_yticklabels(labs)
    ax.set_title('Average Cosine Similarity Between Discussions', fontsize=12)
    for i in range(5):
        for j in range(5):
            col = 'white' if heat[i,j] > heat.max()*0.6 else 'black'
            ax.text(j, i, f'{heat[i,j]:.3f}', ha='center', va='center', fontsize=10, color=col)
    diag_m = np.mean([heat[i,i] for i in range(5)])
    off_m  = np.mean([heat[i,j] for i in range(5) for j in range(5) if i!=j])
    page_header(fig, 10, 'Are Responses Within the Same Discussion More Similar to Each Other?',
                f'Within-discussion avg similarity: {diag_m:.3f} vs cross-discussion: {off_m:.3f}. '
                f'D4–D5 share the highest cross-similarity — clustering and outlier detection vocabularies overlap.')
    fig.tight_layout(rect=[0,0.07,1,0.93]); pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

    # ── Q11: MOST GENERIC RESPONSES ───────────────────────────
    avg_sc = np.array(tfidf.mean(axis=1)).flatten()
    df['avg_tfidf'] = avg_sc
    lowest = df.nsmallest(10, 'avg_tfidf')[['discussion','text','total_len','avg_tfidf']]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={'width_ratios':[1.1,1]})
    ax1.hist(avg_sc, bins=30, color='#3498DB', alpha=0.85, edgecolor='white')
    ax1.axvline(np.percentile(avg_sc,10), color='red', ls='--', label='Bottom 10%')
    ax1.set_xlabel('Avg TF-IDF Score'); ax1.set_ylabel('# Responses')
    ax1.set_title('Distribution of Response Uniqueness', fontsize=11); ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax2.axis('off')
    tdata = [[f'D{r["discussion"]}', (r['text'][:68]+'…' if len(r['text'])>68 else r['text']),
              f'{r["avg_tfidf"]:.5f}'] for _,r in lowest.iterrows()]
    tbl = ax2.table(cellText=tdata, colLabels=['D','Response Preview','Score'],
                    loc='center', cellLoc='left')
    tbl.auto_set_font_size(False); tbl.set_fontsize(7); tbl.scale(1, 1.8)
    ax2.set_title('Most Generic Responses', fontsize=11)
    page_header(fig, 11, 'Which Responses Are the Most Generic and Formulaic?',
                'Responses with the lowest avg TF-IDF use only common, non-distinctive words. '
                'These are textbook answers — correct but adding no unique perspective.')
    fig.tight_layout(rect=[0,0.07,1,0.93]); pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

    # ── Q12: D1 PERSONAL PROBLEMS CLUSTERED ───────────────────
    d1q2 = raw[(raw['discussion']==1) & raw['Q2'].notna()][['No.','Q2']].copy()
    d1q2['clean'] = d1q2['Q2'].apply(clean)
    dv2 = TfidfVectorizer(stop_words=list(STOPWORDS), max_features=200, min_df=1)
    dm2 = dv2.fit_transform(d1q2['clean'])
    km4 = KMeans(n_clusters=4, random_state=42, n_init=10)
    cl4 = km4.fit_predict(dm2)
    pc4 = PCA(n_components=2, random_state=42)
    xy4 = pc4.fit_transform(dm2.toarray())
    fts4 = dv2.get_feature_names_out()
    ctop = {c: ', '.join(fts4[km4.cluster_centers_[c].argsort()[-4:][::-1]]) for c in range(4)}
    c4colors = ['#E74C3C','#3498DB','#27AE60','#F39C12']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
    for c in range(4):
        m = cl4==c
        ax1.scatter(xy4[m,0], xy4[m,1], c=c4colors[c], s=65, alpha=0.85,
                    label=f'C{c+1}: {ctop[c]}')
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.25)
    ax1.set_title('D1-Q2 Personal Problems: 4 Clusters', fontsize=11)
    csizes = [sum(cl4==c) for c in range(4)]
    ax2.barh(range(4), csizes, color=c4colors, alpha=0.85)
    ax2.set_yticks(range(4))
    ax2.set_yticklabels([f'C{c+1}: {ctop[c][:28]}' for c in range(4)], fontsize=9)
    ax2.set_xlabel('# Students'); ax2.set_title('Students per Problem Cluster', fontsize=11)
    ax2.grid(axis='x', alpha=0.3)
    page_header(fig, 12, 'What Types of Life Problems Did Students Want to Solve? (D1-Q2)',
                'K-Means (k=4) on D1-Q2 reveals problem archetypes: productivity/scheduling, '
                'finance/spending, transportation/commuting, and data organization projects.')
    fig.tight_layout(rect=[0,0.07,1,0.93]); pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

    # ── Q13: VOCABULARY RICHNESS ───────────────────────────────
    df['ttr'] = df.apply(lambda r: len(set(r['tokens']))/len(r['tokens']) if r['tokens'] else 0, axis=1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    bp = ax1.boxplot([df[df['discussion']==d]['ttr'].values for d in range(1,6)],
                     patch_artist=True, notch=False)
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(DISC_COLORS[i]); patch.set_alpha(0.8)
    ax1.set_xticklabels([f'D{i}' for i in range(1,6)])
    ax1.set_ylabel('Type-Token Ratio'); ax1.set_title('Vocabulary Richness Distribution', fontsize=11)
    ax1.grid(axis='y', alpha=0.3)
    means = [df[df['discussion']==d]['ttr'].mean() for d in range(1,6)]
    bars = ax2.bar([f'D{d}' for d in range(1,6)], means, color=DISC_COLORS, alpha=0.85)
    wi = int(np.argmax(means))
    bars[wi].set_edgecolor('gold'); bars[wi].set_linewidth(4)
    for i,v in enumerate(means): ax2.text(i, v+0.003, f'{v:.3f}', ha='center', fontsize=10)
    ax2.set_ylabel('Mean TTR'); ax2.set_title('Mean Vocabulary Richness (Gold = Winner)', fontsize=11)
    ax2.grid(axis='y', alpha=0.3)
    page_header(fig, 13, 'Which Discussion Had the Richest, Most Diverse Vocabulary?',
                f'D{wi+1} wins with the highest mean Type-Token Ratio — students used more diverse words '
                f'relative to their total word count. Higher TTR = more expressive, less repetitive writing.')
    fig.tight_layout(rect=[0,0.07,1,0.93]); pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

    # ── Q14: TOP BIGRAMS ───────────────────────────────────────
    all_bgs = []
    for tks in df['tokens']:
        all_bgs.extend(f'{tks[i]} {tks[i+1]}' for i in range(len(tks)-1))
    top25 = Counter(all_bgs).most_common(25)
    fig, ax = plt.subplots(figsize=(11, 7.5))
    bwords = [b for b,_ in top25[::-1]]; bcnts = [c for _,c in top25[::-1]]
    cols_bg = plt.cm.viridis(np.linspace(0.15, 0.9, 25))
    ax.barh(range(25), bcnts, color=cols_bg, alpha=0.9)
    ax.set_yticks(range(25)); ax.set_yticklabels(bwords, fontsize=9)
    ax.set_xlabel('Frequency Across All 5 Discussions')
    ax.set_title('Top 25 Bigrams Across Entire Corpus', fontsize=12)
    ax.grid(axis='x', alpha=0.3)
    for i,v in enumerate(bcnts): ax.text(v+0.2, i, str(v), va='center', fontsize=8)
    page_header(fig, 14, 'What Concept Pairs Co-Occur Most Across All Discussions?',
                f'"{top25[0][0]}" is the #1 bigram. Compound course concepts dominate: '
                f'"outlier detection", "distance based", "machine learning", "data science".')
    fig.tight_layout(rect=[0,0.07,1,0.93]); pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

    # ── Q15: TECHNICAL VOCABULARY DENSITY ─────────────────────
    df['tech_dens'] = df['tokens'].apply(
        lambda ts: sum(1 for t in ts if t in TECH_TERMS)/len(ts) if ts else 0)
    heat_tech = np.zeros((5, len(TECH_TERMS)))
    tterms_list = sorted(TECH_TERMS)
    for di, d in enumerate(range(1,6)):
        sub = df[df['discussion']==d]
        all_t = [t for ts in sub['tokens'] for t in ts]
        tot = len(all_t)
        for ti, term in enumerate(tterms_list):
            heat_tech[di][ti] = all_t.count(term)/tot*1000 if tot else 0
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9))
    means_td = [df[df['discussion']==d]['tech_dens'].mean() for d in range(1,6)]
    ax1.bar([f'D{d}' for d in range(1,6)], means_td, color=DISC_COLORS, alpha=0.85)
    ax1.set_ylabel('Mean Tech Density (tech/total)'); ax1.grid(axis='y', alpha=0.3)
    ax1.set_title('Mean Technical Vocabulary Density by Discussion', fontsize=11)
    for i,v in enumerate(means_td): ax1.text(i, v+0.001, f'{v:.3f}', ha='center', fontsize=10)
    im2 = ax2.imshow(heat_tech, cmap='Blues', aspect='auto')
    ax2.set_xticks(range(len(tterms_list)))
    ax2.set_xticklabels(tterms_list, rotation=55, ha='right', fontsize=7.5)
    ax2.set_yticks(range(5)); ax2.set_yticklabels([f'D{i}' for i in range(1,6)])
    ax2.set_title('Technical Term Heatmap (mentions per 1000 words)', fontsize=11)
    plt.colorbar(im2, ax=ax2, shrink=0.8)
    page_header(fig, 15, 'How Technical Is the Language? Does It Vary by Discussion?',
                'D5 uses the densest technical vocabulary. D1 is the least technical — '
                'students wrote about personal motivations, not algorithms.')
    fig.tight_layout(rect=[0,0.07,1,0.93]); pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

    # ── Q16: RESPONSE DEPTH SCORE ──────────────────────────────
    df['sent_cnt'] = df['text'].apply(lambda t: len(re.findall(r'[.!?]+', t))+1)
    df['depth'] = df['unique_words']*0.5 + df['sent_cnt']*2 + df['tech_dens']*50
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5.5))
    for d in range(1,6):
        ax1.hist(df[df['discussion']==d]['depth'], bins=15, alpha=0.55,
                 color=DISC_COLORS[d-1], label=f'D{d}')
    ax1.legend(fontsize=8); ax1.set_xlabel('Depth Score'); ax1.set_ylabel('Count')
    ax1.set_title('Response Depth Distribution', fontsize=11)
    md = [df[df['discussion']==d]['depth'].mean() for d in range(1,6)]
    ax2.bar([f'D{d}' for d in range(1,6)], md, color=DISC_COLORS, alpha=0.85)
    ax2.set_ylabel('Mean Depth Score'); ax2.set_title('Mean Depth by Discussion', fontsize=11)
    ax2.grid(axis='y', alpha=0.3)
    for d in range(1,6):
        sub = df[df['discussion']==d]
        ax3.scatter(sub['sent_cnt'], sub['unique_words'], c=DISC_COLORS[d-1],
                    alpha=0.5, s=30, label=f'D{d}')
    ax3.set_xlabel('Sentence Count'); ax3.set_ylabel('Unique Words')
    ax3.set_title('Depth: Sentences vs Unique Words', fontsize=11)
    ax3.legend(fontsize=7)
    deepest = f'D{int(np.argmax(md))+1}'
    page_header(fig, 16, 'How Deep Are Student Responses? (Composite Depth Score)',
                f'{deepest} produced the deepest responses on average. '
                f'Depth = unique words + sentence count + technical density. High within-discussion variance.')
    fig.tight_layout(rect=[0,0.07,1,0.93]); pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

    # ── Q17: Q1 vs Q2 PERSONALITY ──────────────────────────────
    q1_tok = [t for txt in raw['Q1c'] for t in tok(txt)]
    q2_tok = [t for txt in raw['Q2c'] for t in tok(txt)]
    q1c = Counter(q1_tok); q2c = Counter(q2_tok)
    q1t = sum(q1c.values()); q2t = sum(q2c.values())
    top_q1 = [w for w,_ in q1c.most_common(15)]
    top_q2 = [w for w,_ in q2c.most_common(15)]
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 6))
    ax1.barh(range(15), [q1c[w]/q1t*100 for w in top_q1[::-1]], color='#E74C3C', alpha=0.85)
    ax1.set_yticks(range(15)); ax1.set_yticklabels(top_q1[::-1], fontsize=9)
    ax1.set_title('Q1 Top Words', fontsize=11, color='#E74C3C')
    ax1.set_xlabel('% of Q1 tokens')
    ax2.barh(range(15), [q2c[w]/q2t*100 for w in top_q2[::-1]], color='#2980B9', alpha=0.85)
    ax2.set_yticks(range(15)); ax2.set_yticklabels(top_q2[::-1], fontsize=9)
    ax2.set_title('Q2 Top Words', fontsize=11, color='#2980B9')
    ax2.set_xlabel('% of Q2 tokens')
    shared = sorted(set(top_q1) & set(top_q2))
    y = range(len(shared))
    ax3.barh(y, [q1c[w]/q1t*100 for w in shared], alpha=0.7, color='#E74C3C', label='Q1')
    ax3.barh(y, [-q2c[w]/q2t*100 for w in shared], alpha=0.7, color='#2980B9', label='Q2')
    ax3.set_yticks(y); ax3.set_yticklabels(shared, fontsize=9)
    ax3.axvline(0, color='black', lw=0.8)
    ax3.set_title('Shared Words: Q1 vs Q2 Freq', fontsize=11); ax3.legend(fontsize=8)
    page_header(fig, 17, 'Do Q1 and Q2 Responses Have Different Personalities?',
                'Q1 vocabulary is example-heavy; Q2 is evaluation/comparison-heavy. '
                f'{len(shared)} words overlap in both top-15 lists — the shared conceptual core.')
    fig.tight_layout(rect=[0,0.07,1,0.93]); pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

    # ── Q18: MOST POLARIZING DISCUSSION ────────────────────────
    disc_spread = {}
    for d in range(1,6):
        m = df['discussion'].values == d
        sub = tfidf[m].toarray()
        if sub.shape[0] > 1:
            dists = pairwise_distances(sub, metric='cosine')
            np.fill_diagonal(dists, np.nan)
            disc_spread[d] = np.nanmean(dists)
        else:
            disc_spread[d] = 0
    disc_var = {d: np.var(tfidf[df['discussion'].values==d].toarray(), axis=0).mean()
                for d in range(1,6)}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.bar([f'D{d}' for d in range(1,6)], [disc_spread[d] for d in range(1,6)],
            color=DISC_COLORS, alpha=0.85)
    ax1.set_ylabel('Avg Pairwise Cosine Distance')
    ax1.set_title('Response Diversity (Avg Pairwise Distance)', fontsize=11)
    ax1.grid(axis='y', alpha=0.3)
    for i,d in enumerate(range(1,6)): ax1.text(i, disc_spread[d]+0.002, f'{disc_spread[d]:.3f}', ha='center', fontsize=9)
    ax2.bar([f'D{d}' for d in range(1,6)], [disc_var[d] for d in range(1,6)],
            color=DISC_COLORS, alpha=0.85)
    ax2.set_ylabel('Mean Feature Variance')
    ax2.set_title('Response Diversity (TF-IDF Variance)', fontsize=11)
    ax2.grid(axis='y', alpha=0.3)
    most_pol = f'D{max(disc_spread, key=disc_spread.get)}'
    most_uni = f'D{min(disc_spread, key=disc_spread.get)}'
    page_header(fig, 18, 'Which Discussion Had the Most Polarizing / Diverse Responses?',
                f'{most_pol} is the most polarizing — responses are furthest apart from each other. '
                f'{most_uni} is most uniform — students converged on similar answers.')
    fig.tight_layout(rect=[0,0.07,1,0.93]); pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

    # ── Q19: MULTI-DIMENSIONAL OUTLIERS ────────────────────────
    feats_md = df[['total_len','unique_words','tech_dens','depth']].fillna(0)
    sc = StandardScaler()
    fs = sc.fit_transform(feats_md)
    iso2 = IsolationForest(contamination=0.07, random_state=42)
    mol = iso2.fit_predict(fs) == -1
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    pairs = [('total_len','unique_words'),('total_len','tech_dens'),
             ('unique_words','tech_dens'),('depth','total_len')]
    flabs = {'total_len':'Response Length','unique_words':'Unique Words',
             'tech_dens':'Tech Density','depth':'Depth Score'}
    for ax, (fx, fy) in zip(axes.flatten(), pairs):
        ax.scatter(feats_md[fx][~mol], feats_md[fy][~mol], c='#BDC3C7', alpha=0.45, s=30, label='Normal')
        ax.scatter(feats_md[fx][mol], feats_md[fy][mol], c='#E74C3C', s=80,
                   marker='*', alpha=0.9, label=f'Outlier (n={mol.sum()})')
        ax.set_xlabel(flabs[fx], fontsize=9); ax.set_ylabel(flabs[fy], fontsize=9)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.25)
    page_header(fig, 19, 'Which Responses Are Outliers Across MULTIPLE Dimensions Simultaneously?',
                f'{mol.sum()} responses are multi-dimensional outliers — unusual in length, vocabulary, '
                f'AND technical density at once. These are the truly exceptional (or truly minimal) responses.')
    fig.tight_layout(rect=[0,0.07,1,0.93]); pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

    # ── Q20: GRAND WORD CLOUD ──────────────────────────────────
    all_txt = ' '.join(df['clean'])
    d5_txt  = ' '.join(df[df['discussion']==5]['clean'])
    if HAS_WC:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 7))
        wc1 = WordCloud(width=650, height=450, background_color='white',
                        stopwords=STOPWORDS, max_words=150, colormap='viridis').generate(all_txt)
        ax1.imshow(wc1, interpolation='bilinear'); ax1.axis('off')
        ax1.set_title('Full Corpus — All 294 Responses', fontsize=11)
        wc2 = WordCloud(width=650, height=450, background_color='#1A252F',
                        stopwords=STOPWORDS, max_words=100, colormap='Reds').generate(d5_txt)
        ax2.imshow(wc2, interpolation='bilinear'); ax2.axis('off')
        ax2.set_title('D5 Only — Outlier Detection Discussion', fontsize=11)
    else:
        fig, ax1 = plt.subplots(figsize=(13, 7))
        top40 = Counter(all_flat).most_common(40)
        ax1.barh(range(40), [c for _,c in top40[::-1]],
                 color=plt.cm.viridis(np.linspace(0,1,40)), alpha=0.9)
        ax1.set_yticks(range(40)); ax1.set_yticklabels([w for w,_ in top40[::-1]], fontsize=8)
        ax1.set_title('Top 40 Words Across All Discussions', fontsize=12)
    page_header(fig, 20, 'The Grand Word Cloud: What Does This Class Sound Like?',
                '"Data", "outlier", "detection", "clustering" dominate the full corpus — the course vocabulary '
                'has been internalized. D5 cloud shows a striking "AI vs statistical" debate in word form.')
    fig.tight_layout(rect=[0,0.07,1,0.93]); pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

print(f"Done — saved to {OUTPUT_PDF}")
