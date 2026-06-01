"""
EDA Script v2 — 위험 운전 행동 분류 프로젝트
분석 흐름:
  전처리: 완전 중복 특징 제거 → 표준화
  ① 클래스 분포              → Macro-F1 선택 근거
  ② 가우시안 검정 + KDE      → Q1: 가우시안 가정 성립 여부
  ③ 특징 간 상관             → Q1: NB 독립 가정 위반
  ④ 클래스별 공분산 비교      → Q2: LDA vs QDA (F-통계량 기반 특징 선택)
  ⑤ PCA vs LDA 투영 비교    → Q3: 선형 결정 경계 충분한가
  ⑥ QDA 모수 수 vs 표본 수  → Q2+Q4: QDA 차원 문제 정량화
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif
from scipy import stats
import warnings, platform, logging
from pathlib import Path

warnings.filterwarnings('ignore')
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
_sys = platform.system()
_font = 'AppleGothic' if _sys == 'Darwin' else ('Malgun Gothic' if _sys == 'Windows' else 'DejaVu Sans')
matplotlib.rc('font', family=_font)
matplotlib.rcParams['axes.unicode_minus'] = False

# ── 설정 ─────────────────────────────────────────────────
DATA_PATH  = Path(__file__).parent.parent / 'data' / 'features_14.csv'
SAVE_DIR   = Path(__file__).parent / 'result'
SAVE_DIR.mkdir(exist_ok=True)
CLASS_NAMES = {1: '급가속', 2: '급우회전', 3: '급좌회전', 4: '급정거'}
COLORS = ['#E24B4A', '#378ADD', '#1D9E75', '#BA7517']

# ── 데이터 로드 ───────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
X  = df.drop(columns=['Target'])
y  = df['Target']
classes = sorted(y.unique())

print("=" * 65)
print("EDA v2 — 위험 운전 행동 분류")
print("=" * 65)
print(f"원본: {X.shape[0]}개 표본, {X.shape[1]}개 특징")

# ── 전처리 0: 완전 중복 특징 제거 ────────────────────────
corr_mat = X.corr().abs().values.copy()
np.fill_diagonal(corr_mat, 0)
corr_df = pd.DataFrame(corr_mat, index=X.columns, columns=X.columns)

to_drop = set()
for i, col_i in enumerate(X.columns):
    if col_i in to_drop:
        continue
    for col_j in X.columns[i+1:]:
        if col_j not in to_drop and corr_df.loc[col_i, col_j] > 0.999:
            to_drop.add(col_j)

X_clean = X.drop(columns=list(to_drop))
feat_names = X_clean.columns.tolist()
d = len(feat_names)

print(f"중복 제거 후: {d}개 특징  (제거: {len(to_drop)}개 → {list(to_drop)[:5]}{'...' if len(to_drop)>5 else ''})")

# ── 전처리 1: 표준화 ──────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_clean)


# ────────────────────────────────────────────────────────
# ① 클래스 분포
# ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
counts = y.value_counts().sort_index()
bars = ax.bar([CLASS_NAMES[i] for i in counts.index], counts.values, color=COLORS)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
            str(val), ha='center', va='bottom', fontsize=10)
ax.set_title('① 클래스 분포')
ax.set_ylabel('샘플 수')
ax.set_ylim(0, counts.max() * 1.15)
plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/eda_1_class_dist.png', dpi=150)
plt.close()

imbalance = counts.max() / counts.min()
print(f"\n[①] 클래스 분포  (최대/최소 비율: {imbalance:.2f})")
for c in classes:
    n = (y == c).sum()
    print(f"  클래스 {c} ({CLASS_NAMES[c]}): {n}개 ({n/len(y)*100:.1f}%)")


# ────────────────────────────────────────────────────────
# ② 가우시안 검정 + KDE  →  Q1
# ────────────────────────────────────────────────────────
repr_feats = ['AccMeanX', 'AccMeanZ', 'AccStdX', 'AccStdZ',
              'GyroMeanY', 'GyroMeanZ', 'GyroStdY', 'GyroStdZ']
repr_feats = [f for f in repr_feats if f in feat_names]

fig, axes = plt.subplots(2, 4, figsize=(14, 6))
for i, feat in enumerate(repr_feats[:8]):
    ax = axes[i // 4][i % 4]
    for j, cls in enumerate(classes):
        X_clean.loc[y == cls, feat].plot.kde(
            ax=ax, label=CLASS_NAMES[cls], color=COLORS[j], linewidth=1.5)
    ax.set_title(feat, fontsize=9)
    ax.legend(fontsize=7)
    ax.set_ylabel('')
plt.suptitle('② 클래스별 특징 분포 (KDE)\n→ Q1: 가우시안 가정 성립 여부', y=1.02)
plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/eda_2_kde.png', dpi=150, bbox_inches='tight')
plt.close()

norm_results = {}
for cls in classes:
    X_cls = X_clean[y == cls]
    rej = sum(1 for f in feat_names if stats.shapiro(X_cls[f].values)[1] < 0.05)
    norm_results[cls] = rej / d

print(f"\n[②] 정규성 검정 (Shapiro-Wilk, α=0.05)  →  Q1")
for cls, r in norm_results.items():
    print(f"  클래스 {cls} ({CLASS_NAMES[cls]}): {r*100:.0f}% 기각")


# ────────────────────────────────────────────────────────
# ③ 특징 간 상관  →  Q1 (NB 독립 가정)
# ────────────────────────────────────────────────────────
# F-통계량 상위 20개 특징만 시각화 (판별력 높은 특징 중심)
f_stats, _ = f_classif(X_scaled, y)
top20_idx = np.argsort(f_stats)[-20:][::-1]
top20 = [feat_names[i] for i in top20_idx]

fig, ax = plt.subplots(figsize=(10, 8))
corr_top = X_clean[top20].corr()
mask = np.triu(np.ones_like(corr_top, dtype=bool))
short = [f.replace('Acc', 'A_').replace('Gyro', 'G_') for f in top20]
sns.heatmap(corr_top, ax=ax, mask=mask, cmap='RdBu_r', center=0,
            annot=True, fmt='.1f', annot_kws={'size': 7},
            xticklabels=short, yticklabels=short)
ax.set_title('③ 특징 간 상관 행렬 (F-통계량 상위 20)\n→ Q1: NB 독립 가정 위반 정도', fontsize=11)
plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/eda_3_feature_corr.png', dpi=150, bbox_inches='tight')
plt.close()

cm = X_clean.corr().abs().values.copy()
np.fill_diagonal(cm, 0)
total_pairs = d * (d - 1) // 2
high07 = int((cm > 0.7).sum() / 2)
print(f"\n[③] 특징 간 상관  →  Q1")
print(f"  |r|>0.7 쌍: {high07}개 / {total_pairs}쌍 ({high07/total_pairs*100:.1f}%)")


# ────────────────────────────────────────────────────────
# ④ 클래스별 공분산 비교  →  Q2  (F-통계량 기반 top15)
# ────────────────────────────────────────────────────────
top15 = [feat_names[i] for i in np.argsort(f_stats)[-15:][::-1]]

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for i, cls in enumerate(classes):
    corr = X_clean.loc[y == cls, top15].corr()
    sns.heatmap(corr, ax=axes[i], cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                xticklabels=False, yticklabels=False, cbar=(i == 3))
    axes[i].set_title(f'클래스 {cls}\n({CLASS_NAMES[cls]})', fontsize=9)
plt.suptitle('④ 클래스별 상관 행렬 (F-통계량 상위 15)\n→ Q2: 클래스 간 공분산 구조 차이', y=1.04)
plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/eda_4_class_cov.png', dpi=150, bbox_inches='tight')
plt.close()

corr_mats = {cls: X_clean.loc[y == cls, top15].corr().values for cls in classes}
pairs = [(classes[a], classes[b])
         for a in range(len(classes)) for b in range(a+1, len(classes))]
diffs = [np.linalg.norm(corr_mats[ca] - corr_mats[cb], 'fro') for ca, cb in pairs]
print(f"\n[④] 클래스별 공분산 차이 (Frobenius norm)  →  Q2")
for (ca, cb), diff in zip(pairs, diffs):
    print(f"  클래스 {ca} vs {cb}: {diff:.2f}")
print(f"  평균: {np.mean(diffs):.2f}")


# ────────────────────────────────────────────────────────
# ⑤ PCA vs LDA 투영 비교  →  Q3
# ────────────────────────────────────────────────────────
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
exp_var = pca.explained_variance_ratio_

lda_proj = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda_proj.fit_transform(X_scaled, y)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for j, cls in enumerate(classes):
    mask = (y == cls).values
    axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1],
                    label=CLASS_NAMES[cls], alpha=0.5, s=15, color=COLORS[j])
    axes[1].scatter(X_lda[mask, 0], X_lda[mask, 1],
                    label=CLASS_NAMES[cls], alpha=0.5, s=15, color=COLORS[j])

axes[0].set_xlabel(f'PC1 ({exp_var[0]*100:.1f}%)')
axes[0].set_ylabel(f'PC2 ({exp_var[1]*100:.1f}%)')
axes[0].set_title('⑤-A  PCA 투영\n(비지도: 분산 최대화)')
axes[0].legend(fontsize=9)

axes[1].set_xlabel('LD1')
axes[1].set_ylabel('LD2')
axes[1].set_title('⑤-B  LDA 투영\n(지도: 클래스 분리 최대화)')
axes[1].legend(fontsize=9)

plt.suptitle('→ Q3: 선형 결합으로 클래스가 분리되는가', y=1.02)
plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/eda_5_pca_lda.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"\n[⑤] PCA vs LDA 투영  →  Q3")
print(f"  PCA PC1+PC2 설명 분산: {sum(exp_var)*100:.1f}%")
print(f"  → LDA 투영 산점도에서 선형 분리 가능 여부 확인 (그림 참고)")


# ────────────────────────────────────────────────────────
# ⑥ QDA 모수 수 vs 표본 수  →  Q2 + Q4
# ────────────────────────────────────────────────────────
print(f"\n[⑥] QDA 모수 수 vs 표본 수  →  Q2 + Q4")
print(f"  특징 차원 d = {d}  (중복 제거 후)")
qda_params_per_class = d * (d + 1) // 2
print(f"  QDA: 클래스당 공분산 모수 = d(d+1)/2 = {qda_params_per_class}")
print(f"  LDA: 공통 공분산 모수 = {qda_params_per_class}  (전체 합산 1회)")
for cls in classes:
    n_cls = int((y == cls).sum())
    ratio = n_cls / qda_params_per_class
    flag = "⚠ 표본 부족" if ratio < 2 else "OK"
    print(f"  클래스 {cls} ({CLASS_NAMES[cls]}): n={n_cls}, n/모수={ratio:.2f}  {flag}")

pca_full = PCA().fit(X_scaled)
n_for_90 = int(np.argmax(np.cumsum(pca_full.explained_variance_ratio_) >= 0.90)) + 1
qda_pca_params = n_for_90 * (n_for_90 + 1) // 2
print(f"\n  PCA {n_for_90}차원 축소 후 QDA 모수: {qda_pca_params}")
print(f"  → PCA 전처리로 QDA 분산 폭발 완화 가능 여부 실험 필요")


# ────────────────────────────────────────────────────────
# 가설 종합 (Q1~Q4 대응)
# ────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("▶  EDA 기반 가설 — 2.2의 연구 목표 Q1~Q4 대응")
print("=" * 65)
avg_rej = np.mean(list(norm_results.values()))
cov_avg = np.mean(diffs)
hc_ratio = high07 / total_pairs

print(f"""
H1 [Q1 — 가우시안 가정]:
  특징의 평균 {avg_rej*100:.0f}%가 정규성 기각, 특징 간 상관 쌍 {high07}개
  → 가우시안 생성 모델(LDA/QDA/NB) 전반의 성능 제한 예상
  → NB는 독립 가정도 위반하므로 생성 모델 중 가장 불리할 것

H2 [Q2 — 공분산 구조]:
  클래스 간 공분산 차이 평균 {cov_avg:.2f} → QDA 이론적 우위
  단, d={d}에서 QDA 클래스당 모수={qda_params_per_class} vs 표본 ~250개 → 분산 폭발 위험
  → PCA 차원 축소 전후 QDA 성능 비교 실험 필요

H3 [Q3 — 결정 경계 선형성]:
  PCA 2D 설명 분산 {sum(exp_var)*100:.1f}% — 클래스 1·4 선형 분리 어려움
  → LDA 투영 결과와 비교하여 선형 모델의 한계 범위 확인
  → 비선형 모델(커널 SVM, 트리/앙상블)이 선형 모델보다 유리할 것

H4 [Q4 — 모델 복잡도와 일반화]:
  비겹침 윈도우 기준 유효 독립 표본 수 소규모 예상
  → 고복잡도 모델(깊은 트리, 과도한 앙상블)은 과적합 위험
  → 학습 곡선으로 편향-분산 트레이드오프 확인 필요
""")
print("=" * 65)
print(f"그림 저장: {SAVE_DIR}/eda_1~5.png")