"""
코드 ③ — 진단 실험 3종
  실험 1: QDA × PCA 차원 sweep   → H2 (차원 축소로 QDA 회복 여부)
  실험 2: 학습 곡선 (편향-분산)   → H4 (모델별 train-test 격차)
  실험 3: 무작위 분할 vs purged   → 누수 시연
출력: exp1_qda_pca.png, exp2_learning_curve.png, exp3_leakage.png + 콘솔
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import warnings, platform, logging
from pathlib import Path
from data_prep import load_and_clean, purged_temporal_split, PurgedBlockedCV

_OUT_DIR = Path(__file__).parent

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

warnings.filterwarnings('ignore')
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
_sys = platform.system()
_font = 'AppleGothic' if _sys == 'Darwin' else ('Malgun Gothic' if _sys == 'Windows' else 'DejaVu Sans')
matplotlib.rc('font', family=_font)
matplotlib.rcParams['axes.unicode_minus'] = False

X, y, _ = load_and_clean()
X, y = X.values, y.values
tr, te = purged_temporal_split(y, train_ratio=0.7)
X_tr, X_te, y_tr, y_te = X[tr], X[te], y[tr], y[te]

def pipe(*steps):
    return Pipeline(list(steps))

# ══════════════════════════════════════════════════════════
# 실험 1: QDA × PCA 차원 sweep  →  H2
# ══════════════════════════════════════════════════════════
dims = [2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 35, 40, 45, 52]
qda_f1, lda_f1 = [], []
for k in dims:
    # reg_param=0.0은 고차원에서 공분산이 singular → 적합 불가(=H2의 극단형).
    # 곡선을 끝까지 그리기 위해 아주 작은 정규화(0.01) 고정.
    try:
        qda = pipe(('sc', StandardScaler()), ('pca', PCA(n_components=k)),
                   ('clf', QuadraticDiscriminantAnalysis(reg_param=0.01)))
        qda.fit(X_tr, y_tr)
        qda_f1.append(f1_score(y_te, qda.predict(X_te), average='macro'))
    except np.linalg.LinAlgError:
        qda_f1.append(np.nan)

    lda = pipe(('sc', StandardScaler()), ('pca', PCA(n_components=k)),
               ('clf', LinearDiscriminantAnalysis()))
    lda.fit(X_tr, y_tr)
    lda_f1.append(f1_score(y_te, lda.predict(X_te), average='macro'))

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(dims, qda_f1, 'o-', color='#E24B4A', label='QDA', linewidth=2)
ax.plot(dims, lda_f1, 's-', color='#378ADD', label='LDA', linewidth=2)
best_k = dims[int(np.argmax(qda_f1))]
ax.axvline(best_k, color='gray', linestyle='--', alpha=0.6)
ax.text(best_k + 1, min(qda_f1), f'QDA 최적 {best_k}차원', fontsize=9, color='gray')
ax.set_xlabel('PCA 차원 수')
ax.set_ylabel('Test macro-F1')
ax.set_title('실험 1: QDA × PCA 차원 sweep → H2\n저차원에서 QDA 회복, 고차원에서 분산 폭발')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(_OUT_DIR / 'exp1_qda_pca.png', dpi=150)
plt.close()

print("=" * 60)
print("[실험 1] QDA × PCA 차원 sweep  → H2")
print(f"  QDA 최고 macro-F1: {max(qda_f1):.4f} @ {best_k}차원")
print(f"  QDA 52차원(전체):  {qda_f1[-1]:.4f}")
print(f"  LDA는 차원 변화에 안정적: {min(lda_f1):.4f} ~ {max(lda_f1):.4f}")

# ══════════════════════════════════════════════════════════
# 실험 2: 학습 곡선 (편향-분산)  →  H4
# ══════════════════════════════════════════════════════════
def subsample_first_frac(X_tr, y_tr, frac):
    """클래스별 시간순 앞 frac만 사용 (블록 구조·시간순 보존)."""
    idx = []
    for c in np.unique(y_tr):
        pos = np.where(y_tr == c)[0]
        idx.extend(pos[:max(2, int(len(pos) * frac))].tolist())
    idx = np.array(sorted(idx))
    return X_tr[idx], y_tr[idx]

fracs = [0.2, 0.35, 0.5, 0.65, 0.8, 1.0]
lc_models = {
    'LDA (저분산)':  pipe(('sc', StandardScaler()),
                        ('clf', LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))),
    'QDA (고분산)':  pipe(('sc', StandardScaler()), ('pca', PCA(n_components=15)),
                        ('clf', QuadraticDiscriminantAnalysis(reg_param=0.1))),
    'RandomForest': pipe(('sc', StandardScaler()),
                        ('clf', RandomForestClassifier(n_estimators=300, random_state=0))),
    'GradBoost':    pipe(('sc', StandardScaler()),
                        ('clf', GradientBoostingClassifier(n_estimators=200,
                                learning_rate=0.1, max_depth=3, random_state=0))),
}

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
print("\n[실험 2] 학습 곡선  → H4 (마지막 점 train/test):")
for ax, (name, model) in zip(axes.flatten(), lc_models.items()):
    train_f1, test_f1, sizes = [], [], []
    for frac in fracs:
        Xs, ys = subsample_first_frac(X_tr, y_tr, frac)
        model.fit(Xs, ys)
        train_f1.append(f1_score(ys, model.predict(Xs), average='macro'))
        test_f1.append(f1_score(y_te, model.predict(X_te), average='macro'))
        sizes.append(len(ys))
    ax.plot(sizes, train_f1, 'o-', color='#E24B4A', label='Train')
    ax.plot(sizes, test_f1, 's-', color='#378ADD', label='Test')
    ax.fill_between(sizes, train_f1, test_f1, alpha=0.1, color='gray')
    ax.set_title(name)
    ax.set_xlabel('학습 표본 수')
    ax.set_ylabel('macro-F1')
    ax.set_ylim(0.5, 1.02)
    ax.legend()
    ax.grid(alpha=0.3)
    print(f"  {name}: train {train_f1[-1]:.3f}, test {test_f1[-1]:.3f}, "
          f"격차 {train_f1[-1]-test_f1[-1]:.3f}")
plt.suptitle('실험 2: 학습 곡선 — Train/Test 격차로 편향-분산 확인', y=1.0)
plt.tight_layout()
plt.savefig(_OUT_DIR / 'exp2_learning_curve.png', dpi=150)
plt.close()

# ══════════════════════════════════════════════════════════
# 실험 3: 무작위 분할 vs purged 분할  →  누수 시연
# ══════════════════════════════════════════════════════════
print("\n[실험 3] 무작위 분할 vs purged 분할  → 누수 시연")
leak_models = {
    'KNN':          KNeighborsClassifier(n_neighbors=5),
    'RandomForest': RandomForestClassifier(n_estimators=300, random_state=0),
}
labels, rand_scores, purg_scores = [], [], []
for name, clf in leak_models.items():
    # 무작위 분할 (누수 발생)
    Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(
        X, y, train_size=0.7, shuffle=True, stratify=y, random_state=0)
    m = pipe(('sc', StandardScaler()), ('clf', clf))
    m.fit(Xr_tr, yr_tr)
    rand = f1_score(yr_te, m.predict(Xr_te), average='macro')
    # purged 분할 (정직)
    m2 = pipe(('sc', StandardScaler()), ('clf', clf))
    m2.fit(X_tr, y_tr)
    purg = f1_score(y_te, m2.predict(X_te), average='macro')
    labels.append(name); rand_scores.append(rand); purg_scores.append(purg)
    print(f"  {name}: 무작위 {rand:.4f}  vs  purged {purg:.4f}  "
          f"(부풀림 {rand-purg:+.4f})")

fig, ax = plt.subplots(figsize=(7, 5))
x = np.arange(len(labels)); w = 0.35
ax.bar(x - w/2, rand_scores, w, label='무작위 분할 (누수)', color='#E24B4A', alpha=0.8)
ax.bar(x + w/2, purg_scores, w, label='Purged 분할 (정직)', color='#1D9E75', alpha=0.8)
for i in range(len(labels)):
    ax.text(x[i]-w/2, rand_scores[i]+0.01, f'{rand_scores[i]:.3f}', ha='center', fontsize=9)
    ax.text(x[i]+w/2, purg_scores[i]+0.01, f'{purg_scores[i]:.3f}', ha='center', fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel('Test macro-F1'); ax.set_ylim(0, 1.05)
ax.set_title('실험 3: 무작위 분할의 누수로 인한 성능 부풀림')
ax.legend()
plt.tight_layout()
plt.savefig(_OUT_DIR / 'exp3_leakage.png', dpi=150)
plt.close()

print("\n저장: exp1_qda_pca.png, exp2_learning_curve.png, exp3_leakage.png")
print("=" * 60)