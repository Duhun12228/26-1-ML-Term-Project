"""
코드 ④ — 혼동행렬 + ROC 커브 + 실패 분석
- results_models.joblib (코드 ②) 로드 → 재학습 없이 평가
- 상위 3개 모델 혼동행렬 (true 기준 정규화 = recall 관점)
- 최고 모델 one-vs-rest ROC 커브 + 클래스별/ macro AUC
- 클래스별 precision/recall/F1 + 가장 많이 혼동된 쌍
출력: fig_confusion.png, fig_roc.png + 콘솔
"""

import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib
import warnings, platform, logging
from pathlib import Path
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_curve, roc_auc_score)
from sklearn.preprocessing import label_binarize

warnings.filterwarnings('ignore')
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
_sys = platform.system()
_font = 'AppleGothic' if _sys == 'Darwin' else ('Malgun Gothic' if _sys == 'Windows' else 'DejaVu Sans')
matplotlib.rc('font', family=_font)
matplotlib.rcParams['axes.unicode_minus'] = False

_OUT_DIR = Path(__file__).parent
CLASS_NAMES = {1: '급가속', 2: '급우회전', 3: '급좌회전', 4: '급정거'}
COLORS = ['#E24B4A', '#378ADD', '#1D9E75', '#BA7517']

# ── 로드 ──────────────────────────────────────────────────
data = joblib.load(_OUT_DIR / 'results_models.joblib')
results, y_test, classes = data['results'], data['y_test'], data['classes']
summary = data['summary']
names = [CLASS_NAMES[c] for c in classes]

# 상위 3개 모델 (Baseline 제외)
top = [m for m in summary['Model'] if 'Baseline' not in m][:3]
best = top[0]
print(f"상위 3개 모델: {top}")
print(f"최고 모델: {best}")

# ── 혼동행렬 (상위 3개, recall 정규화) ────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
for ax, name in zip(axes, top):
    yp = results[name]['y_pred']
    cm = confusion_matrix(y_test, yp, labels=classes, normalize='true')
    im = ax.imshow(cm, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(names, rotation=30, fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel('예측'); ax.set_ylabel('실제')
    ax.set_title(name, fontsize=10)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f'{cm[i, j]:.2f}', ha='center', va='center',
                    color='white' if cm[i, j] > 0.5 else 'black', fontsize=9)
plt.suptitle('혼동행렬 (행=실제, 정규화: 행 기준 = recall)', y=1.02)
plt.tight_layout()
plt.savefig(_OUT_DIR / 'fig_confusion.png', dpi=150, bbox_inches='tight')
plt.close()

# ── ROC 커브 (최고 모델, one-vs-rest) ─────────────────────
y_bin = label_binarize(y_test, classes=classes)
proba = results[best]['y_proba']

fig, ax = plt.subplots(figsize=(7, 6))
for i, c in enumerate(classes):
    fpr, tpr, _ = roc_curve(y_bin[:, i], proba[:, i])
    auc_i = roc_auc_score(y_bin[:, i], proba[:, i])
    ax.plot(fpr, tpr, color=COLORS[i], linewidth=2,
            label=f'{CLASS_NAMES[c]} (AUC={auc_i:.3f})')
macro_auc = roc_auc_score(y_bin, proba, average='macro')
ax.plot([0, 1], [0, 1], 'k--', alpha=0.4)
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title(f'{best} — One-vs-Rest ROC (macro AUC={macro_auc:.3f})')
ax.legend(loc='lower right', fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(_OUT_DIR / 'fig_roc.png', dpi=150)
plt.close()

# ── 분류 리포트 + 실패 분석 ───────────────────────────────
print("\n" + "=" * 60)
print(f"[{best}] 클래스별 분류 리포트")
print("=" * 60)
print(classification_report(y_test, results[best]['y_pred'],
                            labels=classes, target_names=names, digits=3))

print("가장 많이 혼동된 쌍 (실제 → 예측):")
cm_cnt = confusion_matrix(y_test, results[best]['y_pred'], labels=classes)
confusions = []
for i in range(len(classes)):
    for j in range(len(classes)):
        if i != j and cm_cnt[i, j] > 0:
            confusions.append((cm_cnt[i, j], classes[i], classes[j]))
for cnt, ci, cj in sorted(confusions, reverse=True)[:5]:
    print(f"  {CLASS_NAMES[ci]} → {CLASS_NAMES[cj]}: {cnt}건")

print("\n저장: fig_confusion.png, fig_roc.png")
print("=" * 60)