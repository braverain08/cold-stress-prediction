import matplotlib.pyplot as plt
import numpy as np

# 数据
subsets = ['May-Jun', 'Jul-Aug', 'Sep-Oct']
recall = [0.0, 0.0, 1.000]
precision = [0.0, 0.0, 0.528]
sample_sizes = [61, 62, 58]
cold_ratios = [0.0, 0.0, 0.328]

x = np.arange(len(subsets))
width = 0.35

color_recall = '#E69F00'   # 橙色
color_precision = '#0072B2'  # 蓝色

fig, ax = plt.subplots(figsize=(6, 5))
bars1 = ax.bar(x - width/2, recall, width, label='Recall', color=color_recall, alpha=0.8)
bars2 = ax.bar(x + width/2, precision, width, label='Precision', color=color_precision, alpha=0.8)

# 柱顶标签
for bar in bars1:
    height = bar.get_height()
    if height > 0:
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02, f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    else:
        ax.text(bar.get_x() + bar.get_width()/2., 0.02, 'N/A', ha='center', va='bottom', fontsize=8, color='gray')

for bar in bars2:
    height = bar.get_height()
    if height > 0:
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02, f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    else:
        ax.text(bar.get_x() + bar.get_width()/2., 0.02, 'N/A', ha='center', va='bottom', fontsize=8, color='gray')

ax.axhline(y=1.0, color='black', linestyle='--', linewidth=0.8, alpha=0.7)

ax.set_ylabel('Score')
ax.set_xlabel('Temporal Subset (2025)')
ax.set_title('Recall and Precision on Different Seasonal Subsets')
ax.set_xticks(x)
ax.set_xticklabels(subsets)
ax.set_ylim(0, 1.1)

# 文本框放在左上角
info_text = (f"May-Jun: n={sample_sizes[0]}, cold={cold_ratios[0]*100:.0f}%\n"
             f"Jul-Aug: n={sample_sizes[1]}, cold={cold_ratios[1]*100:.0f}%\n"
             f"Sep-Oct: n={sample_sizes[2]}, cold={cold_ratios[2]*100:.1f}%")
ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
        fontsize=8, verticalalignment='top', horizontalalignment='left',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# 图例放在左侧中部偏下（避免与文本框重叠）
ax.legend(loc='center left', bbox_to_anchor=(0.02, 0.4), frameon=True, framealpha=0.9)

plt.tight_layout()
plt.savefig('figure5_temporal_generalization.png', dpi=300)
plt.show()