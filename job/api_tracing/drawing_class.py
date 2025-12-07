import numpy as np
import matplotlib
matplotlib.use('Agg')  # 远程服务器无显示环境
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib.colors import to_rgb

# ======================
# 读取特征
# ======================
features, labels = [], []
for cls_id in range(22):  # 0~20 seen, 21 unseen
    file = f"/work/xz464/zxp/new_dataset_experiment_dkucc/aasist-main/job/api_tracing/feature/api_tracing/class_{cls_id}.npy"
    try:
        data = np.load(file)
        features.append(data)
        labels.extend([cls_id] * len(data))
    except FileNotFoundError:
        print(f"skip {file}")
        continue

features = np.concatenate(features, axis=0)
labels = np.array(labels)

# ======================
# t-SNE降维
# ======================
tsne = TSNE(n_components=2, init='pca', random_state=0, perplexity=30)
features_2d = tsne.fit_transform(features)

# ======================
# 可视化
# ======================
plt.figure(figsize=(10, 6))  # 高度缩小

# 组合 tab20 + tab20b 颜色
tab20_colors = [to_rgb(plt.cm.tab20(i)) for i in range(20)]
tab20b_colors = [to_rgb(plt.cm.tab20b(i)) for i in range(20)]
all_colors = tab20_colors + tab20b_colors  # 40个颜色

# 分 seen/unseen
seen_mask = labels != 21
unseen_mask = labels == 21

# 绘制 seen 类
plt.scatter(features_2d[seen_mask, 0], features_2d[seen_mask, 1],
            c=[all_colors[i] for i in labels[seen_mask]], s=5, marker='o')

# 绘制 unseen 类
plt.scatter(features_2d[unseen_mask, 0], features_2d[unseen_mask, 1],
            c='k', s=5, marker='^')

# 去掉坐标
plt.xticks([])
plt.yticks([])

# ======================
# 图例
# ======================
# seen 类图例
handles = [plt.Line2D([0], [0], marker='o', color='w',
                      markerfacecolor=all_colors[i], markersize=6, linestyle='None')
           for i in range(21)]

# unseen 类图例
handles.append(plt.Line2D([0], [0], marker='^', color='k',
                          markerfacecolor='k', markersize=6, linestyle='None'))

# 标签
legend_labels = [f"A{i}" for i in range(21)] + ["unseen"]

# 创建图例，分两列显示，字体放大
plt.legend(handles, legend_labels, title="API",bbox_to_anchor=(1.05, 0.5),
           loc='center left', ncol=2, fontsize=14, title_fontsize=14, frameon=False)

plt.tight_layout()
plt.savefig("/work/xz464/zxp/new_dataset_experiment_dkucc/aasist-main/job/api_tracing/drawing/tsne_class.png", dpi=300)
print("t-SNE figure saved successfully.")
