import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE
import numpy as np
import seaborn as sns
from scipy.stats import ttest_ind


data = pd.read_csv("data/插层率影响.csv")

# 各属性随插层率变化散点图
# columns = ['厚度', '孔隙率', '压缩回弹性率', '过滤阻力', '过滤效率', '透气性']
# for col in columns:
#     fig = plt.figure(figsize=(10, 10))
#     plt.plot(data["插层率"], data[col], "r.")
#     plt.savefig(f"results/question1/插层率-{col}散点图.png")

# spearman correlation
# corr = data.corr(method="spearman")
# sns.heatmap(corr, annot=True, fmt=".2f")
# plt.savefig(f"results/question1/spearman相关性热力图.png")
# corr.to_csv("results/question1/Spearman相关性.csv")
