import random
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')


def train_test(data_x, data_y, embeddings, TR, seed):
    # 类别二值化
    data_y = MultiLabelBinarizer().fit_transform(data_y)
    train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=TR, random_state=seed)
    train_x = np.array([embeddings[c] for c in train_x])
    test_x = np.array([embeddings[c] for c in test_x])
    # 下面使用逻辑回归进行点分类
    clf = OneVsRestClassifier(LogisticRegression())
    clf.fit(train_x, train_y)
    prob_y = clf.predict_proba(test_x)  # 预测每一类的概率
    test_y_class_len = np.sum(test_y, axis=1)
    pred_y = []
    # 处理一下预测的概率结果
    for index, class_len in enumerate(test_y_class_len):
        prob = prob_y[index]
        x = np.argsort(-prob)[:class_len]
        prob[:] = 0
        prob[x] = 1
        pred_y.append(prob)
    pred_y = np.array(pred_y)
    averages = ["micro", "macro", "samples", "weighted"]
    results = {}
    for average in averages:
        results[average] = f1_score(test_y, pred_y, average=average)
    results['acc'] = accuracy_score(test_y, pred_y)
    return results


if __name__ == "__main__":
    # 载入已经训练好的模型
    dataset = "wiki"
    model = Word2Vec.load(f"../Node2Vec/{dataset}_0.25_4.model")
    data_x, data_y = [], []
    with open(f"../data/{dataset}/{dataset}_labels.txt", mode='r') as f:
        data = defaultdict(list)
        for line in f.readlines():
            x, y = line.split(" ")
            data[x].append(int(y))
        for k, v in data.items():
            data_x.append(k)
            data_y.append(v)
    embeddings = {}
    for u in data_x:
        embeddings[u] = model.wv[u]

    micros, macros = 0.0, 0.0
    for s in tqdm(range(10)):
        ans = train_test(data_x, data_y, embeddings, 0.2, s)
        micros += ans['micro']
        macros += ans['macro']
    print(micros / 10, macros / 10)
