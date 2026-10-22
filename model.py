"""
Unsupervised Learning: Segment, Detect, Propagate with Scikit-Learn

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - blobs_data
import os
import joblib
import numpy as np

from sklearn.datasets import make_blobs, make_moons, load_digits
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, accuracy_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# ── 第 1 部分：K-Means 与聚类数选择 ──────────────────────────

# Step 1
def blobs_data(random_state=42):
    centers = [
        [0.2, 2.3],
        [-1.5, 2.3],
        [-2.8, 1.8],
        [-2.8, 2.8],
        [-2.8, 1.3],
    ]
    X, y = make_blobs(
        n_samples=2000,
        centers=centers,
        cluster_std=[0.4, 0.3, 0.1, 0.1, 0.1],
        random_state=random_state,
    )
    return X, y


# Step 2
def fit_kmeans(X, k, random_state=42):
    model = KMeans(
        n_clusters=k,
        n_init=10,
        random_state=random_state,
    )
    return model.fit(X)


# Step 3
def inertia_curve(X, ks):
    return {
        k: float(fit_kmeans(X, k).inertia_)
        for k in ks
    }


# Step 4
def silhouette_curve(X, ks):
    scores = {}

    for k in ks:
        model = fit_kmeans(X, k)
        scores[k] = float(silhouette_score(X, model.labels_))

    return scores


def best_k_by_silhouette(curve):
    # 先按 silhouette 从高到低；若分数相同，选较小的 k
    return min(curve, key=lambda k: (-curve[k], k))


# ── 第 2 部分：DBSCAN、GMM 与异常检测 ─────────────────────────

# Step 5
def fit_dbscan(X, eps=0.2, min_samples=5):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    return model.fit(X)


def dbscan_summary(dbscan):
    labels = dbscan.labels_
    clusters = set(labels.tolist()) - {-1}

    return {
        "n_clusters": int(len(clusters)),
        "n_noise": int(np.sum(labels == -1)),
        "n_core": int(len(dbscan.core_sample_indices_)),
    }


# Step 6
def dbscan_predict(dbscan, X_new, n_neighbors=50):
    core_X = dbscan.components_
    core_labels = dbscan.labels_[dbscan.core_sample_indices_]

    # 如果没有核心点，就把新样本都标成噪声
    if len(core_X) == 0:
        return np.full(len(X_new), -1, dtype=int)

    # 核心点不足 n_neighbors 个时，使用实际可用的数量
    k = min(n_neighbors, len(core_X))

    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(core_X, core_labels)
    return model.predict(X_new).astype(int)


# Step 7
def fit_gmm(X, n_components, random_state=42):
    model = GaussianMixture(
        n_components=n_components,
        n_init=10,
        random_state=random_state,
    )
    return model.fit(X)


def bic_curve(X, ks):
    scores = {}

    for k in ks:
        model = fit_gmm(X, n_components=k)
        scores[k] = float(model.bic(X))

    return scores


# Step 8
def flag_anomalies(gmm, X, contamination=0.04):
    # score_samples 越低，说明样本在模型下越不常见
    densities = gmm.score_samples(X)
    threshold = np.percentile(densities, 100 * contamination)

    return densities < threshold


# ── 第 3 部分：用少量标签学习手写数字 ─────────────────────────

# Step 9
def digits_data(test_size=0.25, random_state=42):
    digits = load_digits()
    X = digits.data
    y = digits.target

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


# Step 10
def baseline_50_random(
    X_train, y_train, X_test, y_test,
    n_labeled=50, random_state=42
):
    # 题目说明：训练集已打乱，所以直接拿前 n_labeled 个样本
    model = LogisticRegression(max_iter=10000)
    model.fit(X_train[:n_labeled], y_train[:n_labeled])

    return float(accuracy_score(y_test, model.predict(X_test)))


# Step 11
def representative_digits(X_train, k=50, random_state=42):
    kmeans = fit_kmeans(X_train, k, random_state=random_state)

    # transform 的形状是 (样本数, 聚类数)
    # 第 j 列是所有样本到第 j 个中心的距离
    distances = kmeans.transform(X_train)

    # 每个聚类里挑离该聚类中心最近的训练样本
    rep_idx = np.argmin(distances, axis=0).astype(int)

    return kmeans, rep_idx


# Step 12
def train_on_representatives(
    X_train, y_train, rep_idx, X_test, y_test
):
    model = LogisticRegression(max_iter=10000)
    model.fit(X_train[rep_idx], y_train[rep_idx])

    return float(accuracy_score(y_test, model.predict(X_test)))


# Step 13
def propagate_and_train(
    X_train, y_train, kmeans, rep_idx,
    X_test, y_test, percentile=20
):
    distances = kmeans.transform(X_train)
    cluster_ids = kmeans.labels_
    n_clusters = kmeans.n_clusters

    selected = np.zeros(len(X_train), dtype=bool)
    propagated_labels = np.empty(len(X_train), dtype=int)

    for j in range(n_clusters):
        in_cluster = cluster_ids == j

        # 每个样本到“自己所属聚类中心”的距离
        cluster_distances = distances[in_cluster, j]

        # 只传播到离中心较近的样本
        cutoff = np.percentile(cluster_distances, percentile)
        selected[in_cluster] = cluster_distances <= cutoff

        # 用该聚类代表样本的真实标签作为传播标签
        propagated_labels[in_cluster] = y_train[rep_idx[j]]

    X_selected = X_train[selected]
    y_selected = propagated_labels[selected]

    model = LogisticRegression(max_iter=10000)
    model.fit(X_selected, y_selected)

    label_accuracy = float(np.mean(y_selected == y_train[selected]))
    test_accuracy = float(
        accuracy_score(y_test, model.predict(X_test))
    )

    return {
        "n_propagated": int(selected.sum()),
        "label_accuracy": label_accuracy,
        "test_accuracy": test_accuracy,
    }


# ── 第 4 部分：图像分割与模型保存 ─────────────────────────────

# Step 14
def synthetic_image(size=48):
    image = np.zeros((size, size, 3), dtype=float)

    midpoint = size // 2

    # 左半边：红色 R=1，绿色从 0 渐变到 1，蓝色 B=0
    left_gradient = np.linspace(0.0, 1.0, midpoint)
    image[:, :midpoint, 0] = 1.0
    image[:, :midpoint, 1] = left_gradient
    image[:, :midpoint, 2] = 0.0

    # 右半边：红色 R=0，绿色从 0 渐变到 1，蓝色 B=1
    right_width = size - midpoint
    right_gradient = np.linspace(0.0, 1.0, right_width)
    image[:, midpoint:, 0] = 0.0
    image[:, midpoint:, 1] = right_gradient
    image[:, midpoint:, 2] = 1.0

    # 左上角四分之一大小的方块涂成白色
    quarter = size // 4
    image[:quarter, :quarter] = 1.0

    return image


# Step 15
def segment_colors(image, k=4, random_state=42):
    height, width, channels = image.shape
    pixels = image.reshape(-1, channels)

    kmeans = fit_kmeans(pixels, k, random_state=random_state)

    # 每个像素替换成它所属聚类中心的 RGB 颜色
    segmented_pixels = kmeans.cluster_centers_[kmeans.labels_]

    return segmented_pixels.reshape(height, width, channels)


# Step 16
def save_and_reload_clusterer(kmeans, rep_labels, path):
    bundle = {
        "kmeans": kmeans,
        "rep_labels": np.asarray(rep_labels),
    }

    joblib.dump(bundle, path)
    return joblib.load(path)


# Step 17
def predict_digit_labels(bundle, images):
    # 原始图片是 (n, 8, 8)，展平后是 (n, 64)
    X = np.asarray(images).reshape(len(images), -1)

    cluster_ids = bundle["kmeans"].predict(X)
    labels = bundle["rep_labels"][cluster_ids]

    # 返回 Python 的整数列表
    return labels.astype(int).tolist()

# Step 2 - fit_kmeans
import os
import joblib
import numpy as np

from sklearn.datasets import make_blobs, make_moons, load_digits
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, accuracy_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# ── 第 1 部分：K-Means 与聚类数选择 ──────────────────────────

# Step 1
def blobs_data(random_state=42):
    centers = [
        [0.2, 2.3],
        [-1.5, 2.3],
        [-2.8, 1.8],
        [-2.8, 2.8],
        [-2.8, 1.3],
    ]
    X, y = make_blobs(
        n_samples=2000,
        centers=centers,
        cluster_std=[0.4, 0.3, 0.1, 0.1, 0.1],
        random_state=random_state,
    )
    return X, y


# Step 2
def fit_kmeans(X, k, random_state=42):
    model = KMeans(
        n_clusters=k,
        n_init=10,
        random_state=random_state,
    )
    return model.fit(X)


# Step 3
def inertia_curve(X, ks):
    return {
        k: float(fit_kmeans(X, k).inertia_)
        for k in ks
    }


# Step 4
def silhouette_curve(X, ks):
    scores = {}

    for k in ks:
        model = fit_kmeans(X, k)
        scores[k] = float(silhouette_score(X, model.labels_))

    return scores


def best_k_by_silhouette(curve):
    # 先按 silhouette 从高到低；若分数相同，选较小的 k
    return min(curve, key=lambda k: (-curve[k], k))


# ── 第 2 部分：DBSCAN、GMM 与异常检测 ─────────────────────────

# Step 5
def fit_dbscan(X, eps=0.2, min_samples=5):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    return model.fit(X)


def dbscan_summary(dbscan):
    labels = dbscan.labels_
    clusters = set(labels.tolist()) - {-1}

    return {
        "n_clusters": int(len(clusters)),
        "n_noise": int(np.sum(labels == -1)),
        "n_core": int(len(dbscan.core_sample_indices_)),
    }


# Step 6
def dbscan_predict(dbscan, X_new, n_neighbors=50):
    core_X = dbscan.components_
    core_labels = dbscan.labels_[dbscan.core_sample_indices_]

    # 如果没有核心点，就把新样本都标成噪声
    if len(core_X) == 0:
        return np.full(len(X_new), -1, dtype=int)

    # 核心点不足 n_neighbors 个时，使用实际可用的数量
    k = min(n_neighbors, len(core_X))

    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(core_X, core_labels)
    return model.predict(X_new).astype(int)


# Step 7
def fit_gmm(X, n_components, random_state=42):
    model = GaussianMixture(
        n_components=n_components,
        n_init=10,
        random_state=random_state,
    )
    return model.fit(X)


def bic_curve(X, ks):
    scores = {}

    for k in ks:
        model = fit_gmm(X, n_components=k)
        scores[k] = float(model.bic(X))

    return scores


# Step 8
def flag_anomalies(gmm, X, contamination=0.04):
    # score_samples 越低，说明样本在模型下越不常见
    densities = gmm.score_samples(X)
    threshold = np.percentile(densities, 100 * contamination)

    return densities < threshold


# ── 第 3 部分：用少量标签学习手写数字 ─────────────────────────

# Step 9
def digits_data(test_size=0.25, random_state=42):
    digits = load_digits()
    X = digits.data
    y = digits.target

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


# Step 10
def baseline_50_random(
    X_train, y_train, X_test, y_test,
    n_labeled=50, random_state=42
):
    # 题目说明：训练集已打乱，所以直接拿前 n_labeled 个样本
    model = LogisticRegression(max_iter=10000)
    model.fit(X_train[:n_labeled], y_train[:n_labeled])

    return float(accuracy_score(y_test, model.predict(X_test)))


# Step 11
def representative_digits(X_train, k=50, random_state=42):
    kmeans = fit_kmeans(X_train, k, random_state=random_state)

    # transform 的形状是 (样本数, 聚类数)
    # 第 j 列是所有样本到第 j 个中心的距离
    distances = kmeans.transform(X_train)

    # 每个聚类里挑离该聚类中心最近的训练样本
    rep_idx = np.argmin(distances, axis=0).astype(int)

    return kmeans, rep_idx


# Step 12
def train_on_representatives(
    X_train, y_train, rep_idx, X_test, y_test
):
    model = LogisticRegression(max_iter=10000)
    model.fit(X_train[rep_idx], y_train[rep_idx])

    return float(accuracy_score(y_test, model.predict(X_test)))


# Step 13
def propagate_and_train(
    X_train, y_train, kmeans, rep_idx,
    X_test, y_test, percentile=20
):
    distances = kmeans.transform(X_train)
    cluster_ids = kmeans.labels_
    n_clusters = kmeans.n_clusters

    selected = np.zeros(len(X_train), dtype=bool)
    propagated_labels = np.empty(len(X_train), dtype=int)

    for j in range(n_clusters):
        in_cluster = cluster_ids == j

        # 每个样本到“自己所属聚类中心”的距离
        cluster_distances = distances[in_cluster, j]

        # 只传播到离中心较近的样本
        cutoff = np.percentile(cluster_distances, percentile)
        selected[in_cluster] = cluster_distances <= cutoff

        # 用该聚类代表样本的真实标签作为传播标签
        propagated_labels[in_cluster] = y_train[rep_idx[j]]

    X_selected = X_train[selected]
    y_selected = propagated_labels[selected]

    model = LogisticRegression(max_iter=10000)
    model.fit(X_selected, y_selected)

    label_accuracy = float(np.mean(y_selected == y_train[selected]))
    test_accuracy = float(
        accuracy_score(y_test, model.predict(X_test))
    )

    return {
        "n_propagated": int(selected.sum()),
        "label_accuracy": label_accuracy,
        "test_accuracy": test_accuracy,
    }


# ── 第 4 部分：图像分割与模型保存 ─────────────────────────────

# Step 14
def synthetic_image(size=48):
    image = np.zeros((size, size, 3), dtype=float)

    midpoint = size // 2

    # 左半边：红色 R=1，绿色从 0 渐变到 1，蓝色 B=0
    left_gradient = np.linspace(0.0, 1.0, midpoint)
    image[:, :midpoint, 0] = 1.0
    image[:, :midpoint, 1] = left_gradient
    image[:, :midpoint, 2] = 0.0

    # 右半边：红色 R=0，绿色从 0 渐变到 1，蓝色 B=1
    right_width = size - midpoint
    right_gradient = np.linspace(0.0, 1.0, right_width)
    image[:, midpoint:, 0] = 0.0
    image[:, midpoint:, 1] = right_gradient
    image[:, midpoint:, 2] = 1.0

    # 左上角四分之一大小的方块涂成白色
    quarter = size // 4
    image[:quarter, :quarter] = 1.0

    return image


# Step 15
def segment_colors(image, k=4, random_state=42):
    height, width, channels = image.shape
    pixels = image.reshape(-1, channels)

    kmeans = fit_kmeans(pixels, k, random_state=random_state)

    # 每个像素替换成它所属聚类中心的 RGB 颜色
    segmented_pixels = kmeans.cluster_centers_[kmeans.labels_]

    return segmented_pixels.reshape(height, width, channels)


# Step 16
def save_and_reload_clusterer(kmeans, rep_labels, path):
    bundle = {
        "kmeans": kmeans,
        "rep_labels": np.asarray(rep_labels),
    }

    joblib.dump(bundle, path)
    return joblib.load(path)


# Step 17
def predict_digit_labels(bundle, images):
    # 原始图片是 (n, 8, 8)，展平后是 (n, 64)
    X = np.asarray(images).reshape(len(images), -1)

    cluster_ids = bundle["kmeans"].predict(X)
    labels = bundle["rep_labels"][cluster_ids]

    # 返回 Python 的整数列表
    return labels.astype(int).tolist()

# Step 3 - inertia_curve
import os
import joblib
import numpy as np

from sklearn.datasets import make_blobs, make_moons, load_digits
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, accuracy_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# ── 第 1 部分：K-Means 与聚类数选择 ──────────────────────────

# Step 1
def blobs_data(random_state=42):
    centers = [
        [0.2, 2.3],
        [-1.5, 2.3],
        [-2.8, 1.8],
        [-2.8, 2.8],
        [-2.8, 1.3],
    ]
    X, y = make_blobs(
        n_samples=2000,
        centers=centers,
        cluster_std=[0.4, 0.3, 0.1, 0.1, 0.1],
        random_state=random_state,
    )
    return X, y


# Step 2
def fit_kmeans(X, k, random_state=42):
    model = KMeans(
        n_clusters=k,
        n_init=10,
        random_state=random_state,
    )
    return model.fit(X)


# Step 3
def inertia_curve(X, ks):
    return {
        k: float(fit_kmeans(X, k).inertia_)
        for k in ks
    }


# Step 4
def silhouette_curve(X, ks):
    scores = {}

    for k in ks:
        model = fit_kmeans(X, k)
        scores[k] = float(silhouette_score(X, model.labels_))

    return scores


def best_k_by_silhouette(curve):
    # 先按 silhouette 从高到低；若分数相同，选较小的 k
    return min(curve, key=lambda k: (-curve[k], k))


# ── 第 2 部分：DBSCAN、GMM 与异常检测 ─────────────────────────

# Step 5
def fit_dbscan(X, eps=0.2, min_samples=5):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    return model.fit(X)


def dbscan_summary(dbscan):
    labels = dbscan.labels_
    clusters = set(labels.tolist()) - {-1}

    return {
        "n_clusters": int(len(clusters)),
        "n_noise": int(np.sum(labels == -1)),
        "n_core": int(len(dbscan.core_sample_indices_)),
    }


# Step 6
def dbscan_predict(dbscan, X_new, n_neighbors=50):
    core_X = dbscan.components_
    core_labels = dbscan.labels_[dbscan.core_sample_indices_]

    # 如果没有核心点，就把新样本都标成噪声
    if len(core_X) == 0:
        return np.full(len(X_new), -1, dtype=int)

    # 核心点不足 n_neighbors 个时，使用实际可用的数量
    k = min(n_neighbors, len(core_X))

    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(core_X, core_labels)
    return model.predict(X_new).astype(int)


# Step 7
def fit_gmm(X, n_components, random_state=42):
    model = GaussianMixture(
        n_components=n_components,
        n_init=10,
        random_state=random_state,
    )
    return model.fit(X)


def bic_curve(X, ks):
    scores = {}

    for k in ks:
        model = fit_gmm(X, n_components=k)
        scores[k] = float(model.bic(X))

    return scores


# Step 8
def flag_anomalies(gmm, X, contamination=0.04):
    # score_samples 越低，说明样本在模型下越不常见
    densities = gmm.score_samples(X)
    threshold = np.percentile(densities, 100 * contamination)

    return densities < threshold


# ── 第 3 部分：用少量标签学习手写数字 ─────────────────────────

# Step 9
def digits_data(test_size=0.25, random_state=42):
    digits = load_digits()
    X = digits.data
    y = digits.target

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


# Step 10
def baseline_50_random(
    X_train, y_train, X_test, y_test,
    n_labeled=50, random_state=42
):
    # 题目说明：训练集已打乱，所以直接拿前 n_labeled 个样本
    model = LogisticRegression(max_iter=10000)
    model.fit(X_train[:n_labeled], y_train[:n_labeled])

    return float(accuracy_score(y_test, model.predict(X_test)))


# Step 11
def representative_digits(X_train, k=50, random_state=42):
    kmeans = fit_kmeans(X_train, k, random_state=random_state)

    # transform 的形状是 (样本数, 聚类数)
    # 第 j 列是所有样本到第 j 个中心的距离
    distances = kmeans.transform(X_train)

    # 每个聚类里挑离该聚类中心最近的训练样本
    rep_idx = np.argmin(distances, axis=0).astype(int)

    return kmeans, rep_idx


# Step 12
def train_on_representatives(
    X_train, y_train, rep_idx, X_test, y_test
):
    model = LogisticRegression(max_iter=10000)
    model.fit(X_train[rep_idx], y_train[rep_idx])

    return float(accuracy_score(y_test, model.predict(X_test)))


# Step 13
def propagate_and_train(
    X_train, y_train, kmeans, rep_idx,
    X_test, y_test, percentile=20
):
    distances = kmeans.transform(X_train)
    cluster_ids = kmeans.labels_
    n_clusters = kmeans.n_clusters

    selected = np.zeros(len(X_train), dtype=bool)
    propagated_labels = np.empty(len(X_train), dtype=int)

    for j in range(n_clusters):
        in_cluster = cluster_ids == j

        # 每个样本到“自己所属聚类中心”的距离
        cluster_distances = distances[in_cluster, j]

        # 只传播到离中心较近的样本
        cutoff = np.percentile(cluster_distances, percentile)
        selected[in_cluster] = cluster_distances <= cutoff

        # 用该聚类代表样本的真实标签作为传播标签
        propagated_labels[in_cluster] = y_train[rep_idx[j]]

    X_selected = X_train[selected]
    y_selected = propagated_labels[selected]

    model = LogisticRegression(max_iter=10000)
    model.fit(X_selected, y_selected)

    label_accuracy = float(np.mean(y_selected == y_train[selected]))
    test_accuracy = float(
        accuracy_score(y_test, model.predict(X_test))
    )

    return {
        "n_propagated": int(selected.sum()),
        "label_accuracy": label_accuracy,
        "test_accuracy": test_accuracy,
    }


# ── 第 4 部分：图像分割与模型保存 ─────────────────────────────

# Step 14
def synthetic_image(size=48):
    image = np.zeros((size, size, 3), dtype=float)

    midpoint = size // 2

    # 左半边：红色 R=1，绿色从 0 渐变到 1，蓝色 B=0
    left_gradient = np.linspace(0.0, 1.0, midpoint)
    image[:, :midpoint, 0] = 1.0
    image[:, :midpoint, 1] = left_gradient
    image[:, :midpoint, 2] = 0.0

    # 右半边：红色 R=0，绿色从 0 渐变到 1，蓝色 B=1
    right_width = size - midpoint
    right_gradient = np.linspace(0.0, 1.0, right_width)
    image[:, midpoint:, 0] = 0.0
    image[:, midpoint:, 1] = right_gradient
    image[:, midpoint:, 2] = 1.0

    # 左上角四分之一大小的方块涂成白色
    quarter = size // 4
    image[:quarter, :quarter] = 1.0

    return image


# Step 15
def segment_colors(image, k=4, random_state=42):
    height, width, channels = image.shape
    pixels = image.reshape(-1, channels)

    kmeans = fit_kmeans(pixels, k, random_state=random_state)

    # 每个像素替换成它所属聚类中心的 RGB 颜色
    segmented_pixels = kmeans.cluster_centers_[kmeans.labels_]

    return segmented_pixels.reshape(height, width, channels)


# Step 16
def save_and_reload_clusterer(kmeans, rep_labels, path):
    bundle = {
        "kmeans": kmeans,
        "rep_labels": np.asarray(rep_labels),
    }

    joblib.dump(bundle, path)
    return joblib.load(path)


# Step 17
def predict_digit_labels(bundle, images):
    # 原始图片是 (n, 8, 8)，展平后是 (n, 64)
    X = np.asarray(images).reshape(len(images), -1)

    cluster_ids = bundle["kmeans"].predict(X)
    labels = bundle["rep_labels"][cluster_ids]

    # 返回 Python 的整数列表
    return labels.astype(int).tolist()

# Step 4 - silhouette_curve
import os
import joblib
import numpy as np

from sklearn.datasets import make_blobs, make_moons, load_digits
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, accuracy_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# ── 第 1 部分：K-Means 与聚类数选择 ──────────────────────────

# Step 1
def blobs_data(random_state=42):
    centers = [
        [0.2, 2.3],
        [-1.5, 2.3],
        [-2.8, 1.8],
        [-2.8, 2.8],
        [-2.8, 1.3],
    ]
    X, y = make_blobs(
        n_samples=2000,
        centers=centers,
        cluster_std=[0.4, 0.3, 0.1, 0.1, 0.1],
        random_state=random_state,
    )
    return X, y


# Step 2
def fit_kmeans(X, k, random_state=42):
    model = KMeans(
        n_clusters=k,
        n_init=10,
        random_state=random_state,
    )
    return model.fit(X)


# Step 3
def inertia_curve(X, ks):
    return {
        k: float(fit_kmeans(X, k).inertia_)
        for k in ks
    }


# Step 4
def silhouette_curve(X, ks):
    scores = {}

    for k in ks:
        model = fit_kmeans(X, k)
        scores[k] = float(silhouette_score(X, model.labels_))

    return scores


def best_k_by_silhouette(curve):
    # 先按 silhouette 从高到低；若分数相同，选较小的 k
    return min(curve, key=lambda k: (-curve[k], k))


# ── 第 2 部分：DBSCAN、GMM 与异常检测 ─────────────────────────

# Step 5
def fit_dbscan(X, eps=0.2, min_samples=5):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    return model.fit(X)


def dbscan_summary(dbscan):
    labels = dbscan.labels_
    clusters = set(labels.tolist()) - {-1}

    return {
        "n_clusters": int(len(clusters)),
        "n_noise": int(np.sum(labels == -1)),
        "n_core": int(len(dbscan.core_sample_indices_)),
    }


# Step 6
def dbscan_predict(dbscan, X_new, n_neighbors=50):
    core_X = dbscan.components_
    core_labels = dbscan.labels_[dbscan.core_sample_indices_]

    # 如果没有核心点，就把新样本都标成噪声
    if len(core_X) == 0:
        return np.full(len(X_new), -1, dtype=int)

    # 核心点不足 n_neighbors 个时，使用实际可用的数量
    k = min(n_neighbors, len(core_X))

    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(core_X, core_labels)
    return model.predict(X_new).astype(int)


# Step 7
def fit_gmm(X, n_components, random_state=42):
    model = GaussianMixture(
        n_components=n_components,
        n_init=10,
        random_state=random_state,
    )
    return model.fit(X)


def bic_curve(X, ks):
    scores = {}

    for k in ks:
        model = fit_gmm(X, n_components=k)
        scores[k] = float(model.bic(X))

    return scores


# Step 8
def flag_anomalies(gmm, X, contamination=0.04):
    # score_samples 越低，说明样本在模型下越不常见
    densities = gmm.score_samples(X)
    threshold = np.percentile(densities, 100 * contamination)

    return densities < threshold


# ── 第 3 部分：用少量标签学习手写数字 ─────────────────────────

# Step 9
def digits_data(test_size=0.25, random_state=42):
    digits = load_digits()
    X = digits.data
    y = digits.target

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


# Step 10
def baseline_50_random(
    X_train, y_train, X_test, y_test,
    n_labeled=50, random_state=42
):
    # 题目说明：训练集已打乱，所以直接拿前 n_labeled 个样本
    model = LogisticRegression(max_iter=10000)
    model.fit(X_train[:n_labeled], y_train[:n_labeled])

    return float(accuracy_score(y_test, model.predict(X_test)))


# Step 11
def representative_digits(X_train, k=50, random_state=42):
    kmeans = fit_kmeans(X_train, k, random_state=random_state)

    # transform 的形状是 (样本数, 聚类数)
    # 第 j 列是所有样本到第 j 个中心的距离
    distances = kmeans.transform(X_train)

    # 每个聚类里挑离该聚类中心最近的训练样本
    rep_idx = np.argmin(distances, axis=0).astype(int)

    return kmeans, rep_idx


# Step 12
def train_on_representatives(
    X_train, y_train, rep_idx, X_test, y_test
):
    model = LogisticRegression(max_iter=10000)
    model.fit(X_train[rep_idx], y_train[rep_idx])

    return float(accuracy_score(y_test, model.predict(X_test)))


# Step 13
def propagate_and_train(
    X_train, y_train, kmeans, rep_idx,
    X_test, y_test, percentile=20
):
    distances = kmeans.transform(X_train)
    cluster_ids = kmeans.labels_
    n_clusters = kmeans.n_clusters

    selected = np.zeros(len(X_train), dtype=bool)
    propagated_labels = np.empty(len(X_train), dtype=int)

    for j in range(n_clusters):
        in_cluster = cluster_ids == j

        # 每个样本到“自己所属聚类中心”的距离
        cluster_distances = distances[in_cluster, j]

        # 只传播到离中心较近的样本
        cutoff = np.percentile(cluster_distances, percentile)
        selected[in_cluster] = cluster_distances <= cutoff

        # 用该聚类代表样本的真实标签作为传播标签
        propagated_labels[in_cluster] = y_train[rep_idx[j]]

    X_selected = X_train[selected]
    y_selected = propagated_labels[selected]

    model = LogisticRegression(max_iter=10000)
    model.fit(X_selected, y_selected)

    label_accuracy = float(np.mean(y_selected == y_train[selected]))
    test_accuracy = float(
        accuracy_score(y_test, model.predict(X_test))
    )

    return {
        "n_propagated": int(selected.sum()),
        "label_accuracy": label_accuracy,
        "test_accuracy": test_accuracy,
    }


# ── 第 4 部分：图像分割与模型保存 ─────────────────────────────

# Step 14
def synthetic_image(size=48):
    image = np.zeros((size, size, 3), dtype=float)

    midpoint = size // 2

    # 左半边：红色 R=1，绿色从 0 渐变到 1，蓝色 B=0
    left_gradient = np.linspace(0.0, 1.0, midpoint)
    image[:, :midpoint, 0] = 1.0
    image[:, :midpoint, 1] = left_gradient
    image[:, :midpoint, 2] = 0.0

    # 右半边：红色 R=0，绿色从 0 渐变到 1，蓝色 B=1
    right_width = size - midpoint
    right_gradient = np.linspace(0.0, 1.0, right_width)
    image[:, midpoint:, 0] = 0.0
    image[:, midpoint:, 1] = right_gradient
    image[:, midpoint:, 2] = 1.0

    # 左上角四分之一大小的方块涂成白色
    quarter = size // 4
    image[:quarter, :quarter] = 1.0

    return image


# Step 15
def segment_colors(image, k=4, random_state=42):
    height, width, channels = image.shape
    pixels = image.reshape(-1, channels)

    kmeans = fit_kmeans(pixels, k, random_state=random_state)

    # 每个像素替换成它所属聚类中心的 RGB 颜色
    segmented_pixels = kmeans.cluster_centers_[kmeans.labels_]

    return segmented_pixels.reshape(height, width, channels)


# Step 16
def save_and_reload_clusterer(kmeans, rep_labels, path):
    bundle = {
        "kmeans": kmeans,
        "rep_labels": np.asarray(rep_labels),
    }

    joblib.dump(bundle, path)
    return joblib.load(path)


# Step 17
def predict_digit_labels(bundle, images):
    # 原始图片是 (n, 8, 8)，展平后是 (n, 64)
    X = np.asarray(images).reshape(len(images), -1)

    cluster_ids = bundle["kmeans"].predict(X)
    labels = bundle["rep_labels"][cluster_ids]

    # 返回 Python 的整数列表
    return labels.astype(int).tolist()

# Step 5 - fit_dbscan
import os
import joblib
import numpy as np

from sklearn.datasets import make_blobs, make_moons, load_digits
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, accuracy_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# ── 第 1 部分：K-Means 与聚类数选择 ──────────────────────────

# Step 1
def blobs_data(random_state=42):
    centers = [
        [0.2, 2.3],
        [-1.5, 2.3],
        [-2.8, 1.8],
        [-2.8, 2.8],
        [-2.8, 1.3],
    ]
    X, y = make_blobs(
        n_samples=2000,
        centers=centers,
        cluster_std=[0.4, 0.3, 0.1, 0.1, 0.1],
        random_state=random_state,
    )
    return X, y


# Step 2
def fit_kmeans(X, k, random_state=42):
    model = KMeans(
        n_clusters=k,
        n_init=10,
        random_state=random_state,
    )
    return model.fit(X)


# Step 3
def inertia_curve(X, ks):
    return {
        k: float(fit_kmeans(X, k).inertia_)
        for k in ks
    }


# Step 4
def silhouette_curve(X, ks):
    scores = {}

    for k in ks:
        model = fit_kmeans(X, k)
        scores[k] = float(silhouette_score(X, model.labels_))

    return scores


def best_k_by_silhouette(curve):
    # 先按 silhouette 从高到低；若分数相同，选较小的 k
    return min(curve, key=lambda k: (-curve[k], k))


# ── 第 2 部分：DBSCAN、GMM 与异常检测 ─────────────────────────

# Step 5
def fit_dbscan(X, eps=0.2, min_samples=5):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    return model.fit(X)


def dbscan_summary(dbscan):
    labels = dbscan.labels_
    clusters = set(labels.tolist()) - {-1}

    return {
        "n_clusters": int(len(clusters)),
        "n_noise": int(np.sum(labels == -1)),
        "n_core": int(len(dbscan.core_sample_indices_)),
    }


# Step 6
def dbscan_predict(dbscan, X_new, n_neighbors=50):
    core_X = dbscan.components_
    core_labels = dbscan.labels_[dbscan.core_sample_indices_]

    # 如果没有核心点，就把新样本都标成噪声
    if len(core_X) == 0:
        return np.full(len(X_new), -1, dtype=int)

    # 核心点不足 n_neighbors 个时，使用实际可用的数量
    k = min(n_neighbors, len(core_X))

    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(core_X, core_labels)
    return model.predict(X_new).astype(int)


# Step 7
def fit_gmm(X, n_components, random_state=42):
    model = GaussianMixture(
        n_components=n_components,
        n_init=10,
        random_state=random_state,
    )
    return model.fit(X)


def bic_curve(X, ks):
    scores = {}

    for k in ks:
        model = fit_gmm(X, n_components=k)
        scores[k] = float(model.bic(X))

    return scores


# Step 8
def flag_anomalies(gmm, X, contamination=0.04):
    # score_samples 越低，说明样本在模型下越不常见
    densities = gmm.score_samples(X)
    threshold = np.percentile(densities, 100 * contamination)

    return densities < threshold


# ── 第 3 部分：用少量标签学习手写数字 ─────────────────────────

# Step 9
def digits_data(test_size=0.25, random_state=42):
    digits = load_digits()
    X = digits.data
    y = digits.target

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


# Step 10
def baseline_50_random(
    X_train, y_train, X_test, y_test,
    n_labeled=50, random_state=42
):
    # 题目说明：训练集已打乱，所以直接拿前 n_labeled 个样本
    model = LogisticRegression(max_iter=10000)
    model.fit(X_train[:n_labeled], y_train[:n_labeled])

    return float(accuracy_score(y_test, model.predict(X_test)))


# Step 11
def representative_digits(X_train, k=50, random_state=42):
    kmeans = fit_kmeans(X_train, k, random_state=random_state)

    # transform 的形状是 (样本数, 聚类数)
    # 第 j 列是所有样本到第 j 个中心的距离
    distances = kmeans.transform(X_train)

    # 每个聚类里挑离该聚类中心最近的训练样本
    rep_idx = np.argmin(distances, axis=0).astype(int)

    return kmeans, rep_idx


# Step 12
def train_on_representatives(
    X_train, y_train, rep_idx, X_test, y_test
):
    model = LogisticRegression(max_iter=10000)
    model.fit(X_train[rep_idx], y_train[rep_idx])

    return float(accuracy_score(y_test, model.predict(X_test)))


# Step 13
def propagate_and_train(
    X_train, y_train, kmeans, rep_idx,
    X_test, y_test, percentile=20
):
    distances = kmeans.transform(X_train)
    cluster_ids = kmeans.labels_
    n_clusters = kmeans.n_clusters

    selected = np.zeros(len(X_train), dtype=bool)
    propagated_labels = np.empty(len(X_train), dtype=int)

    for j in range(n_clusters):
        in_cluster = cluster_ids == j

        # 每个样本到“自己所属聚类中心”的距离
        cluster_distances = distances[in_cluster, j]

        # 只传播到离中心较近的样本
        cutoff = np.percentile(cluster_distances, percentile)
        selected[in_cluster] = cluster_distances <= cutoff

        # 用该聚类代表样本的真实标签作为传播标签
        propagated_labels[in_cluster] = y_train[rep_idx[j]]

    X_selected = X_train[selected]
    y_selected = propagated_labels[selected]

    model = LogisticRegression(max_iter=10000)
    model.fit(X_selected, y_selected)

    label_accuracy = float(np.mean(y_selected == y_train[selected]))
    test_accuracy = float(
        accuracy_score(y_test, model.predict(X_test))
    )

    return {
        "n_propagated": int(selected.sum()),
        "label_accuracy": label_accuracy,
        "test_accuracy": test_accuracy,
    }


# ── 第 4 部分：图像分割与模型保存 ─────────────────────────────

# Step 14
def synthetic_image(size=48):
    image = np.zeros((size, size, 3), dtype=float)

    midpoint = size // 2

    # 左半边：红色 R=1，绿色从 0 渐变到 1，蓝色 B=0
    left_gradient = np.linspace(0.0, 1.0, midpoint)
    image[:, :midpoint, 0] = 1.0
    image[:, :midpoint, 1] = left_gradient
    image[:, :midpoint, 2] = 0.0

    # 右半边：红色 R=0，绿色从 0 渐变到 1，蓝色 B=1
    right_width = size - midpoint
    right_gradient = np.linspace(0.0, 1.0, right_width)
    image[:, midpoint:, 0] = 0.0
    image[:, midpoint:, 1] = right_gradient
    image[:, midpoint:, 2] = 1.0

    # 左上角四分之一大小的方块涂成白色
    quarter = size // 4
    image[:quarter, :quarter] = 1.0

    return image


# Step 15
def segment_colors(image, k=4, random_state=42):
    height, width, channels = image.shape
    pixels = image.reshape(-1, channels)

    kmeans = fit_kmeans(pixels, k, random_state=random_state)

    # 每个像素替换成它所属聚类中心的 RGB 颜色
    segmented_pixels = kmeans.cluster_centers_[kmeans.labels_]

    return segmented_pixels.reshape(height, width, channels)


# Step 16
def save_and_reload_clusterer(kmeans, rep_labels, path):
    bundle = {
        "kmeans": kmeans,
        "rep_labels": np.asarray(rep_labels),
    }

    joblib.dump(bundle, path)
    return joblib.load(path)


# Step 17
def predict_digit_labels(bundle, images):
    # 原始图片是 (n, 8, 8)，展平后是 (n, 64)
    X = np.asarray(images).reshape(len(images), -1)

    cluster_ids = bundle["kmeans"].predict(X)
    labels = bundle["rep_labels"][cluster_ids]

    # 返回 Python 的整数列表
    return labels.astype(int).tolist()

# Step 6 - dbscan_predict
import os
import joblib
import numpy as np

from sklearn.datasets import make_blobs, make_moons, load_digits
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, accuracy_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# ── 第 1 部分：K-Means 与聚类数选择 ──────────────────────────

# Step 1
def blobs_data(random_state=42):
    centers = [
        [0.2, 2.3],
        [-1.5, 2.3],
        [-2.8, 1.8],
        [-2.8, 2.8],
        [-2.8, 1.3],
    ]
    X, y = make_blobs(
        n_samples=2000,
        centers=centers,
        cluster_std=[0.4, 0.3, 0.1, 0.1, 0.1],
        random_state=random_state,
    )
    return X, y


# Step 2
def fit_kmeans(X, k, random_state=42):
    model = KMeans(
        n_clusters=k,
        n_init=10,
        random_state=random_state,
    )
    return model.fit(X)


# Step 3
def inertia_curve(X, ks):
    return {
        k: float(fit_kmeans(X, k).inertia_)
        for k in ks
    }


# Step 4
def silhouette_curve(X, ks):
    scores = {}

    for k in ks:
        model = fit_kmeans(X, k)
        scores[k] = float(silhouette_score(X, model.labels_))

    return scores


def best_k_by_silhouette(curve):
    # 先按 silhouette 从高到低；若分数相同，选较小的 k
    return min(curve, key=lambda k: (-curve[k], k))


# ── 第 2 部分：DBSCAN、GMM 与异常检测 ─────────────────────────

# Step 5
def fit_dbscan(X, eps=0.2, min_samples=5):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    return model.fit(X)


def dbscan_summary(dbscan):
    labels = dbscan.labels_
    clusters = set(labels.tolist()) - {-1}

    return {
        "n_clusters": int(len(clusters)),
        "n_noise": int(np.sum(labels == -1)),
        "n_core": int(len(dbscan.core_sample_indices_)),
    }


# Step 6
def dbscan_predict(dbscan, X_new, n_neighbors=50):
    core_X = dbscan.components_
    core_labels = dbscan.labels_[dbscan.core_sample_indices_]

    # 如果没有核心点，就把新样本都标成噪声
    if len(core_X) == 0:
        return np.full(len(X_new), -1, dtype=int)

    # 核心点不足 n_neighbors 个时，使用实际可用的数量
    k = min(n_neighbors, len(core_X))

    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(core_X, core_labels)
    return model.predict(X_new).astype(int)


# Step 7
def fit_gmm(X, n_components, random_state=42):
    model = GaussianMixture(
        n_components=n_components,
        n_init=10,
        random_state=random_state,
    )
    return model.fit(X)


def bic_curve(X, ks):
    scores = {}

    for k in ks:
        model = fit_gmm(X, n_components=k)
        scores[k] = float(model.bic(X))

    return scores


# Step 8
def flag_anomalies(gmm, X, contamination=0.04):
    # score_samples 越低，说明样本在模型下越不常见
    densities = gmm.score_samples(X)
    threshold = np.percentile(densities, 100 * contamination)

    return densities < threshold


# ── 第 3 部分：用少量标签学习手写数字 ─────────────────────────

# Step 9
def digits_data(test_size=0.25, random_state=42):
    digits = load_digits()
    X = digits.data
    y = digits.target

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


# Step 10
def baseline_50_random(
    X_train, y_train, X_test, y_test,
    n_labeled=50, random_state=42
):
    # 题目说明：训练集已打乱，所以直接拿前 n_labeled 个样本
    model = LogisticRegression(max_iter=10000)
    model.fit(X_train[:n_labeled], y_train[:n_labeled])

    return float(accuracy_score(y_test, model.predict(X_test)))


# Step 11
def representative_digits(X_train, k=50, random_state=42):
    kmeans = fit_kmeans(X_train, k, random_state=random_state)

    # transform 的形状是 (样本数, 聚类数)
    # 第 j 列是所有样本到第 j 个中心的距离
    distances = kmeans.transform(X_train)

    # 每个聚类里挑离该聚类中心最近的训练样本
    rep_idx = np.argmin(distances, axis=0).astype(int)

    return kmeans, rep_idx


# Step 12
def train_on_representatives(
    X_train, y_train, rep_idx, X_test, y_test
):
    model = LogisticRegression(max_iter=10000)
    model.fit(X_train[rep_idx], y_train[rep_idx])

    return float(accuracy_score(y_test, model.predict(X_test)))


# Step 13
def propagate_and_train(
    X_train, y_train, kmeans, rep_idx,
    X_test, y_test, percentile=20
):
    distances = kmeans.transform(X_train)
    cluster_ids = kmeans.labels_
    n_clusters = kmeans.n_clusters

    selected = np.zeros(len(X_train), dtype=bool)
    propagated_labels = np.empty(len(X_train), dtype=int)

    for j in range(n_clusters):
        in_cluster = cluster_ids == j

        # 每个样本到“自己所属聚类中心”的距离
        cluster_distances = distances[in_cluster, j]

        # 只传播到离中心较近的样本
        cutoff = np.percentile(cluster_distances, percentile)
        selected[in_cluster] = cluster_distances <= cutoff

        # 用该聚类代表样本的真实标签作为传播标签
        propagated_labels[in_cluster] = y_train[rep_idx[j]]

    X_selected = X_train[selected]
    y_selected = propagated_labels[selected]

    model = LogisticRegression(max_iter=10000)
    model.fit(X_selected, y_selected)

    label_accuracy = float(np.mean(y_selected == y_train[selected]))
    test_accuracy = float(
        accuracy_score(y_test, model.predict(X_test))
    )

    return {
        "n_propagated": int(selected.sum()),
        "label_accuracy": label_accuracy,
        "test_accuracy": test_accuracy,
    }


# ── 第 4 部分：图像分割与模型保存 ─────────────────────────────

# Step 14
def synthetic_image(size=48):
    image = np.zeros((size, size, 3), dtype=float)

    midpoint = size // 2

    # 左半边：红色 R=1，绿色从 0 渐变到 1，蓝色 B=0
    left_gradient = np.linspace(0.0, 1.0, midpoint)
    image[:, :midpoint, 0] = 1.0
    image[:, :midpoint, 1] = left_gradient
    image[:, :midpoint, 2] = 0.0

    # 右半边：红色 R=0，绿色从 0 渐变到 1，蓝色 B=1
    right_width = size - midpoint
    right_gradient = np.linspace(0.0, 1.0, right_width)
    image[:, midpoint:, 0] = 0.0
    image[:, midpoint:, 1] = right_gradient
    image[:, midpoint:, 2] = 1.0

    # 左上角四分之一大小的方块涂成白色
    quarter = size // 4
    image[:quarter, :quarter] = 1.0

    return image


# Step 15
def segment_colors(image, k=4, random_state=42):
    height, width, channels = image.shape
    pixels = image.reshape(-1, channels)

    kmeans = fit_kmeans(pixels, k, random_state=random_state)

    # 每个像素替换成它所属聚类中心的 RGB 颜色
    segmented_pixels = kmeans.cluster_centers_[kmeans.labels_]

    return segmented_pixels.reshape(height, width, channels)


# Step 16
def save_and_reload_clusterer(kmeans, rep_labels, path):
    bundle = {
        "kmeans": kmeans,
        "rep_labels": np.asarray(rep_labels),
    }

    joblib.dump(bundle, path)
    return joblib.load(path)


# Step 17
def predict_digit_labels(bundle, images):
    # 原始图片是 (n, 8, 8)，展平后是 (n, 64)
    X = np.asarray(images).reshape(len(images), -1)

    cluster_ids = bundle["kmeans"].predict(X)
    labels = bundle["rep_labels"][cluster_ids]

    # 返回 Python 的整数列表
    return labels.astype(int).tolist()

# Step 7 - fit_gmm
import os
import joblib
import numpy as np

from sklearn.datasets import make_blobs, make_moons, load_digits
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, accuracy_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# ── 第 1 部分：K-Means 与聚类数选择 ──────────────────────────

# Step 1
def blobs_data(random_state=42):
    centers = [
        [0.2, 2.3],
        [-1.5, 2.3],
        [-2.8, 1.8],
        [-2.8, 2.8],
        [-2.8, 1.3],
    ]
    X, y = make_blobs(
        n_samples=2000,
        centers=centers,
        cluster_std=[0.4, 0.3, 0.1, 0.1, 0.1],
        random_state=random_state,
    )
    return X, y


# Step 2
def fit_kmeans(X, k, random_state=42):
    model = KMeans(
        n_clusters=k,
        n_init=10,
        random_state=random_state,
    )
    return model.fit(X)


# Step 3
def inertia_curve(X, ks):
    return {
        k: float(fit_kmeans(X, k).inertia_)
        for k in ks
    }


# Step 4
def silhouette_curve(X, ks):
    scores = {}

    for k in ks:
        model = fit_kmeans(X, k)
        scores[k] = float(silhouette_score(X, model.labels_))

    return scores


def best_k_by_silhouette(curve):
    # 先按 silhouette 从高到低；若分数相同，选较小的 k
    return min(curve, key=lambda k: (-curve[k], k))


# ── 第 2 部分：DBSCAN、GMM 与异常检测 ─────────────────────────

# Step 5
def fit_dbscan(X, eps=0.2, min_samples=5):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    return model.fit(X)


def dbscan_summary(dbscan):
    labels = dbscan.labels_
    clusters = set(labels.tolist()) - {-1}

    return {
        "n_clusters": int(len(clusters)),
        "n_noise": int(np.sum(labels == -1)),
        "n_core": int(len(dbscan.core_sample_indices_)),
    }


# Step 6
def dbscan_predict(dbscan, X_new, n_neighbors=50):
    core_X = dbscan.components_
    core_labels = dbscan.labels_[dbscan.core_sample_indices_]

    # 如果没有核心点，就把新样本都标成噪声
    if len(core_X) == 0:
        return np.full(len(X_new), -1, dtype=int)

    # 核心点不足 n_neighbors 个时，使用实际可用的数量
    k = min(n_neighbors, len(core_X))

    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(core_X, core_labels)
    return model.predict(X_new).astype(int)


# Step 7
def fit_gmm(X, n_components, random_state=42):
    model = GaussianMixture(
        n_components=n_components,
        n_init=10,
        random_state=random_state,
    )
    return model.fit(X)


def bic_curve(X, ks):
    scores = {}

    for k in ks:
        model = fit_gmm(X, n_components=k)
        scores[k] = float(model.bic(X))

    return scores


# Step 8
def flag_anomalies(gmm, X, contamination=0.04):
    # score_samples 越低，说明样本在模型下越不常见
    densities = gmm.score_samples(X)
    threshold = np.percentile(densities, 100 * contamination)

    return densities < threshold


# ── 第 3 部分：用少量标签学习手写数字 ─────────────────────────

# Step 9
def digits_data(test_size=0.25, random_state=42):
    digits = load_digits()
    X = digits.data
    y = digits.target

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


# Step 10
def baseline_50_random(
    X_train, y_train, X_test, y_test,
    n_labeled=50, random_state=42
):
    # 题目说明：训练集已打乱，所以直接拿前 n_labeled 个样本
    model = LogisticRegression(max_iter=10000)
    model.fit(X_train[:n_labeled], y_train[:n_labeled])

    return float(accuracy_score(y_test, model.predict(X_test)))


# Step 11
def representative_digits(X_train, k=50, random_state=42):
    kmeans = fit_kmeans(X_train, k, random_state=random_state)

    # transform 的形状是 (样本数, 聚类数)
    # 第 j 列是所有样本到第 j 个中心的距离
    distances = kmeans.transform(X_train)

    # 每个聚类里挑离该聚类中心最近的训练样本
    rep_idx = np.argmin(distances, axis=0).astype(int)

    return kmeans, rep_idx


# Step 12
def train_on_representatives(
    X_train, y_train, rep_idx, X_test, y_test
):
    model = LogisticRegression(max_iter=10000)
    model.fit(X_train[rep_idx], y_train[rep_idx])

    return float(accuracy_score(y_test, model.predict(X_test)))


# Step 13
def propagate_and_train(
    X_train, y_train, kmeans, rep_idx,
    X_test, y_test, percentile=20
):
    distances = kmeans.transform(X_train)
    cluster_ids = kmeans.labels_
    n_clusters = kmeans.n_clusters

    selected = np.zeros(len(X_train), dtype=bool)
    propagated_labels = np.empty(len(X_train), dtype=int)

    for j in range(n_clusters):
        in_cluster = cluster_ids == j

        # 每个样本到“自己所属聚类中心”的距离
        cluster_distances = distances[in_cluster, j]

        # 只传播到离中心较近的样本
        cutoff = np.percentile(cluster_distances, percentile)
        selected[in_cluster] = cluster_distances <= cutoff

        # 用该聚类代表样本的真实标签作为传播标签
        propagated_labels[in_cluster] = y_train[rep_idx[j]]

    X_selected = X_train[selected]
    y_selected = propagated_labels[selected]

    model = LogisticRegression(max_iter=10000)
    model.fit(X_selected, y_selected)

    label_accuracy = float(np.mean(y_selected == y_train[selected]))
    test_accuracy = float(
        accuracy_score(y_test, model.predict(X_test))
    )

    return {
        "n_propagated": int(selected.sum()),
        "label_accuracy": label_accuracy,
        "test_accuracy": test_accuracy,
    }


# ── 第 4 部分：图像分割与模型保存 ─────────────────────────────

# Step 14
def synthetic_image(size=48):
    image = np.zeros((size, size, 3), dtype=float)

    midpoint = size // 2

    # 左半边：红色 R=1，绿色从 0 渐变到 1，蓝色 B=0
    left_gradient = np.linspace(0.0, 1.0, midpoint)
    image[:, :midpoint, 0] = 1.0
    image[:, :midpoint, 1] = left_gradient
    image[:, :midpoint, 2] = 0.0

    # 右半边：红色 R=0，绿色从 0 渐变到 1，蓝色 B=1
    right_width = size - midpoint
    right_gradient = np.linspace(0.0, 1.0, right_width)
    image[:, midpoint:, 0] = 0.0
    image[:, midpoint:, 1] = right_gradient
    image[:, midpoint:, 2] = 1.0

    # 左上角四分之一大小的方块涂成白色
    quarter = size // 4
    image[:quarter, :quarter] = 1.0

    return image


# Step 15
def segment_colors(image, k=4, random_state=42):
    height, width, channels = image.shape
    pixels = image.reshape(-1, channels)

    kmeans = fit_kmeans(pixels, k, random_state=random_state)

    # 每个像素替换成它所属聚类中心的 RGB 颜色
    segmented_pixels = kmeans.cluster_centers_[kmeans.labels_]

    return segmented_pixels.reshape(height, width, channels)


# Step 16
def save_and_reload_clusterer(kmeans, rep_labels, path):
    bundle = {
        "kmeans": kmeans,
        "rep_labels": np.asarray(rep_labels),
    }

    joblib.dump(bundle, path)
    return joblib.load(path)


# Step 17
def predict_digit_labels(bundle, images):
    # 原始图片是 (n, 8, 8)，展平后是 (n, 64)
    X = np.asarray(images).reshape(len(images), -1)

    cluster_ids = bundle["kmeans"].predict(X)
    labels = bundle["rep_labels"][cluster_ids]

    # 返回 Python 的整数列表
    return labels.astype(int).tolist()

# Step 8 - flag_anomalies
import os
import joblib
import numpy as np

from sklearn.datasets import make_blobs, make_moons, load_digits
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, accuracy_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# ── 第 1 部分：K-Means 与聚类数选择 ──────────────────────────

# Step 1
def blobs_data(random_state=42):
    centers = [
        [0.2, 2.3],
        [-1.5, 2.3],
        [-2.8, 1.8],
        [-2.8, 2.8],
        [-2.8, 1.3],
    ]
    X, y = make_blobs(
        n_samples=2000,
        centers=centers,
        cluster_std=[0.4, 0.3, 0.1, 0.1, 0.1],
        random_state=random_state,
    )
    return X, y


# Step 2
def fit_kmeans(X, k, random_state=42):
    model = KMeans(
        n_clusters=k,
        n_init=10,
        random_state=random_state,
    )
    return model.fit(X)


# Step 3
def inertia_curve(X, ks):
    return {
        k: float(fit_kmeans(X, k).inertia_)
        for k in ks
    }


# Step 4
def silhouette_curve(X, ks):
    scores = {}

    for k in ks:
        model = fit_kmeans(X, k)
        scores[k] = float(silhouette_score(X, model.labels_))

    return scores


def best_k_by_silhouette(curve):
    # 先按 silhouette 从高到低；若分数相同，选较小的 k
    return min(curve, key=lambda k: (-curve[k], k))


# ── 第 2 部分：DBSCAN、GMM 与异常检测 ─────────────────────────

# Step 5
def fit_dbscan(X, eps=0.2, min_samples=5):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    return model.fit(X)


def dbscan_summary(dbscan):
    labels = dbscan.labels_
    clusters = set(labels.tolist()) - {-1}

    return {
        "n_clusters": int(len(clusters)),
        "n_noise": int(np.sum(labels == -1)),
        "n_core": int(len(dbscan.core_sample_indices_)),
    }


# Step 6
def dbscan_predict(dbscan, X_new, n_neighbors=50):
    core_X = dbscan.components_
    core_labels = dbscan.labels_[dbscan.core_sample_indices_]

    # 如果没有核心点，就把新样本都标成噪声
    if len(core_X) == 0:
        return np.full(len(X_new), -1, dtype=int)

    # 核心点不足 n_neighbors 个时，使用实际可用的数量
    k = min(n_neighbors, len(core_X))

    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(core_X, core_labels)
    return model.predict(X_new).astype(int)


# Step 7
def fit_gmm(X, n_components, random_state=42):
    model = GaussianMixture(
        n_components=n_components,
        n_init=10,
        random_state=random_state,
    )
    return model.fit(X)


def bic_curve(X, ks):
    scores = {}

    for k in ks:
        model = fit_gmm(X, n_components=k)
        scores[k] = float(model.bic(X))

    return scores


# Step 8
def flag_anomalies(gmm, X, contamination=0.04):
    # score_samples 越低，说明样本在模型下越不常见
    densities = gmm.score_samples(X)
    threshold = np.percentile(densities, 100 * contamination)

    return densities < threshold


# ── 第 3 部分：用少量标签学习手写数字 ─────────────────────────

# Step 9
def digits_data(test_size=0.25, random_state=42):
    digits = load_digits()
    X = digits.data
    y = digits.target

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


# Step 10
def baseline_50_random(
    X_train, y_train, X_test, y_test,
    n_labeled=50, random_state=42
):
    # 题目说明：训练集已打乱，所以直接拿前 n_labeled 个样本
    model = LogisticRegression(max_iter=10000)
    model.fit(X_train[:n_labeled], y_train[:n_labeled])

    return float(accuracy_score(y_test, model.predict(X_test)))


# Step 11
def representative_digits(X_train, k=50, random_state=42):
    kmeans = fit_kmeans(X_train, k, random_state=random_state)

    # transform 的形状是 (样本数, 聚类数)
    # 第 j 列是所有样本到第 j 个中心的距离
    distances = kmeans.transform(X_train)

    # 每个聚类里挑离该聚类中心最近的训练样本
    rep_idx = np.argmin(distances, axis=0).astype(int)

    return kmeans, rep_idx


# Step 12
def train_on_representatives(
    X_train, y_train, rep_idx, X_test, y_test
):
    model = LogisticRegression(max_iter=10000)
    model.fit(X_train[rep_idx], y_train[rep_idx])

    return float(accuracy_score(y_test, model.predict(X_test)))


# Step 13
def propagate_and_train(
    X_train, y_train, kmeans, rep_idx,
    X_test, y_test, percentile=20
):
    distances = kmeans.transform(X_train)
    cluster_ids = kmeans.labels_
    n_clusters = kmeans.n_clusters

    selected = np.zeros(len(X_train), dtype=bool)
    propagated_labels = np.empty(len(X_train), dtype=int)

    for j in range(n_clusters):
        in_cluster = cluster_ids == j

        # 每个样本到“自己所属聚类中心”的距离
        cluster_distances = distances[in_cluster, j]

        # 只传播到离中心较近的样本
        cutoff = np.percentile(cluster_distances, percentile)
        selected[in_cluster] = cluster_distances <= cutoff

        # 用该聚类代表样本的真实标签作为传播标签
        propagated_labels[in_cluster] = y_train[rep_idx[j]]

    X_selected = X_train[selected]
    y_selected = propagated_labels[selected]

    model = LogisticRegression(max_iter=10000)
    model.fit(X_selected, y_selected)

    label_accuracy = float(np.mean(y_selected == y_train[selected]))
    test_accuracy = float(
        accuracy_score(y_test, model.predict(X_test))
    )

    return {
        "n_propagated": int(selected.sum()),
        "label_accuracy": label_accuracy,
        "test_accuracy": test_accuracy,
    }


# ── 第 4 部分：图像分割与模型保存 ─────────────────────────────

# Step 14
def synthetic_image(size=48):
    image = np.zeros((size, size, 3), dtype=float)

    midpoint = size // 2

    # 左半边：红色 R=1，绿色从 0 渐变到 1，蓝色 B=0
    left_gradient = np.linspace(0.0, 1.0, midpoint)
    image[:, :midpoint, 0] = 1.0
    image[:, :midpoint, 1] = left_gradient
    image[:, :midpoint, 2] = 0.0

    # 右半边：红色 R=0，绿色从 0 渐变到 1，蓝色 B=1
    right_width = size - midpoint
    right_gradient = np.linspace(0.0, 1.0, right_width)
    image[:, midpoint:, 0] = 0.0
    image[:, midpoint:, 1] = right_gradient
    image[:, midpoint:, 2] = 1.0

    # 左上角四分之一大小的方块涂成白色
    quarter = size // 4
    image[:quarter, :quarter] = 1.0

    return image


# Step 15
def segment_colors(image, k=4, random_state=42):
    height, width, channels = image.shape
    pixels = image.reshape(-1, channels)

    kmeans = fit_kmeans(pixels, k, random_state=random_state)

    # 每个像素替换成它所属聚类中心的 RGB 颜色
    segmented_pixels = kmeans.cluster_centers_[kmeans.labels_]

    return segmented_pixels.reshape(height, width, channels)


# Step 16
def save_and_reload_clusterer(kmeans, rep_labels, path):
    bundle = {
        "kmeans": kmeans,
        "rep_labels": np.asarray(rep_labels),
    }

    joblib.dump(bundle, path)
    return joblib.load(path)


# Step 17
def predict_digit_labels(bundle, images):
    # 原始图片是 (n, 8, 8)，展平后是 (n, 64)
    X = np.asarray(images).reshape(len(images), -1)

    cluster_ids = bundle["kmeans"].predict(X)
    labels = bundle["rep_labels"][cluster_ids]

    # 返回 Python 的整数列表
    return labels.astype(int).tolist()

# Step 9 - digits_data
import os
import joblib
import numpy as np

from sklearn.datasets import make_blobs, make_moons, load_digits
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, accuracy_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# ── 第 1 部分：K-Means 与聚类数选择 ──────────────────────────

# Step 1
def blobs_data(random_state=42):
    centers = [
        [0.2, 2.3],
        [-1.5, 2.3],
        [-2.8, 1.8],
        [-2.8, 2.8],
        [-2.8, 1.3],
    ]
    X, y = make_blobs(
        n_samples=2000,
        centers=centers,
        cluster_std=[0.4, 0.3, 0.1, 0.1, 0.1],
        random_state=random_state,
    )
    return X, y


# Step 2
def fit_kmeans(X, k, random_state=42):
    model = KMeans(
        n_clusters=k,
        n_init=10,
        random_state=random_state,
    )
    return model.fit(X)


# Step 3
def inertia_curve(X, ks):
    return {
        k: float(fit_kmeans(X, k).inertia_)
        for k in ks
    }


# Step 4
def silhouette_curve(X, ks):
    scores = {}

    for k in ks:
        model = fit_kmeans(X, k)
        scores[k] = float(silhouette_score(X, model.labels_))

    return scores


def best_k_by_silhouette(curve):
    # 先按 silhouette 从高到低；若分数相同，选较小的 k
    return min(curve, key=lambda k: (-curve[k], k))


# ── 第 2 部分：DBSCAN、GMM 与异常检测 ─────────────────────────

# Step 5
def fit_dbscan(X, eps=0.2, min_samples=5):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    return model.fit(X)


def dbscan_summary(dbscan):
    labels = dbscan.labels_
    clusters = set(labels.tolist()) - {-1}

    return {
        "n_clusters": int(len(clusters)),
        "n_noise": int(np.sum(labels == -1)),
        "n_core": int(len(dbscan.core_sample_indices_)),
    }


# Step 6
def dbscan_predict(dbscan, X_new, n_neighbors=50):
    core_X = dbscan.components_
    core_labels = dbscan.labels_[dbscan.core_sample_indices_]

    # 如果没有核心点，就把新样本都标成噪声
    if len(core_X) == 0:
        return np.full(len(X_new), -1, dtype=int)

    # 核心点不足 n_neighbors 个时，使用实际可用的数量
    k = min(n_neighbors, len(core_X))

    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(core_X, core_labels)
    return model.predict(X_new).astype(int)


# Step 7
def fit_gmm(X, n_components, random_state=42):
    model = GaussianMixture(
        n_components=n_components,
        n_init=10,
        random_state=random_state,
    )
    return model.fit(X)


def bic_curve(X, ks):
    scores = {}

    for k in ks:
        model = fit_gmm(X, n_components=k)
        scores[k] = float(model.bic(X))

    return scores


# Step 8
def flag_anomalies(gmm, X, contamination=0.04):
    # score_samples 越低，说明样本在模型下越不常见
    densities = gmm.score_samples(X)
    threshold = np.percentile(densities, 100 * contamination)

    return densities < threshold


# ── 第 3 部分：用少量标签学习手写数字 ─────────────────────────

# Step 9
def digits_data(test_size=0.25, random_state=42):
    digits = load_digits()
    X = digits.data
    y = digits.target

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


# Step 10
def baseline_50_random(
    X_train, y_train, X_test, y_test,
    n_labeled=50, random_state=42
):
    # 题目说明：训练集已打乱，所以直接拿前 n_labeled 个样本
    model = LogisticRegression(max_iter=10000)
    model.fit(X_train[:n_labeled], y_train[:n_labeled])

    return float(accuracy_score(y_test, model.predict(X_test)))


# Step 11
def representative_digits(X_train, k=50, random_state=42):
    kmeans = fit_kmeans(X_train, k, random_state=random_state)

    # transform 的形状是 (样本数, 聚类数)
    # 第 j 列是所有样本到第 j 个中心的距离
    distances = kmeans.transform(X_train)

    # 每个聚类里挑离该聚类中心最近的训练样本
    rep_idx = np.argmin(distances, axis=0).astype(int)

    return kmeans, rep_idx


# Step 12
def train_on_representatives(
    X_train, y_train, rep_idx, X_test, y_test
):
    model = LogisticRegression(max_iter=10000)
    model.fit(X_train[rep_idx], y_train[rep_idx])

    return float(accuracy_score(y_test, model.predict(X_test)))


# Step 13
def propagate_and_train(
    X_train, y_train, kmeans, rep_idx,
    X_test, y_test, percentile=20
):
    distances = kmeans.transform(X_train)
    cluster_ids = kmeans.labels_
    n_clusters = kmeans.n_clusters

    selected = np.zeros(len(X_train), dtype=bool)
    propagated_labels = np.empty(len(X_train), dtype=int)

    for j in range(n_clusters):
        in_cluster = cluster_ids == j

        # 每个样本到“自己所属聚类中心”的距离
        cluster_distances = distances[in_cluster, j]

        # 只传播到离中心较近的样本
        cutoff = np.percentile(cluster_distances, percentile)
        selected[in_cluster] = cluster_distances <= cutoff

        # 用该聚类代表样本的真实标签作为传播标签
        propagated_labels[in_cluster] = y_train[rep_idx[j]]

    X_selected = X_train[selected]
    y_selected = propagated_labels[selected]

    model = LogisticRegression(max_iter=10000)
    model.fit(X_selected, y_selected)

    label_accuracy = float(np.mean(y_selected == y_train[selected]))
    test_accuracy = float(
        accuracy_score(y_test, model.predict(X_test))
    )

    return {
        "n_propagated": int(selected.sum()),
        "label_accuracy": label_accuracy,
        "test_accuracy": test_accuracy,
    }


# ── 第 4 部分：图像分割与模型保存 ─────────────────────────────

# Step 14
def synthetic_image(size=48):
    image = np.zeros((size, size, 3), dtype=float)

    midpoint = size // 2

    # 左半边：红色 R=1，绿色从 0 渐变到 1，蓝色 B=0
    left_gradient = np.linspace(0.0, 1.0, midpoint)
    image[:, :midpoint, 0] = 1.0
    image[:, :midpoint, 1] = left_gradient
    image[:, :midpoint, 2] = 0.0

    # 右半边：红色 R=0，绿色从 0 渐变到 1，蓝色 B=1
    right_width = size - midpoint
    right_gradient = np.linspace(0.0, 1.0, right_width)
    image[:, midpoint:, 0] = 0.0
    image[:, midpoint:, 1] = right_gradient
    image[:, midpoint:, 2] = 1.0

    # 左上角四分之一大小的方块涂成白色
    quarter = size // 4
    image[:quarter, :quarter] = 1.0

    return image


# Step 15
def segment_colors(image, k=4, random_state=42):
    height, width, channels = image.shape
    pixels = image.reshape(-1, channels)

    kmeans = fit_kmeans(pixels, k, random_state=random_state)

    # 每个像素替换成它所属聚类中心的 RGB 颜色
    segmented_pixels = kmeans.cluster_centers_[kmeans.labels_]

    return segmented_pixels.reshape(height, width, channels)


# Step 16
def save_and_reload_clusterer(kmeans, rep_labels, path):
    bundle = {
        "kmeans": kmeans,
        "rep_labels": np.asarray(rep_labels),
    }

    joblib.dump(bundle, path)
    return joblib.load(path)


# Step 17
def predict_digit_labels(bundle, images):
    # 原始图片是 (n, 8, 8)，展平后是 (n, 64)
    X = np.asarray(images).reshape(len(images), -1)

    cluster_ids = bundle["kmeans"].predict(X)
    labels = bundle["rep_labels"][cluster_ids]

    # 返回 Python 的整数列表
    return labels.astype(int).tolist()

# Step 10 - baseline_50_random
import os
import joblib
import numpy as np

from sklearn.datasets import make_blobs, make_moons, load_digits
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, accuracy_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# ── 第 1 部分：K-Means 与聚类数选择 ──────────────────────────

# Step 1
def blobs_data(random_state=42):
    centers = [
        [0.2, 2.3],
        [-1.5, 2.3],
        [-2.8, 1.8],
        [-2.8, 2.8],
        [-2.8, 1.3],
    ]
    X, y = make_blobs(
        n_samples=2000,
        centers=centers,
        cluster_std=[0.4, 0.3, 0.1, 0.1, 0.1],
        random_state=random_state,
    )
    return X, y


# Step 2
def fit_kmeans(X, k, random_state=42):
    model = KMeans(
        n_clusters=k,
        n_init=10,
        random_state=random_state,
    )
    return model.fit(X)


# Step 3
def inertia_curve(X, ks):
    return {
        k: float(fit_kmeans(X, k).inertia_)
        for k in ks
    }


# Step 4
def silhouette_curve(X, ks):
    scores = {}

    for k in ks:
        model = fit_kmeans(X, k)
        scores[k] = float(silhouette_score(X, model.labels_))

    return scores


def best_k_by_silhouette(curve):
    # 先按 silhouette 从高到低；若分数相同，选较小的 k
    return min(curve, key=lambda k: (-curve[k], k))


# ── 第 2 部分：DBSCAN、GMM 与异常检测 ─────────────────────────

# Step 5
def fit_dbscan(X, eps=0.2, min_samples=5):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    return model.fit(X)


def dbscan_summary(dbscan):
    labels = dbscan.labels_
    clusters = set(labels.tolist()) - {-1}

    return {
        "n_clusters": int(len(clusters)),
        "n_noise": int(np.sum(labels == -1)),
        "n_core": int(len(dbscan.core_sample_indices_)),
    }


# Step 6
def dbscan_predict(dbscan, X_new, n_neighbors=50):
    core_X = dbscan.components_
    core_labels = dbscan.labels_[dbscan.core_sample_indices_]

    # 如果没有核心点，就把新样本都标成噪声
    if len(core_X) == 0:
        return np.full(len(X_new), -1, dtype=int)

    # 核心点不足 n_neighbors 个时，使用实际可用的数量
    k = min(n_neighbors, len(core_X))

    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(core_X, core_labels)
    return model.predict(X_new).astype(int)


# Step 7
def fit_gmm(X, n_components, random_state=42):
    model = GaussianMixture(
        n_components=n_components,
        n_init=10,
        random_state=random_state,
    )
    return model.fit(X)


def bic_curve(X, ks):
    scores = {}

    for k in ks:
        model = fit_gmm(X, n_components=k)
        scores[k] = float(model.bic(X))

    return scores


# Step 8
def flag_anomalies(gmm, X, contamination=0.04):
    # score_samples 越低，说明样本在模型下越不常见
    densities = gmm.score_samples(X)
    threshold = np.percentile(densities, 100 * contamination)

    return densities < threshold


# ── 第 3 部分：用少量标签学习手写数字 ─────────────────────────

# Step 9
def digits_data(test_size=0.25, random_state=42):
    digits = load_digits()
    X = digits.data
    y = digits.target

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


# Step 10
def baseline_50_random(
    X_train, y_train, X_test, y_test,
    n_labeled=50, random_state=42
):
    # 题目说明：训练集已打乱，所以直接拿前 n_labeled 个样本
    model = LogisticRegression(max_iter=10000)
    model.fit(X_train[:n_labeled], y_train[:n_labeled])

    return float(accuracy_score(y_test, model.predict(X_test)))


# Step 11
def representative_digits(X_train, k=50, random_state=42):
    kmeans = fit_kmeans(X_train, k, random_state=random_state)

    # transform 的形状是 (样本数, 聚类数)
    # 第 j 列是所有样本到第 j 个中心的距离
    distances = kmeans.transform(X_train)

    # 每个聚类里挑离该聚类中心最近的训练样本
    rep_idx = np.argmin(distances, axis=0).astype(int)

    return kmeans, rep_idx


# Step 12
def train_on_representatives(
    X_train, y_train, rep_idx, X_test, y_test
):
    model = LogisticRegression(max_iter=10000)
    model.fit(X_train[rep_idx], y_train[rep_idx])

    return float(accuracy_score(y_test, model.predict(X_test)))


# Step 13
def propagate_and_train(
    X_train, y_train, kmeans, rep_idx,
    X_test, y_test, percentile=20
):
    distances = kmeans.transform(X_train)
    cluster_ids = kmeans.labels_
    n_clusters = kmeans.n_clusters

    selected = np.zeros(len(X_train), dtype=bool)
    propagated_labels = np.empty(len(X_train), dtype=int)

    for j in range(n_clusters):
        in_cluster = cluster_ids == j

        # 每个样本到“自己所属聚类中心”的距离
        cluster_distances = distances[in_cluster, j]

        # 只传播到离中心较近的样本
        cutoff = np.percentile(cluster_distances, percentile)
        selected[in_cluster] = cluster_distances <= cutoff

        # 用该聚类代表样本的真实标签作为传播标签
        propagated_labels[in_cluster] = y_train[rep_idx[j]]

    X_selected = X_train[selected]
    y_selected = propagated_labels[selected]

    model = LogisticRegression(max_iter=10000)
    model.fit(X_selected, y_selected)

    label_accuracy = float(np.mean(y_selected == y_train[selected]))
    test_accuracy = float(
        accuracy_score(y_test, model.predict(X_test))
    )

    return {
        "n_propagated": int(selected.sum()),
        "label_accuracy": label_accuracy,
        "test_accuracy": test_accuracy,
    }


# ── 第 4 部分：图像分割与模型保存 ─────────────────────────────

# Step 14
def synthetic_image(size=48):
    image = np.zeros((size, size, 3), dtype=float)

    midpoint = size // 2

    # 左半边：红色 R=1，绿色从 0 渐变到 1，蓝色 B=0
    left_gradient = np.linspace(0.0, 1.0, midpoint)
    image[:, :midpoint, 0] = 1.0
    image[:, :midpoint, 1] = left_gradient
    image[:, :midpoint, 2] = 0.0

    # 右半边：红色 R=0，绿色从 0 渐变到 1，蓝色 B=1
    right_width = size - midpoint
    right_gradient = np.linspace(0.0, 1.0, right_width)
    image[:, midpoint:, 0] = 0.0
    image[:, midpoint:, 1] = right_gradient
    image[:, midpoint:, 2] = 1.0

    # 左上角四分之一大小的方块涂成白色
    quarter = size // 4
    image[:quarter, :quarter] = 1.0

    return image


# Step 15
def segment_colors(image, k=4, random_state=42):
    height, width, channels = image.shape
    pixels = image.reshape(-1, channels)

    kmeans = fit_kmeans(pixels, k, random_state=random_state)

    # 每个像素替换成它所属聚类中心的 RGB 颜色
    segmented_pixels = kmeans.cluster_centers_[kmeans.labels_]

    return segmented_pixels.reshape(height, width, channels)


# Step 16
def save_and_reload_clusterer(kmeans, rep_labels, path):
    bundle = {
        "kmeans": kmeans,
        "rep_labels": np.asarray(rep_labels),
    }

    joblib.dump(bundle, path)
    return joblib.load(path)


# Step 17
def predict_digit_labels(bundle, images):
    # 原始图片是 (n, 8, 8)，展平后是 (n, 64)
    X = np.asarray(images).reshape(len(images), -1)

    cluster_ids = bundle["kmeans"].predict(X)
    labels = bundle["rep_labels"][cluster_ids]

    # 返回 Python 的整数列表
    return labels.astype(int).tolist()

# Step 11 - representative_digits
import os
import joblib
import numpy as np

from sklearn.datasets import make_blobs, make_moons, load_digits
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, accuracy_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# ── 第 1 部分：K-Means 与聚类数选择 ──────────────────────────

# Step 1
def blobs_data(random_state=42):
    centers = [
        [0.2, 2.3],
        [-1.5, 2.3],
        [-2.8, 1.8],
        [-2.8, 2.8],
        [-2.8, 1.3],
    ]
    X, y = make_blobs(
        n_samples=2000,
        centers=centers,
        cluster_std=[0.4, 0.3, 0.1, 0.1, 0.1],
        random_state=random_state,
    )
    return X, y


# Step 2
def fit_kmeans(X, k, random_state=42):
    model = KMeans(
        n_clusters=k,
        n_init=10,
        random_state=random_state,
    )
    return model.fit(X)


# Step 3
def inertia_curve(X, ks):
    return {
        k: float(fit_kmeans(X, k).inertia_)
        for k in ks
    }


# Step 4
def silhouette_curve(X, ks):
    scores = {}

    for k in ks:
        model = fit_kmeans(X, k)
        scores[k] = float(silhouette_score(X, model.labels_))

    return scores


def best_k_by_silhouette(curve):
    # 先按 silhouette 从高到低；若分数相同，选较小的 k
    return min(curve, key=lambda k: (-curve[k], k))


# ── 第 2 部分：DBSCAN、GMM 与异常检测 ─────────────────────────

# Step 5
def fit_dbscan(X, eps=0.2, min_samples=5):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    return model.fit(X)


def dbscan_summary(dbscan):
    labels = dbscan.labels_
    clusters = set(labels.tolist()) - {-1}

    return {
        "n_clusters": int(len(clusters)),
        "n_noise": int(np.sum(labels == -1)),
        "n_core": int(len(dbscan.core_sample_indices_)),
    }


# Step 6
def dbscan_predict(dbscan, X_new, n_neighbors=50):
    core_X = dbscan.components_
    core_labels = dbscan.labels_[dbscan.core_sample_indices_]

    # 如果没有核心点，就把新样本都标成噪声
    if len(core_X) == 0:
        return np.full(len(X_new), -1, dtype=int)

    # 核心点不足 n_neighbors 个时，使用实际可用的数量
    k = min(n_neighbors, len(core_X))

    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(core_X, core_labels)
    return model.predict(X_new).astype(int)


# Step 7
def fit_gmm(X, n_components, random_state=42):
    model = GaussianMixture(
        n_components=n_components,
        n_init=10,
        random_state=random_state,
    )
    return model.fit(X)


def bic_curve(X, ks):
    scores = {}

    for k in ks:
        model = fit_gmm(X, n_components=k)
        scores[k] = float(model.bic(X))

    return scores


# Step 8
def flag_anomalies(gmm, X, contamination=0.04):
    # score_samples 越低，说明样本在模型下越不常见
    densities = gmm.score_samples(X)
    threshold = np.percentile(densities, 100 * contamination)

    return densities < threshold


# ── 第 3 部分：用少量标签学习手写数字 ─────────────────────────

# Step 9
def digits_data(test_size=0.25, random_state=42):
    digits = load_digits()
    X = digits.data
    y = digits.target

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


# Step 10
def baseline_50_random(
    X_train, y_train, X_test, y_test,
    n_labeled=50, random_state=42
):
    # 题目说明：训练集已打乱，所以直接拿前 n_labeled 个样本
    model = LogisticRegression(max_iter=10000)
    model.fit(X_train[:n_labeled], y_train[:n_labeled])

    return float(accuracy_score(y_test, model.predict(X_test)))


# Step 11
def representative_digits(X_train, k=50, random_state=42):
    kmeans = fit_kmeans(X_train, k, random_state=random_state)

    # transform 的形状是 (样本数, 聚类数)
    # 第 j 列是所有样本到第 j 个中心的距离
    distances = kmeans.transform(X_train)

    # 每个聚类里挑离该聚类中心最近的训练样本
    rep_idx = np.argmin(distances, axis=0).astype(int)

    return kmeans, rep_idx


# Step 12
def train_on_representatives(
    X_train, y_train, rep_idx, X_test, y_test
):
    model = LogisticRegression(max_iter=10000)
    model.fit(X_train[rep_idx], y_train[rep_idx])

    return float(accuracy_score(y_test, model.predict(X_test)))


# Step 13
def propagate_and_train(
    X_train, y_train, kmeans, rep_idx,
    X_test, y_test, percentile=20
):
    distances = kmeans.transform(X_train)
    cluster_ids = kmeans.labels_
    n_clusters = kmeans.n_clusters

    selected = np.zeros(len(X_train), dtype=bool)
    propagated_labels = np.empty(len(X_train), dtype=int)

    for j in range(n_clusters):
        in_cluster = cluster_ids == j

        # 每个样本到“自己所属聚类中心”的距离
        cluster_distances = distances[in_cluster, j]

        # 只传播到离中心较近的样本
        cutoff = np.percentile(cluster_distances, percentile)
        selected[in_cluster] = cluster_distances <= cutoff

        # 用该聚类代表样本的真实标签作为传播标签
        propagated_labels[in_cluster] = y_train[rep_idx[j]]

    X_selected = X_train[selected]
    y_selected = propagated_labels[selected]

    model = LogisticRegression(max_iter=10000)
    model.fit(X_selected, y_selected)

    label_accuracy = float(np.mean(y_selected == y_train[selected]))
    test_accuracy = float(
        accuracy_score(y_test, model.predict(X_test))
    )

    return {
        "n_propagated": int(selected.sum()),
        "label_accuracy": label_accuracy,
        "test_accuracy": test_accuracy,
    }


# ── 第 4 部分：图像分割与模型保存 ─────────────────────────────

# Step 14
def synthetic_image(size=48):
    image = np.zeros((size, size, 3), dtype=float)

    midpoint = size // 2

    # 左半边：红色 R=1，绿色从 0 渐变到 1，蓝色 B=0
    left_gradient = np.linspace(0.0, 1.0, midpoint)
    image[:, :midpoint, 0] = 1.0
    image[:, :midpoint, 1] = left_gradient
    image[:, :midpoint, 2] = 0.0

    # 右半边：红色 R=0，绿色从 0 渐变到 1，蓝色 B=1
    right_width = size - midpoint
    right_gradient = np.linspace(0.0, 1.0, right_width)
    image[:, midpoint:, 0] = 0.0
    image[:, midpoint:, 1] = right_gradient
    image[:, midpoint:, 2] = 1.0

    # 左上角四分之一大小的方块涂成白色
    quarter = size // 4
    image[:quarter, :quarter] = 1.0

    return image


# Step 15
def segment_colors(image, k=4, random_state=42):
    height, width, channels = image.shape
    pixels = image.reshape(-1, channels)

    kmeans = fit_kmeans(pixels, k, random_state=random_state)

    # 每个像素替换成它所属聚类中心的 RGB 颜色
    segmented_pixels = kmeans.cluster_centers_[kmeans.labels_]

    return segmented_pixels.reshape(height, width, channels)


# Step 16
def save_and_reload_clusterer(kmeans, rep_labels, path):
    bundle = {
        "kmeans": kmeans,
        "rep_labels": np.asarray(rep_labels),
    }

    joblib.dump(bundle, path)
    return joblib.load(path)


# Step 17
def predict_digit_labels(bundle, images):
    # 原始图片是 (n, 8, 8)，展平后是 (n, 64)
    X = np.asarray(images).reshape(len(images), -1)

    cluster_ids = bundle["kmeans"].predict(X)
    labels = bundle["rep_labels"][cluster_ids]

    # 返回 Python 的整数列表
    return labels.astype(int).tolist()

# Step 12 - train_on_representatives
import os
import joblib
import numpy as np

from sklearn.datasets import make_blobs, make_moons, load_digits
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, accuracy_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# ── 第 1 部分：K-Means 与聚类数选择 ──────────────────────────

# Step 1
def blobs_data(random_state=42):
    centers = [
        [0.2, 2.3],
        [-1.5, 2.3],
        [-2.8, 1.8],
        [-2.8, 2.8],
        [-2.8, 1.3],
    ]
    X, y = make_blobs(
        n_samples=2000,
        centers=centers,
        cluster_std=[0.4, 0.3, 0.1, 0.1, 0.1],
        random_state=random_state,
    )
    return X, y


# Step 2
def fit_kmeans(X, k, random_state=42):
    model = KMeans(
        n_clusters=k,
        n_init=10,
        random_state=random_state,
    )
    return model.fit(X)


# Step 3
def inertia_curve(X, ks):
    return {
        k: float(fit_kmeans(X, k).inertia_)
        for k in ks
    }


# Step 4
def silhouette_curve(X, ks):
    scores = {}

    for k in ks:
        model = fit_kmeans(X, k)
        scores[k] = float(silhouette_score(X, model.labels_))

    return scores


def best_k_by_silhouette(curve):
    # 先按 silhouette 从高到低；若分数相同，选较小的 k
    return min(curve, key=lambda k: (-curve[k], k))


# ── 第 2 部分：DBSCAN、GMM 与异常检测 ─────────────────────────

# Step 5
def fit_dbscan(X, eps=0.2, min_samples=5):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    return model.fit(X)


def dbscan_summary(dbscan):
    labels = dbscan.labels_
    clusters = set(labels.tolist()) - {-1}

    return {
        "n_clusters": int(len(clusters)),
        "n_noise": int(np.sum(labels == -1)),
        "n_core": int(len(dbscan.core_sample_indices_)),
    }


# Step 6
def dbscan_predict(dbscan, X_new, n_neighbors=50):
    core_X = dbscan.components_
    core_labels = dbscan.labels_[dbscan.core_sample_indices_]

    # 如果没有核心点，就把新样本都标成噪声
    if len(core_X) == 0:
        return np.full(len(X_new), -1, dtype=int)

    # 核心点不足 n_neighbors 个时，使用实际可用的数量
    k = min(n_neighbors, len(core_X))

    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(core_X, core_labels)
    return model.predict(X_new).astype(int)


# Step 7
def fit_gmm(X, n_components, random_state=42):
    model = GaussianMixture(
        n_components=n_components,
        n_init=10,
        random_state=random_state,
    )
    return model.fit(X)


def bic_curve(X, ks):
    scores = {}

    for k in ks:
        model = fit_gmm(X, n_components=k)
        scores[k] = float(model.bic(X))

    return scores


# Step 8
def flag_anomalies(gmm, X, contamination=0.04):
    # score_samples 越低，说明样本在模型下越不常见
    densities = gmm.score_samples(X)
    threshold = np.percentile(densities, 100 * contamination)

    return densities < threshold


# ── 第 3 部分：用少量标签学习手写数字 ─────────────────────────

# Step 9
def digits_data(test_size=0.25, random_state=42):
    digits = load_digits()
    X = digits.data
    y = digits.target

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


# Step 10
def baseline_50_random(
    X_train, y_train, X_test, y_test,
    n_labeled=50, random_state=42
):
    # 题目说明：训练集已打乱，所以直接拿前 n_labeled 个样本
    model = LogisticRegression(max_iter=10000)
    model.fit(X_train[:n_labeled], y_train[:n_labeled])

    return float(accuracy_score(y_test, model.predict(X_test)))


# Step 11
def representative_digits(X_train, k=50, random_state=42):
    kmeans = fit_kmeans(X_train, k, random_state=random_state)

    # transform 的形状是 (样本数, 聚类数)
    # 第 j 列是所有样本到第 j 个中心的距离
    distances = kmeans.transform(X_train)

    # 每个聚类里挑离该聚类中心最近的训练样本
    rep_idx = np.argmin(distances, axis=0).astype(int)

    return kmeans, rep_idx


# Step 12
def train_on_representatives(
    X_train, y_train, rep_idx, X_test, y_test
):
    model = LogisticRegression(max_iter=10000)
    model.fit(X_train[rep_idx], y_train[rep_idx])

    return float(accuracy_score(y_test, model.predict(X_test)))


# Step 13
def propagate_and_train(
    X_train, y_train, kmeans, rep_idx,
    X_test, y_test, percentile=20
):
    distances = kmeans.transform(X_train)
    cluster_ids = kmeans.labels_
    n_clusters = kmeans.n_clusters

    selected = np.zeros(len(X_train), dtype=bool)
    propagated_labels = np.empty(len(X_train), dtype=int)

    for j in range(n_clusters):
        in_cluster = cluster_ids == j

        # 每个样本到“自己所属聚类中心”的距离
        cluster_distances = distances[in_cluster, j]

        # 只传播到离中心较近的样本
        cutoff = np.percentile(cluster_distances, percentile)
        selected[in_cluster] = cluster_distances <= cutoff

        # 用该聚类代表样本的真实标签作为传播标签
        propagated_labels[in_cluster] = y_train[rep_idx[j]]

    X_selected = X_train[selected]
    y_selected = propagated_labels[selected]

    model = LogisticRegression(max_iter=10000)
    model.fit(X_selected, y_selected)

    label_accuracy = float(np.mean(y_selected == y_train[selected]))
    test_accuracy = float(
        accuracy_score(y_test, model.predict(X_test))
    )

    return {
        "n_propagated": int(selected.sum()),
        "label_accuracy": label_accuracy,
        "test_accuracy": test_accuracy,
    }


# ── 第 4 部分：图像分割与模型保存 ─────────────────────────────

# Step 14
def synthetic_image(size=48):
    image = np.zeros((size, size, 3), dtype=float)

    midpoint = size // 2

    # 左半边：红色 R=1，绿色从 0 渐变到 1，蓝色 B=0
    left_gradient = np.linspace(0.0, 1.0, midpoint)
    image[:, :midpoint, 0] = 1.0
    image[:, :midpoint, 1] = left_gradient
    image[:, :midpoint, 2] = 0.0

    # 右半边：红色 R=0，绿色从 0 渐变到 1，蓝色 B=1
    right_width = size - midpoint
    right_gradient = np.linspace(0.0, 1.0, right_width)
    image[:, midpoint:, 0] = 0.0
    image[:, midpoint:, 1] = right_gradient
    image[:, midpoint:, 2] = 1.0

    # 左上角四分之一大小的方块涂成白色
    quarter = size // 4
    image[:quarter, :quarter] = 1.0

    return image


# Step 15
def segment_colors(image, k=4, random_state=42):
    height, width, channels = image.shape
    pixels = image.reshape(-1, channels)

    kmeans = fit_kmeans(pixels, k, random_state=random_state)

    # 每个像素替换成它所属聚类中心的 RGB 颜色
    segmented_pixels = kmeans.cluster_centers_[kmeans.labels_]

    return segmented_pixels.reshape(height, width, channels)


# Step 16
def save_and_reload_clusterer(kmeans, rep_labels, path):
    bundle = {
        "kmeans": kmeans,
        "rep_labels": np.asarray(rep_labels),
    }

    joblib.dump(bundle, path)
    return joblib.load(path)


# Step 17
def predict_digit_labels(bundle, images):
    # 原始图片是 (n, 8, 8)，展平后是 (n, 64)
    X = np.asarray(images).reshape(len(images), -1)

    cluster_ids = bundle["kmeans"].predict(X)
    labels = bundle["rep_labels"][cluster_ids]

    # 返回 Python 的整数列表
    return labels.astype(int).tolist()

# Step 13 - propagate_and_train (not yet solved)
# TODO: implement

# Step 14 - synthetic_image (not yet solved)
# TODO: implement

# Step 15 - segment_colors (not yet solved)
# TODO: implement

# Step 16 - save_and_reload_clusterer (not yet solved)
# TODO: implement

# Step 17 - predict_digit_labels (not yet solved)
# TODO: implement

