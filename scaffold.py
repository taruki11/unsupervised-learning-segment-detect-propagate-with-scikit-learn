"""
Unsupervised Learning: Segment, Detect, Propagate with Scikit-Learn scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""Unsupervised learning with scikit-learn (Hands-On ML, chapter 8).

Story: choose k on five blobs with inertia and silhouette; cluster the moons with
DBSCAN and predict for new points through its core samples; fit Gaussian
mixtures, pick the component count with BIC and flag anomalies by density; label
fifty representative digits, propagate their labels through the clusters and
train a classifier that rivals one trained on far more labels; segment an image's
colors; then save the digit clusterer and serve it on raw images.
"""
import os
import tempfile
import numpy as np
from sklearn.datasets import make_moons, load_digits


def main() -> None:
    # ---- 1. How many clusters? ----
    X, y = blobs_data()
    inertia = inertia_curve(X, [1, 2, 3, 4, 5, 6, 7, 8])
    sil = silhouette_curve(X, [2, 3, 4, 5, 6, 7, 8])
    print("inertia by k:   " + "  ".join(f"{k}:{v:,.0f}" for k, v in inertia.items()))
    print("silhouette by k:" + "  ".join(f"{k}:{v:.3f}" for k, v in sil.items()))
    print(f"silhouette picks k = {best_k_by_silhouette(sil)}; the data was generated with 5 blobs "
          f"(three of them tightly packed, which is why 4 looks good too)")

    # ---- 2. Density-based clustering ----
    Xm, ym = make_moons(n_samples=1000, noise=0.05, random_state=42)
    for eps in (0.05, 0.2):
        s = dbscan_summary(fit_dbscan(Xm, eps=eps))
        print(f"DBSCAN eps={eps}: {s['n_clusters']} clusters, {s['n_noise']} noise points, {s['n_core']} core samples")
    db = fit_dbscan(Xm, eps=0.2)
    new_points = np.array([[-0.5, 0.0], [0.0, 0.5], [1.0, -0.1], [2.0, 1.0]])
    print(f"new points assigned via core-sample KNN: {dbscan_predict(db, new_points).tolist()}")

    # ---- 3. Mixtures, BIC and anomalies ----
    bic = bic_curve(X, [2, 3, 4, 5, 6, 7])
    best = min(bic, key=bic.get)
    print(f"BIC by components: " + "  ".join(f"{k}:{v:,.0f}" for k, v in bic.items()) + f"  -> {best} components")
    gm = fit_gmm(X, best)
    flagged = flag_anomalies(gm, X, contamination=0.04)
    print(f"anomalies at 4% contamination: {int(flagged.sum())} of {len(X)} points flagged by low density")

    # ---- 4. Fifty labels, propagated ----
    Xtr, Xte, ytr, yte = digits_data()
    kmeans, rep_idx = representative_digits(Xtr, k=50)
    print(f"\ndigits, 50-label budget:")
    print(f"  50 random labels             -> test accuracy {baseline_50_random(Xtr, ytr, Xte, yte):.3f}")
    print(f"  50 representative labels     -> test accuracy {train_on_representatives(Xtr, ytr, rep_idx, Xte, yte):.3f}")
    prop = propagate_and_train(Xtr, ytr, kmeans, rep_idx, Xte, yte, percentile=20)
    print(f"  propagated to {prop['n_propagated']} points ({prop['label_accuracy']:.1%} of them correctly) -> test accuracy {prop['test_accuracy']:.3f}")
    print(f"  every label ({len(Xtr)})        -> test accuracy {baseline_50_random(Xtr, ytr, Xte, yte, n_labeled=len(Xtr)):.3f}")

    # ---- 5. Image segmentation ----
    img = synthetic_image(48)
    seg = segment_colors(img, k=4)
    print(f"\ncolor segmentation: {len(np.unique(img.reshape(-1, 3), axis=0))} colors -> {len(np.unique(seg.reshape(-1, 3), axis=0))}")

    # ---- 6. Ship the digit clusterer ----
    path = os.path.join(tempfile.gettempdir(), "digit_clusters.pkl")
    bundle = save_and_reload_clusterer(kmeans, ytr[rep_idx], path)
    digits = load_digits()
    preds = predict_digit_labels(bundle, digits.images[:8])
    print(f"served on 8 raw images: {preds} (truth {digits.target[:8].tolist()})")


if __name__ == "__main__":
    main()

