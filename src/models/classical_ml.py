"""
classical_ml.py — Katman 2: HOG/LBP + SVM ile görüntü sınıflandırması.

Özellik çıkarıcılar:
    HOG  — skimage.feature.hog (histogram of oriented gradients)
    LBP  — skimage.feature.local_binary_pattern (local binary patterns)

Sınıflandırıcı:
    sklearn.svm.SVC (kernel='rbf', probability=True) — GridSearchCV ile C, gamma ayarı

Dışa aktarılanlar:
    ClassicalMLClassifier  : HOG+LBP özellik çıkarımı + SVM pipeline
    build_feature_pipeline(): özellik çıkarım adımlarını sklearn Pipeline olarak döndür
"""

# TODO (Faz 3):
#   build_feature_pipeline(config):
#     1. Görüntüyü config.image_size × image_size'a yeniden boyutlandır.
#     2. HOG çıkar: orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2).
#     3. LBP çıkar: radius=3, n_points=24, method='uniform' → histogram normalizasyonu.
#     4. HOG ve LBP vektörlerini birleştir → StandardScaler.
#
#   ClassicalMLClassifier.fit(image_paths, labels):
#     - Tüm görüntülerden özellik matrisi oluştur (tqdm progress bar).
#     - GridSearchCV({C: [0.1, 1, 10], gamma: ['scale', 'auto']}, cv=3) ile SVM.
#     - En iyi parametreleri experiments/logs/svm_best_params.json'a kaydet.
#
#   ClassicalMLClassifier.save(path) / .load(path):
#     - joblib ile kaydet/yükle.


class ClassicalMLClassifier:
    def __init__(self, config=None):
        raise NotImplementedError("TODO (Faz 3): ClassicalMLClassifier implemente edilmedi.")

    def fit(self, image_paths, labels):
        raise NotImplementedError

    def predict(self, image_paths):
        raise NotImplementedError

    def predict_proba(self, image_paths):
        raise NotImplementedError

    def save(self, path: str):
        raise NotImplementedError

    @classmethod
    def load(cls, path: str):
        raise NotImplementedError


def build_feature_pipeline(config=None):
    """sklearn Pipeline döndür: resize → HOG+LBP → StandardScaler."""
    raise NotImplementedError("TODO (Faz 3): build_feature_pipeline implemente edilmedi.")
