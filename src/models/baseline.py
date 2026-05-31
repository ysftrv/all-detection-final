"""
baseline.py — Katman 1: Klasik segmentasyon + el-yapımı özellik + Lojistik Regresyon.

Pipeline:
    Görüntü → Otsu Eşikleme (segmentasyon) → Özellik Çıkarımı → LogisticRegression

Çıkarılan özellikler (planlanan):
    - Hücre/çekirdek alanı ve çevre (skimage.measure.regionprops)
    - Nükleer-Sitoplazmik (N/C) oran
    - Renk kanalı ortalamaları (RGB & HSV)
    - Şekil tanımlayıcıları: dairesellik, eksantriklik

Dışa aktarılanlar:
    BaselineClassifier   : fit() / predict() / predict_proba() API'siyle sarılmış pipeline
    extract_features()   : tek görüntü → özellik vektörü (numpy array)
"""

# TODO (Faz 3):
#   extract_features(image_path, cfg):
#     1. Görüntüyü oku (OpenCV veya PIL).
#     2. Gri tonlamaya çevir → skimage.filters.threshold_otsu ile ikili maske al.
#     3. Morfolojik açma/kapama ile gürültü temizle.
#     4. skimage.measure.label + regionprops ile en büyük bileşeni çekirdek kabul et.
#     5. Alan, çevre, dairesellik = 4π·alan/çevre², eksantriklik hesapla.
#     6. Orijinal görüntüde N/C oranı için sitoplazmayı tahmin et (basit heuristic).
#     7. Renk histogramı veya ortalamalarını RGB ve HSV uzayında çıkar.
#     8. Tüm özellikler numpy vektöründe birleştir.
#
#   BaselineClassifier.fit(X, y):
#     - StandardScaler + LogisticRegression(C=1.0, max_iter=1000) pipeline kur.
#     - sklearn Pipeline ile sar.
#
#   BaselineClassifier.save(path) / .load(path):
#     - joblib.dump / joblib.load ile modeli kaydet/yükle.


class BaselineClassifier:
    def __init__(self, config=None):
        raise NotImplementedError("TODO (Faz 3): BaselineClassifier implemente edilmedi.")

    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def predict_proba(self, X):
        raise NotImplementedError

    def save(self, path: str):
        raise NotImplementedError

    @classmethod
    def load(cls, path: str):
        raise NotImplementedError


def extract_features(image_path: str, config=None):
    """Tek bir görüntü için özellik vektörü çıkar. → np.ndarray shape (n_features,)"""
    raise NotImplementedError("TODO (Faz 3): extract_features implemente edilmedi.")
