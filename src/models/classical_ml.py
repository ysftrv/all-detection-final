"""
classical_ml.py — HOG + LBP özellik çıkarımı + SVM sınıflandırması.

Pipeline:
    Görüntü → 128×128 gri → HOG (9 yönlü, 8×8 hücre, 2×2 blok)
             → LBP (radius=3, uniform, histogram normalize)
             → concat → StandardScaler → SVM (rbf varsayılan / linear fallback)

Dışa aktarılanlar:
    extract_hog_lbp(image_path, img_size)  → np.ndarray
    ClassicalMLClassifier : fit / fit_from_X / predict_from_X / predict_proba_from_X / save / load
"""

import logging
from pathlib import Path

import cv2
import joblib
import numpy as np
from skimage.feature import hog, local_binary_pattern
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

log = logging.getLogger(__name__)

_IMG_SIZE   = 128
_HOG_PARAMS = dict(
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    channel_axis=None,
    feature_vector=True,
)
_LBP_RADIUS  = 3
_LBP_NPOINTS = 24          # 8 × radius
_LBP_METHOD  = "uniform"
_LBP_BINS    = _LBP_NPOINTS + 2  # uniform histogramı için


def extract_hog_lbp(image_path: str | Path, img_size: int = _IMG_SIZE) -> np.ndarray:
    """
    Tek görüntüden HOG + LBP özellik vektörü çıkar.

    Girdi : image_path, img_size (kare boyut, piksel)
    Çıktı : np.ndarray float32 — concat([hog_feat, lbp_hist])
    """
    img = cv2.imread(str(image_path))
    if img is None:
        log.warning("Görüntü okunamadı, sıfır vektör: %s", image_path)
        cells = img_size // 8 - 1
        hog_dim = cells * cells * 36
        return np.zeros(hog_dim + _LBP_BINS, dtype=np.float32)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (img_size, img_size), interpolation=cv2.INTER_AREA)
    gray_f = gray.astype(np.float32) / 255.0

    hog_feat = hog(gray_f, **_HOG_PARAMS).astype(np.float32)

    lbp      = local_binary_pattern(gray, _LBP_NPOINTS, _LBP_RADIUS, method=_LBP_METHOD)
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=_LBP_BINS,
                                range=(0, _LBP_BINS), density=True)
    lbp_feat = lbp_hist.astype(np.float32)

    return np.concatenate([hog_feat, lbp_feat])


def _build_svm(kernel: str, seed: int):
    """kernel='rbf' → SVC; 'linear' → CalibratedClassifierCV(LinearSVC) ile olasılık."""
    if kernel == "linear":
        return CalibratedClassifierCV(
            LinearSVC(class_weight="balanced", max_iter=5000, random_state=seed)
        )
    return SVC(
        kernel="rbf",
        class_weight="balanced",
        probability=True,
        random_state=seed,
        cache_size=500,
    )


class ClassicalMLClassifier:
    """
    HOG + LBP → StandardScaler + SVM.

    config.phase2.svm_kernel : 'rbf' (varsayılan) | 'linear' (7000+ örnek için hız)
    """

    def __init__(self, config=None):
        self.config   = config or {}
        p2            = self.config.get("phase2", {})
        seed          = p2.get("seed", self.config.get("seed", 42))
        self._img_size = p2.get("image_size", _IMG_SIZE)
        kernel        = p2.get("svm_kernel", "rbf")
        self._kernel  = kernel
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    _build_svm(kernel, seed)),
        ])

    # --- Ham görüntü yoluyla ---

    def fit(self, image_paths, labels) -> "ClassicalMLClassifier":
        """Görüntü yollarından özellik çıkar, SVM pipeline'ı eğit."""
        X = np.vstack(
            [extract_hog_lbp(p, self._img_size) for p in image_paths]
        ).astype(np.float32)
        return self.fit_from_X(X, labels)

    def predict(self, image_paths) -> np.ndarray:
        X = np.vstack([extract_hog_lbp(p, self._img_size) for p in image_paths]).astype(np.float32)
        return self.predict_from_X(X)

    def predict_proba(self, image_paths) -> np.ndarray:
        X = np.vstack([extract_hog_lbp(p, self._img_size) for p in image_paths]).astype(np.float32)
        return self.predict_proba_from_X(X)

    # --- Önceden hesaplanmış özellik matrisiyle ---

    def fit_from_X(self, X, labels) -> "ClassicalMLClassifier":
        """Özellik matrisi üzerinde doğrudan eğit."""
        log.info("ClassicalML SVM (%s) eğitiliyor — özellik boyutu: %d", self._kernel, X.shape[1])
        self.pipeline.fit(X, labels)
        return self

    def predict_from_X(self, X) -> np.ndarray:
        return self.pipeline.predict(X)

    def predict_proba_from_X(self, X) -> np.ndarray:
        return self.pipeline.predict_proba(X)

    # --- Kaydet / yükle ---

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, path)
        log.info("ClassicalML model kaydedildi: %s", path)

    @classmethod
    def load(cls, path: str | Path, config=None) -> "ClassicalMLClassifier":
        obj = cls.__new__(cls)
        obj.config    = config or {}
        obj._kernel   = "rbf"
        obj._img_size = _IMG_SIZE
        obj.pipeline  = joblib.load(path)
        return obj
