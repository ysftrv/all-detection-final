"""
baseline.py — Otsu segmentasyon + morfoloji + el-yapımı özellik + Logistic Regression.

Pipeline:
    Görüntü yolu → gri → Otsu eşik → morfolojik açma/kapama → en büyük bileşen
    → şekil (7) + renk (6) + GLCM doku (4) = 17 özellik → StandardScaler → LR

Dışa aktarılanlar:
    FEATURE_DIM        : 17
    extract_features(image_path, config)  → np.ndarray (17,)
    BaselineClassifier : fit / fit_from_X / predict_from_X / predict_proba_from_X / save / load
"""

import logging
from pathlib import Path

import cv2
import joblib
import numpy as np
from scipy.ndimage import label as nd_label
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import regionprops
from skimage.morphology import closing as morph_closing, disk, opening as morph_opening
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)

FEATURE_DIM  = 17
_GLCM_DISTS  = [1, 3]
_GLCM_ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
_GLCM_PROPS  = ["contrast", "homogeneity", "energy", "correlation"]


def _largest_component(binary: np.ndarray) -> np.ndarray:
    """İkili maskeden yalnızca en büyük bağlı bileşeni döndür."""
    labeled, n = nd_label(binary)
    if n == 0:
        return binary
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    return (labeled == sizes.argmax()).astype(bool)


def extract_features(image_path: str | Path, config=None) -> np.ndarray:
    """
    Tek görüntüden 17 boyutlu el-yapımı özellik vektörü çıkar.

    Özellik grupları:
        [0:7]  — şekil: alan, çevre, eccentricity, solidity, extent, eksen oranı, dairesellik
        [7:13] — renk: R/G/B kanal ort + std (maske içi)
        [13:17]— GLCM doku: contrast, homogeneity, energy, correlation

    Girdi : image_path — görüntü dosyası yolu
    Çıktı : np.ndarray shape (17,) float32
    """
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        log.warning("Görüntü okunamadı, sıfır vektör: %s", image_path)
        return np.zeros(FEATURE_DIM, dtype=np.float32)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # --- Segmentasyon: Otsu + morfolojik temizleme ---
    _, binary = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary = binary.astype(bool)
    selem  = disk(3)
    binary = morph_opening(binary.astype(np.uint8), selem).astype(bool)
    binary = morph_closing(binary.astype(np.uint8), selem).astype(bool)
    mask   = _largest_component(binary)

    # --- Şekil özellikleri (skimage.regionprops) ---
    props_list = regionprops(mask.astype(np.uint8))
    if props_list:
        p           = props_list[0]
        area        = float(p.area)
        perim       = max(float(p.perimeter), 1e-6)
        eccentr     = float(p.eccentricity)
        solidity    = float(p.solidity)
        extent      = float(p.extent)
        major_ax    = float(p.major_axis_length)
        minor_ax    = max(float(p.minor_axis_length), 1e-6)
        ax_ratio    = major_ax / minor_ax
        circularity = 4 * np.pi * area / (perim ** 2)
    else:
        area = perim = eccentr = solidity = extent = ax_ratio = circularity = 0.0

    shape_feats = np.array(
        [area, perim, eccentr, solidity, extent, ax_ratio, circularity],
        dtype=np.float32,
    )

    # --- Renk istatistikleri (R/G/B kanal ort + std, maske içinde) ---
    if mask.sum() > 0:
        color_feats = []
        for ch in range(3):
            vals = img_rgb[:, :, ch][mask]
            color_feats.extend([float(vals.mean()), float(vals.std())])
    else:
        color_feats = [0.0] * 6
    color_feats = np.array(color_feats, dtype=np.float32)

    # --- GLCM doku özellikleri ---
    gray_q = (gray // 16).astype(np.uint8)  # 16 gri seviyesine indirge
    glcm   = graycomatrix(
        gray_q, distances=_GLCM_DISTS, angles=_GLCM_ANGLES,
        levels=16, symmetric=True, normed=True,
    )
    glcm_feats = np.array(
        [float(graycoprops(glcm, prop).mean()) for prop in _GLCM_PROPS],
        dtype=np.float32,
    )

    return np.concatenate([shape_feats, color_feats, glcm_feats])


class BaselineClassifier:
    """
    Otsu + morfoloji + el-yapımı özellik → StandardScaler + LogisticRegression.

    Kullanım (ham yollarla):
        clf = BaselineClassifier(config); clf.fit(paths, labels); clf.predict_proba(paths)

    Kullanım (önceden çıkarılmış özelliklerle — cache için):
        clf.fit_from_X(X, y); clf.predict_proba_from_X(X)
    """

    def __init__(self, config=None):
        self.config = config or {}
        p2   = self.config.get("phase2", {})
        seed = p2.get("seed", self.config.get("seed", 42))
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                solver="lbfgs",
                random_state=seed,
            )),
        ])

    # --- Ham görüntü yoluyla ---

    def fit(self, image_paths, labels) -> "BaselineClassifier":
        """Görüntü yollarından özellik çıkar, pipeline'ı eğit."""
        X = np.vstack([extract_features(p, self.config) for p in image_paths]).astype(np.float32)
        return self.fit_from_X(X, labels)

    def predict(self, image_paths) -> np.ndarray:
        X = np.vstack([extract_features(p, self.config) for p in image_paths]).astype(np.float32)
        return self.predict_from_X(X)

    def predict_proba(self, image_paths) -> np.ndarray:
        X = np.vstack([extract_features(p, self.config) for p in image_paths]).astype(np.float32)
        return self.predict_proba_from_X(X)

    # --- Önceden hesaplanmış özellik matrisiyle (cache için) ---

    def fit_from_X(self, X, labels) -> "BaselineClassifier":
        """Özellik matrisi üzerinde doğrudan eğit."""
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
        log.info("Baseline model kaydedildi: %s", path)

    @classmethod
    def load(cls, path: str | Path, config=None) -> "BaselineClassifier":
        obj = cls.__new__(cls)
        obj.config   = config or {}
        obj.pipeline = joblib.load(path)
        return obj
