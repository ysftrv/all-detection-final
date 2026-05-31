# Donanım Kaydı

Bu dosya, eğitimin koşturulduğu donanım ortamını belgeler.
`scripts/run_test.py` ve `src/train.py` çalıştırıldığında bu bilgiler
otomatik olarak `experiments/logs/hardware_info.json`'a da yazılır.

---

## Eğitim Ortamı (Faz 4'ten sonra doldurul)

| Parametre        | Değer |
|------------------|-------|
| Platform         | _(otomatik: `platform.platform()`)_ |
| Python sürümü    | _(otomatik)_ |
| PyTorch sürümü   | _(otomatik)_ |
| GPU adı          | _(otomatik: `torch.cuda.get_device_name(0)`)_ |
| GPU belleği (GB) | _(otomatik)_ |
| CUDA sürümü      | _(otomatik)_ |
| CPU              | _(otomatik)_ |
| RAM (GB)         | _(otomatik)_ |
| Eğitim tarihi    | _(otomatik)_ |

---

## Google Colab Notları

- Çalışma zamanı türü: **GPU** (Runtime → Change runtime type → GPU)
- Önerilen tip: T4 veya A100 (ücretsiz T4 yeterli olacaktır)
- Colab oturumu düşerse ağırlıklar `experiments/weights/` altında korunur
  (Google Drive mount edilmişse).

---

## Yerel Geliştirme Ortamı

| Parametre        | Değer |
|------------------|-------|
| İşletim sistemi  | Windows 11 |
| Python sürümü    | _(doldurul)_ |
| CPU              | _(doldurul)_ |
| RAM (GB)         | _(doldurul)_ |
| GPU (varsa)      | _(doldurul)_ |
