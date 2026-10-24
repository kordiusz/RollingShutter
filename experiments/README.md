# Rejestr eksperymentów

Każdy trening = jeden wiersz. Szczegóły (komenda, commit kodu, GPU, parametry zbioru, krzywe uczenia)
są w folderze runu na Drive: `MyDrive/rs_project/runs/<run>/` (`run_info.json`, `log.csv`, `results.json`, `eval.json`).
Metryki na zbiorze walidacyjnym (syntetycznym), 256×256. PSNR/SSIM/EPE w obszarze widocznym w kamerze A;
`PSNR_b` = brzegi widoczne tylko w kamerze B.

## Zbiory danych

| zbiór | data | źródło | parametry | rozmiar | gdzie |
|---|---|---|---|---|---|
| rs_dataset_256 | 2026-10-13 | DIV2K train HR | `configs/config_webcam.json` (wartości zgadywane, przed pomiarami kamer) | **1858 train / 236 val** (mały, najpewniej zapis po szybkim teście) | `MyDrive/rs_dataset_256.zip` |
| rs_dataset_256_full | planowany | DIV2K train HR (800 zdjęć), 20 próbek/zdjęcie | `configs/config_webcam.json` | ~14,4k train / ~1,6k val | `MyDrive/rs_project/datasets/rs_dataset_256_full.zip` |

## Treningi

| run | data | etap / model | zbiór | notatnik | GPU | epoki | PSNR [dB] | SSIM | EPE [px] | PSNR_b | ms/kl. | parametry | uwagi |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (bez korekcji) | – | – | rs_dataset_256 | – | – | – | 19.66 | 0.527 | – | – | – | – | punkt odniesienia |
| 2026-10-13_motion_dual | 2026-10-13 | 1 / MotionNet, kamery A+B | rs_dataset_256 | archiwum/train_colab_2026-10-13 | T4 | 30 | **28.51** | **0.874** | **0.57** | – | 18.4 | 3.58 M | model główny etapu 1 |
| 2026-10-13_motion_single | 2026-10-13 | 1 / MotionNet, tylko A | rs_dataset_256 | archiwum/train_colab_2026-10-13 | T4 | 30 | 19.90 | 0.544 | 3.98 | – | 6.6 | 3.48 M | ablacja: bez kamery B prawie brak poprawy |
| 2026-10-13_flow_dual | 2026-10-13 | 1 / U-Net gęste pole, A+B, bez korelacji | rs_dataset_256 | archiwum/train_colab_2026-10-13 | T4 | 30 | 20.45 | 0.588 | 3.15 | – | 18.3 | 2.72 M | porównanie: bez geometrii prawie się nie uczy |
| *_refine_dual | planowany | 2 / MotionNet (zamrożony) + U-Net | rs_dataset_256_full | 03_refine_unet | Colab Pro | 15 | | | | | | | |

## Wnioski na bieżąco
- 2026-10-24: zbiór z 13.10 okazał się mały (2,1 tys. par), więc wyniki etapu 1 są z małego zbioru (walidacja 236 próbek).
  Pełny zbiór: `rs_dataset_256_full`.
- 2026-10-13: kamera B jest kluczowa (+8,9 dB z B vs +0,2 dB bez niej). Pozostały błąd MotionNet to głównie
  brzegi, które wyszły poza kadr A (rysunek `docs/rysunki/analiza_bledu_motion_dual.png`), stąd etap 2.
