# Rejestr eksperymentów

Każdy trening = jeden wiersz. Szczegóły (komenda, commit kodu, GPU, parametry zbioru, krzywe uczenia)
są w folderze runu na Drive: `MyDrive/rs_project/runs/<run>/` (`run_info.json`, `log.csv`, `results.json`, `eval.json`).
Metryki na zbiorze walidacyjnym (syntetycznym), 256×256. PSNR/SSIM/EPE w obszarze widocznym w kamerze A;
`PSNR_b` = brzegi widoczne tylko w kamerze B.

## Zbiory danych

| zbiór | data | źródło | parametry | rozmiar | gdzie |
|---|---|---|---|---|---|
| rs_dataset_256 | 2026-10-13 | DIV2K train HR (800 zdjęć), 20 próbek/zdjęcie | `configs/config_webcam.json` (wartości zgadywane, przed pomiarami kamer) | ~14,4k train / ~1,6k val (podział po zdjęciach, 10%) | `MyDrive/rs_dataset_256.zip` (o ile zapisany w kroku 2b) |
| rs_video_x4k | 2026-10-24 | X4K1000FPS train (nagrania 1000 fps, 4K), `src/synth_video.py --augment` | odczyt 10–30 ms, sync ±2 ms, baza 0–8 px, wycinek 0,5–1,0 krótszego boku | (uzupełnić: train / val, podział po scenach 10%) | `MyDrive/rs_project/datasets/rs_video_x4k.zip` |

## Treningi

| run | data | etap / model | zbiór | notatnik | GPU | epoki | PSNR [dB] | SSIM | EPE [px] | PSNR_b | ms/kl. | parametry | uwagi |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (bez korekcji) | – | – | rs_dataset_256 | – | – | – | 19.66 | 0.527 | – | – | – | – | punkt odniesienia |
| 2026-10-13_motion_dual | 2026-10-13 | 1 / MotionNet, kamery A+B | rs_dataset_256 | archiwum/train_colab_2026-10-13 | T4 | 30 | **28.51** | **0.874** | **0.57** | – | 18.4 | 3.58 M | model główny etapu 1 |
| 2026-10-13_motion_single | 2026-10-13 | 1 / MotionNet, tylko A | rs_dataset_256 | archiwum/train_colab_2026-10-13 | T4 | 30 | 19.90 | 0.544 | 3.98 | – | 6.6 | 3.48 M | ablacja: bez kamery B prawie brak poprawy |
| 2026-10-13_flow_dual | 2026-10-13 | 1 / U-Net gęste pole, A+B, bez korelacji | rs_dataset_256 | archiwum/train_colab_2026-10-13 | T4 | 30 | 20.45 | 0.588 | 3.15 | – | 18.3 | 2.72 M | porównanie: bez geometrii prawie się nie uczy |
| (bez korekcji) | – | – | rs_video_x4k | – | – | – | 26.42 | 0.795 | – | – | – | – | punkt odniesienia dla zbioru z nagrań |
| 2026-10-24_raft_zeroshot | 2026-10-24 | RAFT (gotowe wagi) + geometria RS, bez treningu | rs_video_x4k | 04_raft_geom | Colab Pro | 0 | 32.86 | 0.936 | – | – | (24.1)* | 5.3 M | ruch obiektów; +6,4 dB bez żadnego uczenia |
| 2026-10-24_raft_geom | 2026-10-24 | RAFT dostrojony (L1 obrazu) + geometria RS | rs_video_x4k | 04_raft_geom | Colab Pro | 10 | **34.30** | **0.941** | – | – | (18.2)* | 5.3 M | lr 2e-5, bs 8, 108 s/epokę; nasycenie od ~5. epoki |
| *_refine_dual | planowany | 2 / MotionNet (zamrożony) + U-Net | rs_dataset_256 | 03_refine_unet | Colab Pro | 15 | | | | | | | |

\* `ms_per_pair` w `train_raft.py` = czas całej ewaluacji / liczba par (z wczytywaniem danych i metrykami), nie czysty czas sieci.

## Wnioski na bieżąco
- 2026-10-13: kamera B jest kluczowa (+8,9 dB z B vs +0,2 dB bez niej). Pozostały błąd MotionNet to głównie
  brzegi, które wyszły poza kadr A (rysunek `docs/rysunki/analiza_bledu_motion_dual.png`), stąd etap 2.
- 2026-10-24: przejście na ruch obiektów (cel: wyprostowany wiatrak). Zbiór z nagrań 1000 fps (`rs_video_x4k`),
  model RAFT + geometria (`src/raft_geom.py`). Test geometrii z dokładnym przepływem: ~15 → ~36 dB.
- 2026-10-24: RAFT + geometria: 26,42 → 32,86 dB bez treningu, 34,30 dB po dostrojeniu (+1,4 dB, nasycenie po ~5 epokach).
  Wejście ma już 26,4 dB, bo w wielu próbkach X4K ruch jest mały: średnia zaniża efekt na próbkach z silnym RS.
  Wyników nie porównywać wprost z MotionNet (inny zbiór: DIV2K, ruch kamery, wejście 19,7 dB).
