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

Etap 1 (MotionNet, `rs_dataset_256`): kod i notatniki usunięte 11.11, opis i wyniki w [`docs/etap1_motionnet.md`](../docs/etap1_motionnet.md), kod w commicie `86afe79`.

| run | data | etap / model | zbiór | notatnik | GPU | epoki | PSNR [dB] | SSIM | EPE [px] | PSNR_b | ms/kl. | parametry | uwagi |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (bez korekcji) | – | – | rs_dataset_256 | – | – | – | 19.66 | 0.527 | – | – | – | – | punkt odniesienia |
| 2026-10-13_motion_dual | 2026-10-13 | 1 / MotionNet, kamery A+B | rs_dataset_256 | (usunięty, etap 1) | T4 | 30 | **28.51** | **0.874** | **0.57** | – | 18.4 | 3.58 M | model główny etapu 1 |
| 2026-10-13_motion_single | 2026-10-13 | 1 / MotionNet, tylko A | rs_dataset_256 | (usunięty, etap 1) | T4 | 30 | 19.90 | 0.544 | 3.98 | – | 6.6 | 3.48 M | ablacja: bez kamery B prawie brak poprawy |
| 2026-10-13_flow_dual | 2026-10-13 | 1 / U-Net gęste pole, A+B, bez korelacji | rs_dataset_256 | (usunięty, etap 1) | T4 | 30 | 20.45 | 0.588 | 3.15 | – | 18.3 | 2.72 M | porównanie: bez geometrii prawie się nie uczy |
| (bez korekcji) | – | – | rs_video_x4k | – | – | – | 26.42 | 0.795 | – | – | – | – | punkt odniesienia dla zbioru z nagrań |
| 2026-10-24_raft_zeroshot | 2026-10-24 | RAFT (gotowe wagi) + geometria RS, bez treningu | rs_video_x4k | 04_raft_geom | Colab Pro | 0 | 32.86 | 0.936 | – | – | (24.1)* | 5.3 M | ruch obiektów; +6,4 dB bez żadnego uczenia |
| 2026-10-24_raft_geom | 2026-10-24 | RAFT dostrojony (L1 obrazu) + geometria RS | rs_video_x4k | 04_raft_geom | Colab Pro | 10 | **34.30** | **0.941** | – | – | (18.2)* | 5.3 M | lr 2e-5, bs 8, 108 s/epokę; nasycenie od ~5. epoki |
| 2026-11-03_unet_plain | 2026-11-03 | zwykły U-Net (A, B, czasy odczytu → przepływ + residuum), bez RAFT | rs_video_x4k | 06_unet_plain | L4 | 30 | 28.45 | 0.845 | – | – | (3.9)* | 2.7 M | punkt odniesienia: bez RAFT i geometrii +2 dB |
| (eval) raft_zeroshot, geometria v2 | 2026-11-03 | RAFT (gotowe wagi) + geometria v2 (krawędzie obiektów, dziury z kamery B) | rs_video_x4k | 07_demo_v2 | L4 | 0 | 33.25 | 0.937 | – | – | (32.0)* | 5.3 M | RAFT liczony A→B i B→A |
| (eval) 2026-10-24_raft_geom, geometria v2 | 2026-11-03 | RAFT dostrojony z v1 + geometria v2 | rs_video_x4k | 07_demo_v2 | L4 | – | 34.39 | 0.941 | – | – | (32.0)* | 5.3 M | |
| 2026-11-03_raft_geom_v2 | 2026-11-03 | RAFT dostrajany dalej z geometrią v2 (`--init` z 2026-10-24_raft_geom) | rs_video_x4k | 07_demo_v2 | L4 | 5 | 34.46 | 0.941 | – | – | – | 5.3 M | lr 1e-5, bs 8, 219 s/epokę; prawie bez zysku (+0,07 dB) |
| 2026-11-08_raft_unet | 2026-11-08 | RAFT (zamrożony, wagi z 2026-11-03_raft_geom_v2) + geometria v1 + U-Net refine | rs_video_x4k | 05_unet | L4 | 30 | 35.68 | 0.955 | – | – | 51.6 | 8.0 M (2.7 M uczone) | lr 3e-4, bs 8, 56 s/epokę; sam etap RAFT + geometria w tym modelu 34.36; psnr_full 35.24; kod `d70c697` |
| 2026-11-08_raft_unet_ft | 2026-11-08 | jw., wspólne dostrajanie RAFT + U-Net (`--init` z 2026-11-08_raft_unet) | rs_video_x4k | 05_unet | L4 | 10 | **35.81** | **0.955** | – | – | 52.6 | 8.0 M | RAFT lr 1e-5, U-Net lr 1e-4, 126 s/epokę; +0,14 dB; psnr_full 35.37; waga B w masce 0.41 |
| 2026-11-11_unet_plain_long | 2026-11-11 | zwykły U-Net od zera, jak 2026-11-03_unet_plain, ale 200 epok | rs_video_x4k | 08_synteza | L4 | 200 | 31.05 | 0.895 | – | – | 3.8 | 2.7 M | psnr_full 30.94; najlepsza ep. 149, plateau od ok. ep. 100 (30,9–31,05); 21 s/ep.; strata treningowa 0,0113 (niższa niż corr przy tym samym val) |
| 2026-11-11_ae | 2026-11-11 | autoenkoder bez skipów (latent 16×16×512), synteza obrazu wprost | rs_video_x4k | 08_synteza | L4 | 60 | 23.27 | 0.681 | – | – | 3.9 | 10.2 M | `psnr` = cały obraz; **gorzej niż wejście (26,35)**: wąskie gardło rozmywa obraz; 19 s/ep. |
| 2026-11-11_corr | 2026-11-11 | syjamski enkoder + korelacja A↔B + dekoder ze skipami, synteza wprost (A + poprawka) | rs_video_x4k | 08_synteza | L4 | 80 | 31.06 | 0.887 | – | – | 10.7 | 6.2 M | `psnr` = cały obraz (wejście 26,35); szybki wzrost w ep. 26–45, od ep. 60 +0,1 dB (koniec OneCycle); bez przeuczenia |
| 2026-11-11_unet_plain_single | 2026-11-11 | ablacja: zwykły U-Net, kamera B zastąpiona kopią A (`--single`) | rs_video_x4k | 09_ablacje | T4 | 200 | 26.53 | 0.808 | – | – | – | 2.7 M | psnr_full 26.51 (wejście 26,35): **bez kamery B prawie zero poprawy** (z B, 2026-11-11_unet_plain_long: 30.94); najlepsza ep. 12, potem bez postępu; kod `e47f1dd` |
| *_corr200 | porzucony | jak *_corr, 200 epok (równe epoki w głównym porównaniu architektur) | rs_video_x4k | 09_ablacje (albo kopia 08) | L4 | 200 | | | – | | | 6.2 M | `corr` 80 ep. już na plateau (+0,1 dB w ostatnich 20 ep.), oszczędność ~2 h |
| 2026-11-11_corrgeom | 2026-11-11 | sieć z `corr` (od zera) przewiduje przesunięcie A→B + geometria v1 | rs_video_x4k | 09_ablacje | T4 | 77/200 | – | – | – | – | – | 6.2 M | **nieudany, do wyrzucenia**: strata tylko na obszarze z pikselem źródłowym; w ep. 7 przepływ się rozjechał (poprawne 9% obrazu, psnr_full 11.6), dalej żaden krok optymalizatora (gradienty inf/NaN), `best.pt` = ep. 7; przerwany przez Colab w ep. 77; kod `e47f1dd` |
| 2026-11-14_corrgeom | 2026-11-14 | jw., przepływ ograniczony do ±64 px, wynik i strata na całym obrazie | rs_video_x4k | 09_ablacje | L4 | 9/80 | – | – | – | – | – | 6.2 M | **nieudany**: do ep. 8 bez nauki (26,0–26,4, ep. 4: 24,2), w ep. 9 przepływ nasycony na ±64 px (geometria 0% obrazu), ponad połowa kroków z gradientami inf/NaN, automatyczny STOP; przyczyna: niestabilność sieci `corr` (bez normalizacji) po szczycie lr 3e-4, w fp16 kończy się inf/NaN; kod `2bbee2f` |
| 2026-11-14_corr_single (1. próba) | 2026-11-14 | `corr`, kamera B zastąpiona kopią A | rs_video_x4k | 09_ablacje | L4 | 11/80 | – | – | – | – | – | 6.2 M | **nieudany**: 26,35 → 26,53 do ep. 9, w ep. 10 NaN (niestabilność, fp16), automatyczny STOP po ep. 11; kod `f72552e` |
| 2026-11-14_corr_single (2. próba) | 2026-11-14 | jw., sieć `corr` w bf16 | rs_video_x4k | 09_ablacje | L4 | 26/80 | – | – | – | – | – | 6.2 M | **nieudany**: do ep. 6 26,35 → 26,50, w ep. 7 wyjście nasycone na 0 w `clamp(0, 1)` (czarny obraz, 7,25 dB), bez gradientu, bez NaN, więc STOP nie zadziałał; ta sama niestabilność co w 1. próbie, bf16 zmienił tylko objaw; kod `12ee613`. Ablacja kamery B: para zwykłego U-Netu (200 ep.) + analiza zysku z kamery B w funkcji abs(t_B − t_A) (notatnik 09, bez treningu) |
| *_corrgeom | odłożony | jw., po poprawkach 14.11 (sieć `corr` w bf16 / fp32 zamiast fp16): przepływ ograniczony do ±64 px, wynik i strata na całym obrazie (bez źródła: nieskorygowane A) | rs_video_x4k | 09_ablacje | L4 | 80 | | | – | | | 6.2 M | para: 2026-11-11_corr (80 ep.); szczebel między `corr` a RAFT + geometrią v1 (32,86 bez uczenia / 34,30) |
| *_unet_plain_lr1e-4, *_lr1e-3 | opcjonalny | kontrola: zwykły U-Net z innym lr (domyślnie 3e-4) | rs_video_x4k | 09_ablacje | L4 | 200 | | | – | | | 2.7 M | czy wynik zależy od lr |
| *_corr_w64 | opcjonalny | kontrola: `corr` z width 64 (~4× wag) | rs_video_x4k | 09_ablacje | L4 | 200 | | | – | | | ~25 M | czy sufit ~30 dB to rozmiar sieci, czy brak geometrii |
| *_refine_dual | porzucony | 2 / MotionNet (zamrożony) + U-Net | rs_dataset_256 | (usunięty, etap 1) | – | – | | | | | | | zastąpiony przez RAFT + geometria + U-Net (05) |

\* `ms_per_pair` w `train_raft.py` = czas całej ewaluacji / liczba par (z wczytywaniem danych i metrykami), nie czysty czas sieci. W `train_unet.py` (runy `*_raft_unet*`) `ms_per_pair` to czysty czas sieci dla jednej pary.

## Wnioski na bieżąco
- 2026-10-13: kamera B jest kluczowa (+8,9 dB z B vs +0,2 dB bez niej). Pozostały błąd MotionNet to głównie
  brzegi, które wyszły poza kadr A (rysunek `docs/rysunki/analiza_bledu_motion_dual.png`), stąd etap 2.
- 2026-10-24: przejście na ruch obiektów (cel: wyprostowany wiatrak). Zbiór z nagrań 1000 fps (`rs_video_x4k`),
  model RAFT + geometria (`src/raft_geom.py`). Test geometrii z dokładnym przepływem: ~15 → ~36 dB.
- 2026-10-24: RAFT + geometria: 26,42 → 32,86 dB bez treningu, 34,30 dB po dostrojeniu (+1,4 dB, nasycenie po ~5 epokach).
  Wejście ma już 26,4 dB, bo w wielu próbkach X4K ruch jest mały: średnia zaniża efekt na próbkach z silnym RS.
  Wyników nie porównywać wprost z MotionNet (inny zbiór: DIV2K, ruch kamery, wejście 19,7 dB).
- 2026-11-03: zwykły U-Net na `rs_video_x4k`: 28,45 dB (+2 dB), czyli bez RAFT i geometrii sieć prawie się nie uczy korekcji.
- 2026-11-03: geometria v2 (`correct_ab`): +0,4 dB bez dostrajania, +0,1 dB z dostrojonym RAFT; duży zysk tylko na
  krawędziach ruchomych obiektów (syntetyczne koło 26,9 → 29,1 dB). Dalsze dostrajanie z v2: 34,46 dB (nasycenie).
  Zestawienie i demo: `docs/wyniki_2026-11-08/`.
- 2026-11-08: RAFT + geometria + U-Net (notatnik 05): 35,68 dB (RAFT zamrożony), 35,81 dB po wspólnym dostrajaniu,
  czyli +1,35 dB względem najlepszego RAFT + geometrii. Zysk U-Netu podobny w każdej klasie siły RS (+1,3 do +1,6 dB):
  słaby 36,58 → 38,18, średni 34,29 → 35,76, silny 31,93 → 33,20 dB. Poprawa w 303/321 próbkach. U-Net używa
  wewnętrznie geometrii v1 (kod z `main`); do sprawdzenia wersja z geometrią v2.
- 2026-11-11: synteza bez geometrii (notatnik 08, psnr_full, wejście 26,35 dB): autoenkoder 23,27 dB (gorzej niż wejście,
  rozmycie przez wąskie gardło), zwykły U-Net 200 epok 30,94 dB (30 epok: 28,41), enkoder + korelacja 80 epok 31,06 dB.
  Modele ze skipami zatrzymują się ok. 31 dB, poniżej RAFT + geometrii bez żadnego uczenia (32,86). `corr` osiąga to w 80 epokach,
  U-Net potrzebuje ~100; U-Net ma wyższy SSIM (0,895 vs 0,887), bo przesuwa piksele A (ostry obraz), a `corr` je domalowuje.
- 2026-11-11: zasada dla modeli uczonych od zera (U-Net, `ae`, `corr`, `corrgeom`, ablacje, testy kontrolne): **200 epok**,
  bs 16, lr 3e-4, OneCycle. Wcześniejsze runy 30 / 60 / 80 epok zostają jako dodatkowe punkty (pokazują, ile daje dłuższy trening).
