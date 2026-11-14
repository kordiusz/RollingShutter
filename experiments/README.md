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
| (eval) Farnebäck + geometria v1 | 2026-11-14 | klasyczny przepływ OpenCV (bez uczenia) + geometria v1 | rs_video_x4k | 11_tabela_glowna | CPU | 0 | 32.28 | 0.910 | – | – | 221 (CPU) | – | psnr_full 31.84 (CI95 po scenach [29,72; 33,70]) |
| (eval) DIS + geometria v1 | 2026-11-14 | klasyczny przepływ OpenCV DIS (preset MEDIUM, bez strojenia) + geometria v1 | rs_video_x4k | 11_tabela_glowna | CPU | 0 | **34.10** | 0.934 | – | – | 240 (CPU) | – | **psnr_full 33.44** [31,23; 35,34]: bez uczenia i pretreningu, lepiej niż RAFT gotowy (32,86) i każda sieć bez geometrii |
| (eval) DIS + geometria v2 | 2026-11-14 | jw., geometria v2 (DIS A→B i B→A) | rs_video_x4k | 11_tabela_glowna | CPU | 0 | 34.04 | 0.935 | – | – | 713 (CPU) | – | psnr_full 33.45: v2 nic nie dodaje przy DIS |
| (eval) samo B / średnia A+B | 2026-11-14 | trywialne punkty odniesienia | rs_video_x4k | 11_tabela_glowna | CPU | 0 | – | 0.575 / 0.740 | – | – | – | – | psnr_full 19.50 / 23.47 (wejście A 26.35) |
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

## Tabela główna (14.11, notatnik 11, `src/main_table.py`)

Val `rs_video_x4k` (321 próbek, 13 scen spoza treningu), jedna miara dla wszystkich metod: **psnr_full / ssim_full = cały
obraz, średnia po obrazach**. `psnr_vis` = obszar widoczny w A (dawna kolumna „PSNR” dla RAFT i modeli z 05), tylko dla
porównania ze starszymi wierszami. CI95 = bootstrap po scenach. Różnica = sparowana (te same próbki) względem modelu głównego
(`2026-11-08_raft_unet_ft`), „lepsza w” = w ilu próbkach metoda bije model główny. ms/parę: mediana 20 przebiegów jednej pary
(sama metoda); metody klasyczne na CPU (OpenCV + geometria w PyTorch), reszta na L4, więc czasy CPU i GPU nie są porównywalne
wprost. Kroki = epoki × (próbki / bs) danego runu, bez wcześniejszych etapów (patrz uwagi).

| grupa | metoda | psnr_full [dB] | CI95 (sceny) | ssim_full | psnr_vis | różnica vs model główny [dB] (CI95) | lepsza w | ms/parę | param. [M] | kroki |
|---|---|---|---|---|---|---|---|---|---|---|
| trywialne | bez korekcji (obraz A) | 26,35 | [23,51; 28,96] | 0,797 | 26,35 | −9,02 [−10,84; −7,32] | 8/321 | – | – | – |
| trywialne | sam obraz B | 19,50 | [18,59; 20,52] | 0,575 | 19,50 | −15,87 [−17,00; −14,24] | 0/321 | – | – | – |
| trywialne | średnia A i B | 23,47 | [22,06; 24,85] | 0,740 | 23,47 | −11,90 [−12,79; −10,57] | 0/321 | – | – | – |
| przepływ klasyczny + geometria | Farnebäck + geometria v1 | 31,84 | [29,72; 33,70] | 0,910 | 32,28 | −3,53 [−4,29; −2,82] | 17/321 | 194 (CPU) | – | 0 |
| przepływ klasyczny + geometria | **DIS + geometria v1** | **33,44** | [31,23; 35,33] | 0,935 | 34,10 | −1,93 [−2,40; −1,52] | 41/321 | 179 (CPU) | – | 0 |
| przepływ klasyczny + geometria | DIS + geometria v2 | 33,45 | [31,31; 35,31] | 0,935 | 34,03 | −1,92 [−2,44; −1,50] | 34/321 | 547 (CPU) | – | 0 |
| RAFT + geometria | RAFT gotowy + geometria v1 | 32,34 | [30,42; 34,05] | 0,933 | 32,86 | −3,03 [−3,59; −2,57] | 4/321 | 48 | 5,26 | 0 (pretrening zewn.) |
| RAFT + geometria | RAFT gotowy + geometria v2 | 32,81 | [30,94; 34,44] | 0,936 | 33,25 | −2,56 [−2,97; −2,19] | 7/321 | 59 | 5,26 | 0 (pretrening zewn.) |
| RAFT + geometria | RAFT dostrojony (04) + geometria v1 | 33,60 | [31,58; 35,33] | 0,938 | 34,30 | −1,77 [−2,00; −1,56] | 15/321 | 48 | 5,26 | 2 990 |
| RAFT + geometria | RAFT dostrojony z v2 (07) + geometria v2 | 33,89 | [31,93; 35,54] | 0,940 | 34,46 | −1,48 [−1,68; −1,32] | 17/321 | 60 | 5,26 | 1 495 (+2 990) |
| sieci bez geometrii | autoenkoder bez skipów (ae), 60 ep. | 23,27 | [22,20; 24,31] | 0,681 | 23,27 | −12,09 [−13,26; −10,51] | 0/321 | 4,4 | 10,2 | 8 940 |
| sieci bez geometrii | zwykły U-Net, 30 ep. | 28,42 | [26,01; 30,52] | 0,846 | 28,45 | −6,95 [−8,36; −5,65] | 9/321 | 4,3 | 2,72 | 4 470 |
| sieci bez geometrii | zwykły U-Net, 200 ep. | 30,94 | [28,40; 33,17] | 0,895 | 31,05 | −4,42 [−5,64; −3,43] | 19/321 | 4,3 | 2,72 | 29 800 |
| sieci bez geometrii | enkoder + korelacja (corr), 80 ep. | 31,06 | [28,63; 33,25] | 0,887 | 31,06 | −4,31 [−5,53; −3,22] | 14/321 | 11,3 | 6,24 | 11 920 |
| ablacja: bez kamery B | zwykły U-Net bez kamery B, 200 ep. | 26,51 | [23,79; 29,04] | 0,810 | 26,53 | −8,86 [−10,59; −7,24] | 6/321 | 4,2 | 2,72 | 29 800 |
| RAFT + geometria + U-Net | RAFT (zamrożony) + geometria + U-Net | 35,24 | [33,29; 36,91] | 0,953 | 35,68 | −0,13 [−0,16; −0,10] | 73/321 | 56 | 7,98 | 8 970 (+4 485 RAFT) |
| RAFT + geometria + U-Net | **RAFT + geometria + U-Net, wspólnie dostrojone (model główny)** | **35,37** | [33,41; 37,04] | **0,954** | 35,81 | – | – | 56 | 7,98 | 2 990 (+8 970 +4 485) |

Uwagi: `psnr_vis` zgadza się z wcześniejszymi wpisami (sprawdzone dla wszystkich runów; `corr` 31,062 zamiast 31,064, bo
liczony teraz w bf16). Wariant v1/v2 dotyczy geometrii przy ocenie; U-Net w modelu głównym używa geometrii v1. Kroki RAFT:
04 (2 990) + 07 (1 495) przed U-Netem, do tego pretrening torchvision. Pliki: `MyDrive/rs_project/results/tabela_glowna/`
(`tabela_glowna.csv`, `per_sample/<metoda>.csv`, `methods.json`).

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
- 2026-11-14: ablacja kamery B (notatnik 09, bez treningu, val `rs_video_x4k`, piksele z |t_A| >= 5 ms, zysk względem wejścia):
  | abs(t_B − t_A) | ~1 ms | ~3 | ~5 | ~7,5 | ~10,5 | ~14 | ~18 | ~25,5 |
  |---|---|---|---|---|---|---|---|---|
  | zwykły U-Net, tylko A | +0,31 | +0,30 | +0,31 | +0,31 | +0,31 | +0,30 | +0,33 | +0,29 |
  | zwykły U-Net, A + B | +5,15 | +5,61 | +5,66 | +4,85 | +4,59 | +4,02 | +3,93 | +3,67 |
  | RAFT + geometria + U-Net (2026-11-08_raft_unet_ft) | +6,82 | +8,22 | +11,48 | +13,56 | +13,79 | +11,69 | +11,82 | +11,53 |

  (1) Bez kamery B sieć nie poprawia niczego (+0,3 dB w każdym przedziale), z B +3,7 do +5,7 dB: B jest konieczna.
  (2) **Model z geometrią korzysta z różnicy chwil odczytu**: zysk rośnie z +6,8 dB tam, gdzie obie kamery czytają piksel
  prawie jednocześnie, do +13,6–13,8 dB przy 7,5–10,5 ms różnicy (prędkość = przesunięcie / (t_B − t_A)); przy t_B ≈ t_A
  prędkość jest nieokreślona i geometria uzupełnia ją z sąsiedztwa. Pomiar potwierdza zasadę prostopadłego odczytu.
  (3) Zwykły U-Net z tego nie korzysta: jego zysk z B lekko maleje z różnicą czasów (korelacja po położeniu −0,56), a mapa
  zysku zależy od wiersza (od |t_A|), nie od przekątnej. Traktuje B jako drugi, inaczej zniekształcony widok sceny, a nie jako
  pomiar ruchu w czasie: to tłumaczy, czemu modele bez geometrii stoją na ~31 dB. Rysunek: `results/ablacja_kamery_b/zysk_z_kamery_b.png`.
- 2026-11-14: niezależna recenzja (agent) + tabela główna (`src/main_table.py`, notatnik 11):
  * **Klasyczny przepływ DIS (OpenCV, bez uczenia) + nasza geometria: 33,44 dB psnr_full** (34,10 psnr): więcej niż RAFT gotowy
    (32,86 psnr) i każda sieć bez geometrii (≤ 31,06), prawie tyle co RAFT dostrojony (34,30 psnr). Argument „liczy się geometria”
    nie zależy więc od pretreningu RAFT; zastępuje nieudany `corrgeom` i własną sieć przepływu.
  * **Jedna miara:** `train_raft.py` liczył tylko psnr na obszarze widocznym w A, sieci bez geometrii psnr_full; tabela główna
    (notatnik 11) liczy wszystko tak samo (psnr_full, ssim_full, średnia po obrazach).
  * **Niepewność:** val = 13 scen × ~27 próbek; CI po scenach dla modelu głównego [33,9; 37,3] (po próbkach złudne ±0,45).
    Raportować różnice sparowane: U-Net w modelu głównym nad RAFT + geometrią +1,45 dB, CI [1,28; 1,65], lepiej w 303/321.
  * Do opisania w ograniczeniach: brak osobnego zbioru testowego (best.pt wybierany na val), szum tylko na wejściu RS (cel czysty:
    część zysku U-Netu może być odszumianiem; największy zysk w klasie „słaby RS”), sufit ~36 dB przy dokładnym przepływie był
    liczony na innym zbiorze, nierówne budżety treningu (tabela kroków w notatniku 11), PSNR w analizie kamery B liczony z MSE
    zsumowanego po pikselach (nieporównywalny wprost z tabelą główną).
- 2026-11-14: tabela główna (jedna miara, psnr_full; CI po scenach, różnice sparowane względem modelu głównego):
  * **Hierarchia jest pewna mimo 13 scen**: wszystkie różnice względem modelu głównego mają CI95 z dala od zera. Model główny
    35,37 dB; bez U-Netu (najlepsza sama geometria, RAFT z v2) −1,48 [−1,68; −1,32]; DIS + geometria −1,93; najlepsza sieć
    bez geometrii (`corr`) −4,31 [−5,53; −3,22]; bez kamery B −8,86.
  * **Geometria jest kluczem, nie pretrening RAFT**: DIS + geometria (33,44) > RAFT gotowy + geometria (32,34 v1 / 32,81 v2)
    i tylko o 0,16 dB < RAFT dostrojony + v1 (33,60). Każdy przepływ + geometria (≥ 31,84) jest na poziomie lub powyżej
    najlepszej sieci bez geometrii (31,06 przy 11 920 krokach; zwykły U-Net 30,94 przy 29 800).
  * Geometria v2 daje +0,47 przy RAFT gotowym i 0 przy DIS (RAFT z 07 + v2 kontra RAFT z 04 + v1: +0,29, ale tam zmieniają
    się też wagi RAFT, więc to nie jest czysty efekt geometrii). Wspólne dostrajanie RAFT + U-Net: +0,13
    [0,10; 0,16], lepiej w 248/321.
  * Czas (L4): sieci bez geometrii 4–11 ms, RAFT + geometria 48–60 ms, model główny 56 ms (RAFT large 12 iteracji dominuje).
    DIS + geometria na CPU 179 ms (głównie geometria w PyTorch na CPU, nie sam DIS); do sprawdzenia na GPU, jeśli czas ma być argumentem.
