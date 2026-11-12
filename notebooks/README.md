# Notatniki: co robi każdy i jak trenować

Stan na 11.11.2026. Wszystkie notatniki działają tak samo: kod z GitHuba, dane i wagi z Google Drive, obliczenia
na maszynie Colaba. Wyniki każdego treningu lądują w `MyDrive/rs_project/runs/<data>_<nazwa>/`.

## Jak to działa (raz przeczytać)

1. **Kod zmieniasz lokalnie → commit + push.** Colab nie widzi plików z Twojego dysku, tylko to, co jest na GitHubie
   (gałąź `main`).
2. **Każdy notatnik zaczyna się od tych samych komórek:**
   - **SETUP**: montuje Drive, klonuje repo (`git clone`) albo robi `git pull`, wypisuje commit kodu. Pierwszy raz
     zapyta o token GitHuba (albo weźmie go z `MyDrive/rs_project/.github_token`).
   - **1. Zbiór danych**: kopiuje `MyDrive/rs_project/datasets/<DATASET_NAME>.zip` na dysk Colaba i rozpakowuje.
     Zip musi zawierać folder z podfolderami `train/` i `val/`.
   - **2. Nazwy runów**: nazwa folderu wyników (data + nazwa).
3. **Potem sekcje treningu i wyników.** Trening woła `src/train_raft.py` albo `src/train_unet.py`; w folderze runu
   powstają `best.pt`, `last.pt`, `log.csv` (krzywa uczenia), `results.json` (najlepsza epoka), podglądy `preview_ep*.png`.
4. **Po rozłączeniu Colaba:** SETUP → 1 → 2 → ta sama komórka treningu z `RESUME = True` (albo `resume=True`).
   Jeśli wznawiasz innego dnia, wpisz w komórce 2 datę istniejącego runu, bo nazwa zawiera datę.
5. **Liczbę epok ustala się na starcie** (harmonogram OneCycle). Dłuższego treningu nie da się „dokleić” do
   skończonego runu: trzeba zacząć nowy z większym `--epochs`.
6. **Po treningu:** dopisz wiersz do `experiments/README.md` (run, zbiór, epoki, PSNR, uwagi) i zacommituj notatnik
   razem z wynikami, żeby było widać, co wyszło.

Kilka treningów naraz: zrób kopię pliku notatnika dla każdego treningu (jeden plik = jedno jądro Colaba), w każdej
kopii SETUP → 1 → 2 → swoja sekcja. Każda kopia to osobna maszyna i osobne zużycie jednostek Colab Pro.

## Mapa notatników

Wszystkie liczby: val `rs_video_x4k` (321 próbek ze scen spoza treningu), wejście bez korekcji 26,4 dB.

| notatnik | co robi | potrzebuje | tworzy runy | czas (L4) | wynik |
|---|---|---|---|---|---|
| `04_raft_geom` | RAFT + geometria: ocena gotowego RAFT, dostrojenie RAFT (geometria stała), demo wiatraka | zbiór | `<data>_raft_zeroshot`, `<data>_raft_geom` | ~20 min | 32,9 → 34,3 dB |
| `07_demo_v2` | geometria v2 (lepsze krawędzie obiektów), dalsze dostrojenie RAFT z v2, demo GIF | RAFT z 04 | `<data>_raft_geom_v2` | ~20 min | 34,5 dB |
| `05_unet` | **model główny**: RAFT (zamrożony) + geometria + U-Net, potem wspólne dostrajanie | RAFT z 07 (albo 04) | `<data>_raft_unet`, `<data>_raft_unet_ft` | ~30 + 20 min | **35,8 dB** |
| `06_unet_plain` | zwykły U-Net bez RAFT i geometrii, 30 epok (punkt odniesienia) | zbiór | `<data>_unet_plain` | ~11 min | 28,5 dB |
| `08_synteza` | synteza obrazu bez geometrii: zwykły U-Net 200 epok, autoenkoder `ae`, enkoder + korelacja `corr` | zbiór | `<data>_unet_plain_long`, `<data>_ae`, `<data>_corr` | ~70 / ~20 / ~50 min | 30,94 / 23,27 / 31,06 dB (psnr_full) |
| `09_ablacje` | ablacje, wszystko po 200 epok: zwykły U-Net i `corr` z kopią A zamiast B; `corr` 200 epok (para); korelacja + geometria (`corrgeom`: sieć z `corr` przewiduje przesunięcie A→B, dalej nasza geometria); opcjonalnie testy lr i rozmiaru sieci | `*_unet_plain_long` z 08, runy z 04 | `<data>_unet_plain_single`, `<data>_corr200`, `<data>_corr_single`, `<data>_corrgeom`; opcjonalnie `<data>_unet_plain_lr1e-4`, `_lr1e-3`, `<data>_corr_w64` | ~70 min / ~2 h / ~2 h / ~2,5 h | do zrobienia |

| `10_mix_layers` (gałąź `filip-layers`) | trening na sumie `rs_video_x4k` + `rs_layers` (`train_unet.py --data X4K LAYERS`, val X4K), potem każdy model (stary i nowy) oceniany na val X4K i val warstwowym, z podziałem na tekstury | `rs_layers.rar` (albo `.zip`), RAFT z 07 | `<data>_raft_unet_mix`, `<data>_raft_unet_ft_mix`; opcjonalnie `<data>_unet_plain_mix`, `<data>_corr_mix` | ~1,5 + 1,2 h | do zrobienia |

Kolejność, gdy zaczynasz od zera na nowym zbiorze: **04 → 07 → 05** (każdy bierze RAFT z poprzedniego), a 06, 08, 09
niezależnie, w dowolnej kolejności i równolegle. Etap 1 (MotionNet, sam ruch kamery) został usunięty:
opis i wyniki w `docs/etap1_motionnet.md`.

### Co jest czym (w skrócie)

- **RAFT**: gotowa sieć do przepływu optycznego. Dla każdego piksela obrazu A znajduje ten sam punkt w obrazie B.
- **Geometria**: zwykłe obliczenia bez uczenia. Każdy piksel A i B ma znany czas odczytu (wiersz w A, kolumna w B),
  więc prędkość punktu = przesunięcie między A i B / różnica czasów. Znając prędkość, cofamy piksel do jednej chwili.
- **U-Net na końcu (05)**: poprawia drobne błędy geometrii i łączy A z B (brzegi widoczne tylko w B).
- **Modele bez geometrii (06, 08)**: sieć dostaje A, B i czasy odczytu i sama ma się nauczyć korekcji. Służą do
  porównania architektur w pracy: pokazują, ile daje jawna geometria.
- Metryki w logu: `psnr` (obszar widoczny w A), `psnr_full` (cały obraz: **wspólna miara dla wszystkich modeli**),
  `*_in` (wejście bez korekcji), `psnr_geom` (sam RAFT + geometria, w modelu z 05).

## Trening na nowym zbiorze (np. warstwowym z `synth_layers.py`)

1. Wygeneruj zbiór (folder z `train/`, `val/`, `config.json`), spakuj go do zip i wgraj do
   `MyDrive/rs_project/datasets/<nazwa>.zip`, np. `rs_layers.zip`.
2. W notatniku, w komórce 1 ustaw `DATASET_NAME = 'rs_layers'`.
3. **Zmień nazwy runów w komórce 2**, dopisując nazwę zbioru, np. `f'{date.today()}_raft_unet_layers'`.
   Inne notatniki szukają runów po końcówce nazwy (`P.find_run('raft_unet_ft')` bierze najnowszy) i bez sufiksu
   pomyliłyby wagi z X4K z wagami z nowego zbioru.
4. Uruchom jak zwykle. Wyniki na nowym zbiorze nie są porównywalne z X4K wprost (inna siła RS: wejście ok. 15 dB
   zamiast 26), więc w `experiments/README.md` zapisuj, na jakim zbiorze liczony jest val.

## Plan treningów do końca prac

- **Faza 1 (do ok. 14.11), wszystko na `rs_video_x4k`:** porównanie architektur. Zrobione: 04, 05, 06, 07.
  Do zrobienia: 08 (w trakcie), 09 (ablacje kamery B i geometrii).
- **Faza 2 (ok. 15–19.11), zbiór rozszerzony:** więcej klipów z X4K (`synth_video_v2.py`) + zbiór warstwowy.
  Ponownie: RAFT + geometria + U-Net (05, z geometrią v2 w U-Necie), `corr`, zwykły U-Net. Każdy model oceniany
  także na starym val `rs_video_x4k` (porównanie z fazą 1) i na val zbioru warstwowego (osobno zdjęcia i tekstury
  proceduralne). Trening na kilku zbiorach naraz wymaga jeszcze drobnej zmiany w `train_unet.py`.
