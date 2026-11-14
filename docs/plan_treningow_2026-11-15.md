# Plan treningów na 15.11.2026: zbiór rozszerzony (X4K + warstwowy)

Cel dnia: te same modele co na `rs_video_x4k`, ale uczone na sumie X4K + zbiór warstwowy Filipa, i ocena każdego
modelu (starego i nowego) na **obu** zbiorach walidacyjnych. Odpowiada na pytanie: czy więcej i bogatszych danych
(paralaksa, zasłonięcia, większe przesunięcia) poprawia modele, i który rodzaj modelu zyskuje bardziej.

Notatnik: **`notebooks/10_mix_layers.ipynb` z gałęzi `filip-layers`** (komórka SETUP sama klonuje tę gałąź).
Gałąź ma już wszystkie poprawki z `main` z 14.11 (bf16 w sieci `corr`, automatyczne zatrzymanie martwych runów).

## 0. Przed startem (raz, ~5 min)

1. **Który zbiór warstwowy?** Na Drive (`MyDrive/rs_project/datasets/`) są dwa archiwa: `rs_layers.rar` (11.11, 3,05 GB)
   i `rs_layers_2.rar` (13.11, 3,30 GB). Ustalić z Filipem, który jest właściwy, i wpisać go w komórce 1:
   `LAYERS_NAME = 'rs_layers'` albo `'rs_layers_2'`.
2. **Karta: L4.** A100 się nie opłaca (małe sieci, karta czeka na dane). T4 jest tańszy za epokę, ale `corr` liczy się na nim
   w fp32 (wolno).
3. Notatnik otwierać **prosto z GitHuba** (Plik → Otwórz notatnik → GitHub → `kordiusz/RollingShutter`, gałąź `filip-layers`,
   `notebooks/10_mix_layers.ipynb`) i zapisać kopię na Drive. Stara kopia notatnika ma stare liczby epok.
4. Po SETUP sprawdzić wypisany commit: ma być najnowszy z `filip-layers` (nie `542b59e`).

## 1. Dlaczego tak mało epok

Liczymy **tę samą liczbę kroków optymalizatora** co w parze na X4K, nie tę samą liczbę epok. Zbiór mieszany ma ~7,8 tys.
próbek zamiast 2,4 tys., więc jedna epoka to ~3,2× więcej kroków. Komórka 2 liczy epoki sama:
`epoki_X4K × n_X4K / (n_X4K + n_layers)`. Przy 2397 + 5387 próbkach:

| model | epoki X4K | epoki mix | kroki |
|---|---|---|---|
| RAFT + geometria + U-Net (`raft_unet`) | 30 | 9 | ~8,8 tys. (bs 8), jak w parze |
| wspólne dostrajanie (`raft_unet_ft`) | 10 | 3 | ~2,9 tys. (bs 8) |
| `corr` | 80 | 25 | ~12 tys. (bs 16) |
| zwykły U-Net (opcjonalnie) | 200 | 62 | ~30 tys. (bs 16) |

Dzięki temu nowy run różni się od starego **tylko danymi**, a nie długością treningu, i kosztuje ~3× mniej.

**To nie jest dostrajanie starych wag.** Każdy model uczy się od zera na sumie zbiorów (tak jak stary run na samym X4K).
Wyjątek, tak samo jak w parach: RAFT w 4a startuje z wag `*_raft_geom_v2` (dostrojonych na X4K w notatniku 07) i jest
zamrożony, a 4b startuje z `best.pt` z 4a. Dostrajanie starych modeli X4K na nowym zbiorze byłoby tańsze, ale wtedy model
widziałby X4K dłużej niż para i nie dałoby się rozdzielić wpływu danych od wpływu dłuższego treningu.

## 2. Co trenować (kolejność)

Dwie kopie notatnika = dwie maszyny L4 równolegle. W każdej: **SETUP → 1 → 2**, potem swoja sekcja.

| maszyna | sekcje | run | czas L4 (szacunek) | priorytet |
|---|---|---|---|---|
| 1 | 4a, potem 4b | `<data>_raft_unet_mix`, `<data>_raft_unet_ft_mix` | ~30 + ~20 min | **obowiązkowe** (model główny) |
| 2 | 4d | `<data>_corr_mix` | ~45 min | **obowiązkowe** (najlepsza synteza wprost) |
| 2 (po 4d) | 4c | `<data>_unet_plain_mix` | ~70 min | opcjonalne, jeśli zostaną jednostki |
| 1 (po 4b) | 5 | porównanie na obu walidacjach | ~10–20 min | **obowiązkowe** |

Co obserwować w logu (pierwsze 1–2 epoki):
- 4a / 4b: `RAFT+geom` i `val PSNR` powinny od razu być w okolicy starych runów (34–35 dB na val X4K), `from B` kilka–kilkanaście %.
- 4d: val PSNR powinien powoli rosnąć od 26,35. Jeśli pojawi się `STOP: ...` (NaN, pominięte kroki albo „output is dead”):
  usunąć na Drive folder `runs/<data>_corr_mix`, w komórce 2 ustawić `CORR_LR = 1e-4`, uruchomić komórkę 2 i `train('corr')`
  jeszcze raz. W `experiments/README.md` zapisać, że lr różni się od pary (3e-4). `corr` jest niestabilny przy 3e-4
  (14.11: dwie z trzech prób na X4K się wywróciły).
- Po rozłączeniu Colaba: SETUP → 1 → 2 → `train('<job>', resume=True)`. Innego dnia: w komórce 2 wpisać `DAY` z nazwy folderu.

## 3. Po treningu

1. Sekcja 5 (porównanie): tabela `MyDrive/rs_project/results/porownanie_mix/tabela.csv`, każdy model na val X4K i val warstwowym,
   na warstwowym osobno zdjęcia / tekstury proceduralne.
2. **Od razu odłączyć obie maszyny** (Środowisko wykonawcze → Odłącz i usuń środowisko). Podłączona maszyna liczy jednostki
   także bez treningu.
3. Wyniki do `experiments/README.md` (na `filip-layers`): wiersz zbioru `rs_layers` w tabeli zbiorów (parametry z `config.json`,
   liczba próbek train / val) i wiersze nowych runów (val X4K i val warstwowy osobno).

## 4. Jak czytać wyniki

- **Na val X4K** nowe runy są wprost porównywalne ze starymi (ta sama walidacja, ta sama liczba kroków).
- **Na val warstwowym** liczby bezwzględne będą dużo niższe (wejście ~15 dB zamiast 26): patrzeć na zysk względem wejścia
  i na różnicę stary / nowy model.
- Najciekawsze pytanie: **kto zyskuje więcej na nowych danych, model z geometrią czy synteza (`corr`)?** Jeśli synteza zyska
  wyraźnie więcej, wniosek „synteza potrzebuje dużo więcej danych”; jeśli tyle samo albo mniej, teza o geometrii się wzmacnia.

## 5. Czego nie robimy (decyzje z 14.11)

- `corr_single`, `corrgeom` (notatnik 09): porzucone, sieć `corr` niestabilna. Ablację kamery B niesie para zwykłego U-Netu
  (200 ep.: 30,94 z B, 26,51 bez B) i etap 1 (MotionNet). Geometrię uzasadnia RAFT + geometria bez uczenia (32,86) kontra
  najlepsza synteza (31,06).
- `corr200`, testy lr i `corr_w64`: niepotrzebne do tezy.
- Nowy wariant autoenkodera: trzy sieci syntezy już stoją na ~31 dB, kolejny nic nowego nie powie.
- Własna sieć przepływu (strata fotometryczna): tylko jeśli po południu 4a–4d są gotowe i zostały jednostki; jedna próba,
  bez poprawek. Inaczej opis w „dalszych pracach”.
