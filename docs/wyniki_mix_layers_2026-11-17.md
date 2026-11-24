# Wyniki: trening na X4K + syntetycznym zbiorze warstwowym `rs_layers_2`

Stan na 17.11.2026. Notatnik: `notebooks/10_mix_layers_2.ipynb` (gałąź `filip-layers`).
Wagi: `MyDrive/rs_project/runs/2026-11-16_*_mix/`. Tabela: `MyDrive/rs_project/results/porownanie_mix/tabela.csv`.

## 1. Pytanie

Czy dołożenie do treningu w pełni syntetycznego zbioru `rs_layers_2` poprawia modele korekcji rolling shutter,
w porównaniu z tymi samymi modelami uczonymi tylko na `rs_video_x4k`?

## 2. Zbiory danych

Każdy zbiór jest podzielony na dwie rozłączne części. Na `train/` model się uczy. `val/` jest odłożony i służy
tylko do sprawdzania: model nigdy go nie widzi, więc wynik na `val/` pokazuje, jak działa na nowych obrazach.

| zbiór | co to jest | train | val | jakość wejścia (bez korekcji) |
|---|---|---|---|---|
| `rs_video_x4k` („X4K”) | prawdziwe nagrania z szybkiej kamery; efekt RS symulowany z kolejnych klatek | 2397 | **321** („val X4K”) | 26,35 dB, umiarkowane zniekształcenie |
| `rs_layers_2` („warstwowy”) | w pełni syntetyczny (`src/synth_layers.py`): tło ze zdjęć DIV2K i wycięte obiekty z COCO, poruszające się na różnych głębokościach | 5387 | **613** („val warstwowy”) | 16,09 dB, bardzo silne zniekształcenie |

Trening „X4K + layers” to suma obu `train/`, czyli 7784 próbki: 31% realne wideo, 69% syntetyk.

## 3. Modele

Każdy model dostaje dwa obrazy tej samej sceny z efektem RS: kamera A czyta wierszami, kamera B kolumnami.
Ma z nich odtworzyć obraz bez zniekształceń, czyli całą klatkę z jednej chwili.

| model | sekcja | jak działa | parametry |
|---|---|---|---|
| `raft_unet` | 4a | RAFT (gotowa sieć przepływu optycznego, **zamrożona**) znajduje odpowiadające sobie punkty w A i B. Wzór geometryczny (bez uczenia) liczy z tego ruch i cofa piksele do jednej chwili. Mały U-Net na końcu poprawia błędy i dobiera, co wziąć z B. | 7,98 M (uczone 2,73 M) |
| `raft_unet_ft` | 4b | To samo co 4a, ale startuje z wyniku 4a i **dostraja też RAFT** (wspólne dostrajanie). **Model główny pracy.** | 7,98 M (wszystkie uczone) |
| `unet_plain` | 4c | zwykły U-Net bez RAFT i bez geometrii | 2,72 M |
| `corr` | 4d | sieć bez geometrii: wspólny enkoder dla A i B oraz warstwa korelacji porównująca A z B. **Ruchu musi nauczyć się sama.** | 6,24 M |
| `ae` | 4e | autoenkoder bez połączeń skrótowych: ściska wejście do małej reprezentacji (16×16×512) i maluje obraz od nowa. Punkt odniesienia. | 10,20 M |

`unet_plain` (4c) **nie był trenowany** na zbiorze mieszanym (pominięty, żeby oszczędzić jednostki Colab).
W tabeli jest tylko jego stara wersja.

## 4. Jak było zrobione porównanie

- **Para dla każdego modelu:** stary run uczony tylko na X4K i nowy run uczony na X4K + layers. Architektura,
  learning rate i batch size są takie same; różnią się **tylko dane treningowe**.
- **Ta sama liczba kroków treningu, nie ta sama liczba epok.** Zbiór mieszany jest ~3,2× większy, więc epok jest
  ~3,2× mniej: `epoki_mix = epoki_X4K × 2397 / 7784`.

  | model | epoki: tylko X4K → mix | czas na L4 (mix) |
  |---|---|---|
  | `raft_unet` | 30 → 9 | 9 × 171 s ≈ 26 min |
  | `raft_unet_ft` | 10 → 3 | 3 × 403 s ≈ 20 min |
  | `corr` | 80 → 25 | 25 × 109 s ≈ 46 min |
  | `ae` | 60 → 18 | 18 × 57 s ≈ 17 min |

- **Wybór najlepszej epoki w trakcie treningu: tylko na val X4K.** Wynik na realnym wideo jest więc wprost
  porównywalny ze starymi runami.
- **Na końcu każdy model (stary i nowy) oceniony osobno na obu `val`:** to dwa niezależne „egzaminy”
  (sekcja 5 notatnika).

## 5. Miary (kolumny tabeli)

| miara | co mierzy | jak czytać |
|---|---|---|
| **PSNR** (dB) | podobieństwo do prawdziwego obrazu, piksel po pikselu | więcej = lepiej; +1 dB to wyraźna poprawa, 0,1–0,2 dB to mała różnica |
| `psnr` | PSNR tylko w obszarze widocznym w obrazie A | brzegi widoczne tylko w B są pominięte |
| **`psnr_full`** | PSNR na **całym obrazie** | **wspólna miara dla wszystkich modeli, główna liczba** |
| `ssim` | podobieństwo struktury (krawędzie, tekstury), 0–1 | więcej = lepiej; 1 = identyczny |
| `RAFT+geom` | wynik samego RAFT + wzoru geometrycznego, bez końcowego U-Netu | pokazuje, ile daje sam pierwszy etap modelu głównego |

`corr` i `ae` nie liczą ruchu piksel po pikselu, tylko malują cały obraz, więc u nich `psnr` = `psnr_full`.

## 6. Pełna tabela wyników

„Trening” oznacza dane, na których model się uczył. Kolumny oznaczają, na czym go sprawdzano.

| model | trening | run | X4K `psnr` | **X4K `psnr_full`** | X4K `ssim` | warstw. `psnr` | **warstw. `psnr_full`** | warstw. `ssim` | X4K `RAFT+geom` | warstw. `RAFT+geom` |
|---|---|---|---|---|---|---|---|---|---|---|
| bez korekcji | – | – | 26,35 | **26,35** | 0,80 | 16,09 | **16,09** | 0,42 | – | – |
| `raft_unet` | X4K | `2026-11-08_raft_unet` | 35,68 | **35,24** | 0,95 | 24,16 | **23,81** | 0,86 | 34,36 | 23,36 |
| `raft_unet` | X4K + layers | `2026-11-16_raft_unet_mix` | 35,53 | **35,12** | 0,95 | 24,70 | **24,33** | 0,87 | 34,36 | 23,36 |
| `raft_unet_ft` | X4K | `2026-11-08_raft_unet_ft` | 35,81 | **35,37** | 0,96 | 24,23 | **23,87** | 0,86 | 34,42 | 23,44 |
| `raft_unet_ft` | X4K + layers | `2026-11-16_raft_unet_ft_mix` | 35,55 | **35,14** | 0,95 | 25,15 | **24,72** | 0,88 | 34,30 | 23,77 |
| `unet_plain` | X4K | `2026-11-11_unet_plain_long` | 31,05 | **30,94** | 0,90 | 20,37 | **20,29** | 0,68 | – | – |
| `corr` | X4K | `2026-11-11_corr` | 31,06 | **31,06** | 0,89 | 20,61 | **20,61** | 0,64 | – | – |
| `corr` | X4K + layers | `2026-11-16_corr_mix` | 31,78 | **31,78** | 0,90 | 22,46 | **22,46** | 0,72 | – | – |
| `ae` | X4K | `2026-11-11_ae` | 23,27 | **23,27** | 0,68 | 17,65 | **17,65** | 0,44 | – | – |
| `ae` | X4K + layers | `2026-11-16_ae_mix` | 23,10 | **23,10** | 0,68 | 18,70 | **18,70** | 0,48 | – | – |

Notatnik ma jeszcze kolumnę `layers psnr (photos)`. W `rs_layers_2` wszystkie tekstury pochodzą ze zdjęć (nie ma
proceduralnych), więc ta kolumna to po prostu cały val warstwowy i nie wnosi nic nowego.

## 7. Zmiana po dodaniu `rs_layers_2` (`psnr_full`, nowy run minus stary)

| model | realne wideo (val X4K) | Twoje sceny (val warstwowy) | czy pomogło? |
|---|---|---|---|
| `raft_unet` (4a) | **−0,12** | **+0,52** | na realnym wideo nie, na syntetyku tak |
| `raft_unet_ft` (4b, główny) | **−0,23** | **+0,85** | na realnym wideo nie, na syntetyku tak |
| `corr` (4d) | **+0,72** | **+1,85** | **tak, na obu** |
| `ae` (4e) | **−0,17** | **+1,05** | na realnym wideo nie, na syntetyku tak |

## 8. Wyjaśnienie każdego wyniku

### Bez korekcji
- **X4K 26,35 dB, SSIM 0,80:** tak bardzo zniekształcone jest wejście z realnego wideo. Każdy model musi być
  powyżej tej liczby, żeby w ogóle pomagać.
- **Warstwowy 16,09 dB, SSIM 0,42:** Twoje sceny są dużo trudniejsze, z szybszym ruchem i silniejszym RS. Dlatego
  wszystkie wyniki na val warstwowym są o ~10 dB niższe niż na X4K. **Liczb z dwóch kolumn nie porównuje się
  ze sobą**, tylko stary run z nowym w tej samej kolumnie.

### `raft_unet` (4a): RAFT zamrożony + geometria + U-Net
- **Tylko X4K: 35,24 dB na X4K, 23,81 dB na warstwowym.** Bardzo dobry na realnym wideo (+8,9 dB względem
  wejścia). Na Twoich scenach też mocno poprawia (+7,7 dB), choć nigdy ich nie widział.
- **X4K + layers: 35,12 (−0,12) i 24,33 (+0,52).**
  - Na realnym wideo minimalnie gorzej: w tej samej liczbie kroków realne wideo stanowiło tylko 31% przykładów.
  - Na Twoich scenach lepiej, bo U-Net nauczył się poprawiać typowe dla nich błędy.
- **`RAFT+geom` identyczne w obu (34,36 / 23,36):** w 4a RAFT jest zamrożony, więc pierwszy etap jest
  dokładnie ten sam. Cała różnica pochodzi z końcowego U-Netu.
- **Odsetek obrazu brany z kamery B** pod koniec treningu: ~40–43% (log `from B`). U-Net aktywnie korzysta z obu kamer.

### `raft_unet_ft` (4b): model główny, RAFT dostrajany
- **Tylko X4K: 35,37 dB na X4K (najlepszy wynik na realnym wideo), 23,87 dB na warstwowym.**
- **X4K + layers: 35,14 (−0,23) i 24,72 (+0,85).**
  - Na realnym wideo spadek o 0,23 dB. To mało, ale więcej niż rozrzut między powtórzeniami (~0,05 dB, punkt 9),
    więc raczej realny.
  - Na Twoich scenach to **najlepszy wynik ze wszystkich modeli: 24,72 dB**, +8,6 dB względem wejścia.
- **`RAFT+geom`:**
  - Dostrojony na mixie RAFT jest **lepszy na Twoich scenach (23,77 vs 23,44)**, bo nauczył się ich ruchu,
    i **odrobinę gorszy na X4K (34,30 vs 34,42)**.
  - Zysk modelu głównego na syntetyku pochodzi więc częściowo z lepszego przepływu optycznego, a częściowo z U-Netu.
- **Dostrajanie na mixie dało mało względem 4a:** X4K 35,12 → 35,14, warstwowy 24,33 → 24,72. Przy uczeniu tylko
  na X4K dostrajanie dawało więcej (35,24 → 35,37).

### `unet_plain` (4c): zwykły U-Net, bez geometrii
- **Tylko X4K: 30,94 dB na X4K, 20,29 dB na warstwowym.** Bez geometrii o ~4,4 dB gorzej niż model główny.
  Wersji na mixie nie ma, bo nie był trenowany.

### `corr` (4d): korelacja A↔B, bez geometrii
- **Tylko X4K: 31,06 dB na X4K, 20,61 dB na warstwowym.**
- **X4K + layers: 31,78 (+0,72) i 22,46 (+1,85).**
  - **Jedyny model, któremu Twoje dane pomogły także na realnym wideo.** Ten model nie ma wzoru geometrycznego
    i musi sam nauczyć się, jak wygląda ruch. Twój zbiór daje mu dużo różnorodnego ruchu (obiekty na różnych
    głębokościach, różne prędkości), a to przenosi się też na realne nagrania.
  - Na Twoich scenach zysk jest największy ze wszystkich modeli (+1,85 dB). SSIM 0,64 → 0,72, czyli wyraźnie
    lepiej odtworzona struktura.
- **Mimo zysku nadal ~3,4 dB gorszy od modelu głównego na X4K** (31,78 vs 35,14). Jawna geometria daje więcej
  niż dodatkowe dane.
- Trening stabilny (bez `STOP`), mimo że `corr` przy lr 3e-4 bywał wcześniej niestabilny.

### `ae` (4e): autoenkoder
- **Tylko X4K: 23,27 dB na X4K, czyli gorzej niż wejście (26,35).** Wąskie gardło (16×16) gubi szczegóły:
  obraz wychodzi rozmyty. Model „poprawia” geometrię, ale psuje ostrość.
- **X4K + layers: 23,10 (−0,17) i 18,70 (+1,05).**
  - Na Twoich scenach lepiej: tam wejście jest tak zniekształcone (16,09 dB), że nawet rozmyty, ale geometrycznie
    poprawiony obraz bije wejście o 2,6 dB.
  - Na realnym wideo bez zmian, nadal poniżej wejścia. **Problemem jest architektura, nie dane.**
- W epoce 17 jeden krok na 486 pominięty (`SKIPPED 1/486`): zabezpieczenie przed niestabilnym krokiem, bez wpływu na wynik.

## 9. Powtarzalność

Trening na mixie (4a, 4b, 4d) wykonano **dwa razy**: 16.11 wieczorem (wagi nie zapisały się na Drive; liczby są
w historii notatnika, commit `5367e97`) i ponownie 16–17.11.

| model | `psnr_full` X4K: przebieg 1 / 2 | `psnr_full` warstw.: przebieg 1 / 2 |
|---|---|---|
| `raft_unet` mix | 35,13 / 35,12 | 24,41 / 24,33 |
| `raft_unet_ft` mix | 35,18 / 35,14 | 24,78 / 24,72 |
| `corr` mix | 31,79 / 31,78 | 22,50 / 22,46 |

Rozrzut między przebiegami wynosi **≤ 0,08 dB**. Różnice z punktu 7 dla `corr` (+0,72 / +1,85) i dla zysków na
val warstwowym są dużo większe, więc pewne. Spadek modelu głównego na X4K (−0,23) jest większy od rozrzutu,
ale mały.

Ograniczenie: runy „tylko X4K” mają po jednym przebiegu, a test istotności (sparowany, po scenach) jeszcze nie był
liczony.

## 10. Dlaczego modelowi głównemu dane syntetyczne nie pomogły na realnym wideo

1. **Ruch liczy u niego RAFT z geometrią, nie uczenie.** W 4a RAFT jest zamrożony, a pierwszy etap daje identyczne
   34,36 dB z syntetykiem i bez niego. Dodatkowe dane uczą głównie małego U-Netu, który ma niewiele do poprawienia.
2. **Rozcieńczenie realnych danych.** Przy tej samej liczbie kroków realne wideo to tylko 31% przykładów, więc model
   widzi je ~3× rzadziej niż przy treningu tylko na X4K.
3. **Inny wygląd scen.** Syntetyk ma obiekty „naklejone” z ostrymi krawędziami, sztuczny ruch i znacznie silniejszy
   RS (wejście 16 dB vs 26 dB). Część tego, czego model się na nim nauczy, nie przenosi się na realne nagrania.

`corr` zyskał, bo nie ma wbudowanej geometrii: każdy dodatkowy, różnorodny przykład ruchu czegoś go uczy.

## 11. Obserwacja poboczna (do sprawdzenia)

15.11 przez pomyłkę przetrenowano model główny na **starszej wersji zbioru (`rs_layers` v1)** i **~3,2× dłużej**
(30 + 10 epok na mixie). Wynik na X4K: **35,51 dB `psnr_full`**, lepiej niż wszystko powyżej. Nie wiadomo, czy to
zasługa danych, czy samej długości treningu: brakuje kontroli „tylko X4K, tyle samo kroków”. Runy:
`2026-11-15_raft_unet_mix_layers1_30ep`, `2026-11-15_raft_unet_ft_mix_layers1_10ep`
(notatnik `10_mix_layers_1.ipynb`).

## 12. Wniosek

> Syntetyczne dane warstwowe poprawiają wszystkie modele na trudnych scenach syntetycznych (+0,5 do +1,9 dB)
> i pomagają na realnym wideo modelowi, który sam uczy się ruchu (`corr`, +0,7 dB). Modelowi z jawną geometrią
> (RAFT + geometria + U-Net) na realnym wideo nie pomagają (−0,1 do −0,2 dB): ruch liczy u niego geometria,
> a dodatkowe dane rozcieńczają realne przykłady.

**Możliwe dalsze kroki:**
- kontrola długości treningu (punkt 11),
- krótkie dostrojenie gotowego modelu z X4K na mixie zamiast treningu od zera,
- losowanie 50/50 X4K i syntetyku,
- bardziej realistyczny syntetyk (rozmycie ruchu, słabszy RS),
- test na prawdziwych nagraniach RS.
