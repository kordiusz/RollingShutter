# Kompensacja rolling shutter z dwóch prostopadłych kamer

Praca inżynierska (AGH, Instytut Telekomunikacji, opiekun: dr inż. Jarosław Bułat).
Autorzy: Igor Kosakowski (dane syntetyczne, modele), Mach (stanowisko, dane rzeczywiste).

Dwie kamery RS obok siebie, kamera B obrócona o 90° (czyta kolumny zamiast wierszy).
Model dostaje po jednej klatce z każdej kamery i zwraca klatkę kamery A bez zniekształceń RS.

## Struktura

```
src/                 kod (generatory, modele, trening, demo); opis techniczny: src/README.md
  synth_rs.py        generator: zdjęcie jako płaska scena + ruch kamery (baza dla pozostałych generatorów)
  synth_video.py     generator z nagrań 1000 fps X4K (ruch kamery i obiektów) -> rs_video_x4k
  synth_video_v2.py  jw., z filtrem słabych próbek i większą liczbą klipów (zbiór rozszerzony)
  synth_layers.py    generator warstwowy (Filip): tło + wycięte obiekty na różnych głębokościach
  make_cutouts.py    wycinanie obiektów (PNG z przezroczystością) do synth_layers.py
  gallery.py         galeria podglądów zbioru
  raft_geom.py       RAFT + geometria RS, Dataset (RSSampleDataset)
  raft_unet.py       modele: RAFT + geometria + U-Net, zwykły U-Net, autoenkoder, enkoder + korelacja
  train_raft.py      ewaluacja / dostrajanie RAFT + geometria
  train_unet.py      trening / ewaluacja modeli z raft_unet.py (zapisuje run_info.json, results.json, eval.json)
  train_common.py    metryki (maskowane PSNR, SSIM) i zapis informacji o runie
  model.py           klocki sieci: U-Net, warp, warstwa korelacji
  demo.py            demo na próbkach ze zbioru lub parze obrazów (PNG + GIF przed/po)
  demo_real.py       demo na parze zdjęć z prawdziwych kamer (np. wiatrak)
  preview.py         podgląd zbioru
  project_paths.py   układ folderów na Google Drive (używany przez notatniki)
configs/             parametry generatorów (kamery, zakresy ruchu)
notebooks/           notatniki Colab, numerowane w kolejności użycia; jak trenować: notebooks/README.md
  04_raft_geom.ipynb     RAFT + geometria: ewaluacja, dostrajanie, demo wiatraka
  05_unet.ipynb          RAFT + geometria + U-Net poprawiający (najlepszy model, 35,81 dB)
  06_unet_plain.ipynb    zwykły U-Net bez RAFT i geometrii (punkt odniesienia, 30 epok)
  07_demo_v2.ipynb       geometria v2, dostrojenie RAFT z v2, demo (koło, próbki ze zbioru)
  08_synteza.ipynb       synteza obrazu wprost: zwykły U-Net od zera (długo), autoenkoder, enkoder + korelacja
  09_ablacje.ipynb       ablacje: bez kamery B (U-Net, corr) i korelacja + geometria (czy kluczem jest geometria, czy RAFT)
experiments/README.md  REJESTR wszystkich treningów: co, kiedy, na czym, wyniki, gdzie wagi
docs/                karta tematu, plan spotkań, wyniki; literatura (PDF-y tylko lokalnie)
  etap1_motionnet.md   etap 1 (MotionNet, sam ruch kamery, 13.10): opis, wyniki, jak odtworzyć usunięty kod
```

Duże pliki **nie są w gicie**: zbiory, wagi (`*.pt`) i wyniki są na Google Drive.

```
MyDrive/rs_project/
  datasets/rs_video_x4k.zip, ...                   zbiory (rs_dataset_256.zip: etap 1)
  real/wiatrak/kamera_A.jpg, kamera_B.jpg          zdjęcia z kamer do demo
  runs/<RRRR-MM-DD>_<nazwa>/                       jeden folder na trening: best.pt, last.pt,
                                                   log.csv, run_info.json (komenda, commit kodu, GPU,
                                                   parametry zbioru), results.json, eval.json, podglądy
  .github_token                                    (opcjonalnie) token do klonowania repo w Colabie
```

## Jak pracujemy (VS Code + rozszerzenie Colab + GitHub)

Notatnik edytujesz i uruchamiasz w VS Code, ale **kod wykonuje się na maszynie Colaba**, która nie widzi plików
z Twojego dysku. Dlatego kod trafia tam z GitHuba, a dane i wagi z Google Drive.

1. Zmieniasz kod w `src/` → **commit + push** (VS Code: Source Control).
2. W notatniku uruchamiasz komórkę **SETUP**: montuje Drive, robi `git clone`/`git pull`, wypisuje commit kodu.
3. Trenujesz; wyniki lądują w `MyDrive/rs_project/runs/<data>_<nazwa>/`.
4. Dopisujesz run do `experiments/README.md` i commitujesz notatnik razem z wynikami (widać, co wyszło).

Rozszerzenie Colab w VS Code: `drive.mount()` działa, **sekrety Colaba (`userdata`) nie**, `files.download()` nie.
Dlatego token GitHuba jest wpisywany przy starcie (albo zapisany w `MyDrive/rs_project/.github_token`),
a wagi do testów lokalnych pobierasz z Google Drive.

### Pierwsza konfiguracja (raz)
1. **Git dla Windows** (jeśli nie ma): https://git-scm.com, potem w terminalu VS Code:
   `git config --global user.name "Imię Nazwisko"` i `git config --global user.email "..."`.
2. **GitHub:** utwórz puste **prywatne** repozytorium (bez README), potem w terminalu w tym folderze:
   `git remote add origin https://github.com/LOGIN/NAZWA.git` i `git push -u origin main`.
   Dodaj współautora: Settings → Collaborators.
3. **Token do Colaba:** GitHub → Settings → Developer settings → Fine-grained tokens: tylko to repozytorium,
   uprawnienie *Contents: Read-only*. Wpisujesz go przy pierwszym uruchomieniu SETUP albo zapisujesz w pliku
   `MyDrive/rs_project/.github_token`.
4. **VS Code:** zainstaluj rozszerzenia *Jupyter* i *Google Colab*. Otwórz notatnik → *Select Kernel* → *Colab* →
   wybierz GPU → zaloguj się kontem z Colab Pro.
5. W komórce SETUP każdego notatnika wpisz `REPO = 'LOGIN/NAZWA'`.

## Lokalnie (bez GPU)

```bash
pip install -r requirements.txt
python src/demo.py examples/kolo examples/x4k_val_00273_02 --out demo.png --gif demo.gif              # RAFT + geometria
python src/demo.py examples/kolo --unet-ckpt best.pt --out unet.png --gif unet.gif                       # + U-Net
python src/demo_real.py --a kameraA.jpg --b kameraB.jpg --center-a 320 240 --t-full 30 --out wynik.png   # prawdziwe kamery
```
