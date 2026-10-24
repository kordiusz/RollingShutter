# Kompensacja rolling shutter z dwóch prostopadłych kamer

Praca inżynierska (AGH, Instytut Telekomunikacji, opiekun: dr inż. Jarosław Bułat).
Autorzy: Igor Kosakowski (dane syntetyczne, modele), Mach (stanowisko, dane rzeczywiste).

Dwie kamery RS obok siebie, kamera B obrócona o 90° (czyta kolumny zamiast wierszy).
Model dostaje po jednej klatce z każdej kamery i zwraca klatkę kamery A bez zniekształceń RS.

## Struktura

```
src/                 kod (generator, dataset, modele, trening, inferencja); opis techniczny: src/README.md
  synth_rs.py        generator danych syntetycznych (zdjęcia + ruch kamery)
  synth_video.py     generator z nagrań 1000 fps (ruch kamery i obiektów)
  raft_geom.py       RAFT + geometria RS (korekcja ruchu obiektów)
  train_raft.py      ewaluacja / dostrajanie RAFT + geometria
  demo_real.py       demo na parze zdjęć z kamer (np. wiatrak)
  rs_dataset.py      Dataset PyTorch (wejścia, wzorzec, pola przesunięć, maski)
  model.py           MotionNet (etap 1), RefinePipeline = MotionNet + U-Net (etap 2), U-Net porównawczy
  train.py           trening i ewaluacja (zapisuje run_info.json, results.json, eval.json)
  infer.py           uruchamianie gotowego modelu (CPU wystarczy), także na klatkach z kamer
  preview.py         podgląd zbioru
  project_paths.py   układ folderów na Google Drive (używany przez notatniki)
configs/             parametry generatora (kamery, zakresy ruchu)
notebooks/           notatniki Colab, numerowane w kolejności użycia
  01_dataset.ipynb       generowanie zbioru -> Drive
  02_motionnet.ipynb     etap 1 (wytrenowany 13.10), ewaluacja / ponowny trening
  03_refine_unet.ipynb   etap 2 (U-Net doszlifowujący)
  04_raft_geom.ipynb     RAFT + geometria: ewaluacja, dostrajanie, demo wiatraka  <- AKTUALNY
  archiwum/              stare notatniki (do odtworzenia historii)
experiments/README.md  REJESTR wszystkich treningów: co, kiedy, na czym, wyniki, gdzie wagi
docs/                karta tematu, plan spotkań, rysunki do pracy; literatura (PDF-y tylko lokalnie)
```

Duże pliki **nie są w gicie**: zbiory, wagi (`*.pt`) i wyniki są na Google Drive.

```
MyDrive/rs_project/
  datasets/rs_dataset_256.zip, rs_video_x4k.zip   zbiory
  real/wiatrak/kamera_A.jpg, kamera_B.jpg          zdjęcia z kamer do demo
  runs/<RRRR-MM-DD>_<nazwa>/                       jeden folder na trening: best.pt, last.pt,
                                                   log.csv, run_info.json (komenda, commit kodu, GPU,
                                                   parametry zbioru), results.json, eval.json, podglądy
  results/<run>/                                   wyniki infer.py
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
python src/infer.py --ckpt best.pt --data <zbiór>/val --n 8 --out results
python src/infer.py --ckpt best.pt --a kameraA.png --b kameraB.png --f 620 --out results
```
