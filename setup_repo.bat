@echo off
chcp 65001 >nul
rem Jednorazowo: zakłada repozytorium git w tym folderze i robi pierwszy commit.
cd /d "%~dp0"
where git >nul 2>nul || (echo Brak gita. Zainstaluj Git for Windows z https://git-scm.com i uruchom ponownie. & pause & exit /b 1)
if exist .git (echo Repozytorium juz istnieje: & git status --short & pause & exit /b 0)

echo Przenosze stare kopie (sprzed porzadkow) do _stare_do_usuniecia\ ...
if not exist _stare_do_usuniecia mkdir _stare_do_usuniecia
if exist rs_synth move rs_synth _stare_do_usuniecia\ >nul
if exist plan_spotkanie1 move plan_spotkanie1 _stare_do_usuniecia\ >nul
if exist "inz_rollig-shutter_Mach-Kosakowski_stage3 (1).odt" move "inz_rollig-shutter_Mach-Kosakowski_stage3 (1).odt" _stare_do_usuniecia\ >nul
if exist inz_synt.pdf move inz_synt.pdf _stare_do_usuniecia\ >nul

git init -b main
git config user.name >nul 2>nul || git config user.name "Igor Kosakowski"
git config user.email >nul 2>nul || git config user.email "igorkosakowski12@gmail.com"
git add -A
git commit -q -m "Initial project structure: code, configs, notebooks, experiments registry, docs" -m "- src/: generator, dataset, MotionNet + refine pipeline, training with run tracking, inference" -m "- notebooks/: 01 dataset, 02 MotionNet (stage 1), 03 refinement U-Net (stage 2), VS Code + Colab workflow" -m "- experiments/README.md: registry of all training runs; docs/: topic card, meeting plan, figures" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_01Vm4LiuvyQG1QgFd6243big"
echo.
git log --oneline -1
git status --short
echo.
echo Gotowe. Dalej: utworz puste prywatne repo na GitHubie i w terminalu:
echo   git remote add origin https://github.com/LOGIN/NAZWA.git
echo   git push -u origin main
pause
