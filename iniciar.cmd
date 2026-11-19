@echo off
rem EixoAI: dois cliques e o app abre no navegador, pedindo só o login.
rem Na primeira vez, prepara o ambiente e cria o .env a partir do .env.exemplo.
chcp 65001 >nul
cd /d "%~dp0"

rem Python: o ambiente do projeto (.venv) ou o que fica fora do OneDrive.
set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=%USERPROFILE%\.venvs\eixoai\Scripts\python.exe"
if exist "%PY%" goto :env

echo Primeira vez neste computador: preparando o ambiente. Leva alguns minutos.
python -m venv .venv
if errorlevel 1 (
  echo Nao encontrei o Python. Instale o Python 3.12 em python.org e rode de novo.
  pause
  exit /b 1
)
set "PY=%~dp0.venv\Scripts\python.exe"
"%PY%" -m pip install --upgrade pip >nul
"%PY%" -m pip install -r requirements.txt
if errorlevel 1 (
  echo A instalacao falhou. Veja a mensagem acima.
  pause
  exit /b 1
)

:env
if exist ".env" goto :rodar
copy ".env.exemplo" ".env" >nul
echo Criei o arquivo .env. Preencha a chave do Groq e os usuarios, salve e rode de novo.
notepad ".env"
exit /b 0

:rodar
cd src
"%PY%" app.py
pause
