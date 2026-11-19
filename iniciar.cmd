@echo off
rem EixoAI: dois cliques e o app abre no navegador, pedindo só o login.
rem Na primeira vez, prepara o ambiente e cria o .env a partir do .env.exemplo.
rem Se a instalação for interrompida, rodar de novo continua a instalação:
rem ela só conta como feita quando o arquivo .venv\instalado.ok existe.
chcp 65001 >nul
cd /d "%~dp0"

rem 1) Ambiente do projeto (.venv): usa se a instalação terminou; senão, termina agora.
set "PY=%~dp0.venv\Scripts\python.exe"
if exist "%PY%" (
  if exist "%~dp0.venv\instalado.ok" goto :env
  goto :instalar
)

rem 2) Ambiente fora do OneDrive, se já existir.
set "PY=%USERPROFILE%\.venvs\eixoai\Scripts\python.exe"
if exist "%PY%" goto :env

rem 3) Primeira vez: cria o ambiente.
echo Primeira vez neste computador: preparando o ambiente. Leva alguns minutos.
python -m venv .venv
if errorlevel 1 (
  echo Nao encontrei o Python. Instale o Python 3.12 em python.org, marcando "Add python.exe to PATH", e rode de novo.
  pause
  exit /b 1
)

:instalar
set "PY=%~dp0.venv\Scripts\python.exe"
echo Instalando as bibliotecas. Pode levar de 10 a 20 minutos na primeira vez.
"%PY%" -m pip install --upgrade pip >nul
"%PY%" -m pip install -r requirements.txt
if errorlevel 1 (
  echo A instalacao falhou. Confira a internet e rode o iniciar.cmd de novo: ele continua daqui.
  pause
  exit /b 1
)
echo ok> "%~dp0.venv\instalado.ok"

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
