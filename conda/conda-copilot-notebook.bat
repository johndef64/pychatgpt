@echo off
echo Opening Anaconda Prompt and changing directory...

REM Get the current working directory
set "current_dir=%cd%"

REM Replace "myenv" with your conda enviroment
start cmd /k "call C:\Users\Utente\anaconda3\Scripts\activate.bat && cd /d %current_dir% && conda activate myenv && jupyter notebook --NotebookApp.token='' --NotebookApp.password='' "%current_dir%\copilot_notebook.ipynb" "