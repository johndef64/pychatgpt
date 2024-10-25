@echo off

:: Check if conda is installed
conda --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo Conda is not installed. Installing Anaconda...
    :: Download Anaconda installer
    curl -O https://repo.anaconda.com/archive/Anaconda3-2023.03-Windows-x86_64.exe
    :: Install Anaconda
    start /wait Anaconda3-2023.03-Windows-x86_64.exe /InstallationType=JustMe /RegisterPython=0 /S /D=%UserProfile%\Anaconda3
    :: Refresh environment variables
    set "PATH=%UserProfile%\Anaconda3;%UserProfile%\Anaconda3\Library\mingw-w64\bin;%UserProfile%\Anaconda3\Library\usr\bin;%UserProfile%\Anaconda3\Library\bin;%UserProfile%\Anaconda3\Scripts;%PATH%"
    call conda init
) ELSE (
    echo Conda is already installed.
)

:: Create 'myenv' environment and install jupyter
echo Creating env 'myenv' and installing 'jupyter'...
conda create --name myenv python=3.10 -y
conda activate myenv
pip install jupyter

echo Done!
pause