@echo off

:: Create 'myenv' environment and install jupyter
echo Creating env 'myenv' and installing 'jupyter'...
conda create --name myenv python=3.10 -y
conda activate myenv
pip install jupyter

echo Done!
pause