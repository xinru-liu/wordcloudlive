@echo off

echo Creating virtual environment...
python -m venv venv

echo Activating virtual environment...
call venv\Scripts\activate

echo Upgrading pip...
pip install --upgrade pip

echo Installing requirements...
pip install -r requirements.txt

echo Running your Python script...
python app.py

echo Done.
pause
