# rec-whisper-paste

Audio Recording and Transcribing with OpenAI Whisper working in the Background

## Description

The project is a python script that uses the OpenAI library, specifically the Whisper ASR System, to record and transcribe audio working in the background. The code handles microphone selection, facilitates audio recording, transcribes the recorded audio using Whisper, and pastes the transcribed text to the current window.

## Primary Features

- Installs the required python modules if needed.
    - 'pyaudio' for recording audio.
    - 'pyperclip' for clipboard operations.
    - 'openai' for transcribing audio.
    - 'pyautogui' for automation GUI.
    - 'wave' for handling WAV audio format.
    
- Requests the OpenAI API key for authentication and kept for consequent uses.

- Displays a list of the available microphones and allows the user to select which one to use.

- Initiates an audio recording session when the 'alt+a' key combination is pressed. The session can be interrupted by pressing the 'alt+a+s' key combo.

- Saves the recorded audio session in WAV file format.

- Transcribes the recorded audio utilizing OpenAI's Whisper system, stores the transcribed text in the clipboard and pastes in the current window.

## Usage

To start recording, press 'alt+a'. The recording can be interrupted by pressing 'alt+s'. After each recording session, the transcribed text is automatically pasted into the active window.

Note: You need to be using OpenAI Python v0.27.0 for the code to work correctly. 

## Dependencies 

- Python 3.6+
- OpenAI Python v0.27.0
- pyaudio
- pyperclip
- openai
- pyautogui
- wave
- pandas

