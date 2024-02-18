import os
import importlib
import requests

def simple_bool(message):
    choose = input(message+" (y/n): ").lower()
    your_bool = choose in ["y", "yes"]
    return your_bool

def check_and_install_module(module_name):
    try:
        # Check if the module is already installed
        importlib.import_module(module_name)
    except ImportError:
        # If the module is not installed, try installing it
        install = simple_bool(
            "\n" + module_name + "  module is not installed.\nWould you like to install it?")
        if install:
            import subprocess
            subprocess.check_call(["pip", "install", module_name])
            print(f"The module '{module_name}' was installed correctly.")
        else:
            exit()

check_and_install_module('pyaudio')
check_and_install_module('keyboard')
check_and_install_module('pyperclip')
check_and_install_module('openai')
check_and_install_module('pyautogui')
check_and_install_module('wave')

# Note: you need to be using OpenAI Python v0.27.0 for the code below to work

def get_gitfile(url, flag='', dir = os.getcwd()):
    url = url.replace('blob','raw')
    response = requests.get(url)
    file_name = flag + url.rsplit('/',1)[1]
    file_path = os.path.join(dir, file_name)
    with open(file_path, 'wb') as file:
        file.write(response.content)

if not os.path.exists('pychatgpt.py'):
    handle="https://raw.githubusercontent.com/johndef64/pychatgpt/main/"
    files = ["pychatgpt.py"]
    for file in files:
        url = handle+file
        get_gitfile(url)
os.getcwd().endswith('pychatgpt.py')
os.path.exists('pychatgpt.py')

import pychatgpt as op

#----------------------------------------------

import pandas as pd
import pyperclip as pc
import pyautogui
import keyboard
import pyaudio
import wave
import time

input_device_id = 0
audio = pyaudio.PyAudio()


# Parameters
chunk = 1024  # Number of frames per buffer
sample_format = pyaudio.paInt16  # 16 bits per sample
channels = 1  # Mono audio
rate = 44100  # Sampling rate in Hz

# Choose input device
list = []
for index in range(audio.get_device_count()):
    info = audio.get_device_info_by_index(index)
    list.append(f"Device {index}: {info['name']}")
mics = pd.DataFrame(list)
#input_device_id = input("/Select your microphone from the following list:\n"+mics.to_string(index=False))

# Choose whisper mode
translate = simple_bool('Transcribe (n) or Translate (y)?\n')

start = 'Alt+Q'
stop = 'Alt+W'
print("\nTo start record press "+start)
while True:

    if keyboard.is_pressed(start):
        stream = audio.open(format=sample_format,
                            channels=channels,
                            rate=rate,
                            frames_per_buffer=chunk,
                            input=True,
                            #input_device_index=int(input_device_id)
                            )
        frames = []
        print("Recording...")
        print("press "+stop+" to stop")

        while True:
            if keyboard.is_pressed(stop):  # if key 'ctrl + c' is pressed
                break  # finish the loop
            else:
                data = stream.read(chunk)
                frames.append(data)

        print("Finished recording.")


        # Save the audio data to a WAV file
        filename = "recorded_audio.wav"
        wf = wave.open(filename, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(audio.get_sample_size(sample_format))
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))
        wf.close()

        # Wav to Whisper
        #audio_file= open("recorded_audio.wav", "rb")
        if translate:
            op.whisper_translate("recorded_audio.wav", 'text',False)
        else:
            op.whisper("recorded_audio.wav", 'text',False)
        pc.copy(op.transcript)
        print(op.transcript)
        pyautogui.hotkey('ctrl', 'v')

        print("\nTo start record press "+start)

