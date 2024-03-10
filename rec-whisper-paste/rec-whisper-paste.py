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

from datetime import datetime
import pyperclip as pc
import keyboard as kb
import pandas as pd
import pyautogui
import pyaudio
import wave
import time

input_device_id = 1
audio = pyaudio.PyAudio()

# inizialize log:-----------------------------------
if not os.path.isfile('whisper_log.txt'):
    with open('whisper_log.txt', 'w', encoding= 'utf-8') as file:
        file.write('Rec-Whisper\n\nTranscription LOG:\n')
        print(str('\nwhisper_log.txt created at ' + os.getcwd()))

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
translate = False#simple_bool('Transcribe (n) or Translate (y)?\n')



def stream_on(chunk = 1024,  # Number of frames per buffer
              sample_format = pyaudio.paInt16,  # 16 bits per sample
              channels = 1, # Mono audio
              rate = 44100):  # Sampling rate in Hz
    stream = audio.open(format=sample_format,
                        channels=channels,
                        rate=rate,
                        frames_per_buffer=chunk,
                        input=True,
                        #input_device_index=int(input_device_id)
                        )
    return stream

def save_audio(frames,
               filename= 'recorded_audio.mp3',
               sample_format = pyaudio.paInt16,
               channels = 1,
               rate = 44100):
    # Save the audio data as audio file (wav, mp3)
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(audio.get_sample_size(sample_format))
    wf.setframerate(rate)
    wf.writeframes(b''.join(frames))
    wf.close()

def loop_audio(start='Alt+Q',
               stop='Alt+W'):
    while True:
        if kb.is_pressed(start):
            stream = stream_on(chunk = 1024)

            frames = []
            print("Recording...\npress "+stop+" to stop")

            while True:
                if kb.is_pressed(stop):
                    print("Recording Stopped.")
                    break
                else:
                    data = stream.read(1024)
                    frames.append(data)

            save_audio(frames, 'recorded_audio.mp3')
            break



start = 'alt'
print("\nTo start record press "+start)
while True:

    loop_audio(start,'ctrl')

    # audio file to Whisper
    op.whisper("recorded_audio.mp3", translate,'text',False)

    pc.copy(op.transcript)
    print(op.transcript)
    pyautogui.hotkey('ctrl', 'v')

    with open('whisper_log.txt', 'a', encoding= 'utf-8') as file:
        file.write('---------------------------')
        file.write('\n'+str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))+'\n')
        file.write(op.transcript+ '\n')
    print("\nTo start record press "+start)


#%%
