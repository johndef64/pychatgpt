import os
import importlib

def get_boolean_input(prompt):
    while True:
        try:
            return {"1": True, "0": False}[input(prompt).lower()]
        except KeyError:
            print("Invalid input. Please enter '1' or '0'.")

def check_and_install_module(module_name):
    try:
        # Check if the module is already installed
        importlib.import_module(module_name)
        #import module_name
        #print(f"The module '{module_name}' is already installed.")
    except ImportError:
        # If the module is not installed, try installing it
        x = get_boolean_input(
            "\n" + module_name + "  module is not installed.\nwould you like to install it? (yes:1/no:0):")
        if x:
            import subprocess
            subprocess.check_call(["pip", "install", module_name])
            print(f"The module '{module_name}' was installed correctly.")
        else:
            exit()

check_and_install_module('pyaudio')
check_and_install_module('pyperclip')
check_and_install_module('openai')
check_and_install_module('pyautogui')
check_and_install_module('wave')

# Note: you need to be using OpenAI Python v0.27.0 for the code below to work
import openai

current_dir = os.getcwd()
api_key = None
if not os.path.isfile(current_dir + '/openai_api_key.txt'):
    with open(current_dir + '/openai_api_key.txt', 'w') as file:
        file.write(input('insert here your openai api key:'))

api_key = open(current_dir + '/openai_api_key.txt', 'r').read()
openai.api_key = str(api_key)

#----------------------------------------------

import time
import pyperclip
import pyautogui
import pyaudio
import wave

import pandas as pd

input_device_id = 0

print('\nSeleziona il tuo microfono dalla seguente lista:')
audio = pyaudio.PyAudio()

list = []
for index in range(audio.get_device_count()):
    info = audio.get_device_info_by_index(index)
    list.append(f"Device {index}: {info['name']}")
mics = pd.DataFrame(list)
input_device_id = input("Seleziona il tuo microfono dalla seguente lista:"+mics.to_string(index=False))

chunk = 1024  # Number of frames per buffer
sample_format = pyaudio.paInt16  # 16 bits per sample
channels = 1  # Mono audio
rate = 44100  # Sampling rate in Hz

import msvcrt
import keyboard

def get_key_press():
    while True:
        if msvcrt.kbhit():  # Check if a key is pressed
            key = msvcrt.getch().decode()  # Get the pressed key
            return key  # Return the pressed key

print("\nTo start record press alt+a")
while True:

    if keyboard.is_pressed('alt+a'):
        stream = audio.open(format=sample_format,
                            channels=channels,
                            rate=rate,
                            frames_per_buffer=chunk,
                            input=True,
                            input_device_index=int(input_device_id))

        frames = []
        print("Recording...")
        print("press alt+s to interrupt")

        while True:
            if keyboard.is_pressed('alt+s'):  # if key 'ctrl + c' is pressed
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

        audio_file= open("recorded_audio.wav", "rb")
        transcript = openai.Audio.transcribe("whisper-1", audio_file)
        text_value = transcript['text']
        pyperclip.copy(text_value)
        #pyperclip.paste()
        pyautogui.hotkey('ctrl', 'v')
        print('\n',text_value)
