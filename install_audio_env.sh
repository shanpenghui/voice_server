#!/usr/bin/env bash
set -e
sudo apt update
sudo apt install -y python3-pip python3-venv build-essential portaudio19-dev libasound2-dev libpulse-dev
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install sounddevice numpy websockets pynput
echo "DONE. Now you can run voice_client.py"

