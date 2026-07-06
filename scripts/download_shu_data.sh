#!/bin/bash

# Prompt the user for the password (input hidden)
wget -c https://ndownloader.figshare.com/files/36728994

source .venv/bin/activate
python scripts/unzip_file.py

mkdir pretrained_weights
cd pretrained_weights
wget https://huggingface.co/weighting666/CBraMod/resolve/main/pretrained_weights.pth
