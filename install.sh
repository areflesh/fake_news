#!/bin/bash

# Create a virtual environment
python3 -m venv env

# Activate the virtual environment
source env/bin/activate

# Install the packages
pip install -e .

# Deactivate the virtual environment
deactivate