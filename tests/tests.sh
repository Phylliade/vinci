#!/usr/bin/env bash

echo "Running import test"
python imports.py

echo "Running DDPG test"
python ddpg.py

echo "Running experiment test"
python experiment.py
