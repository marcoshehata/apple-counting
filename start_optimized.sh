#!/bin/bash
# Optimized startup for Apple Counting on Jetson Thor
export USE_TENSORRT=true
# Ensure port 8000 is free
fuser -k 8000/tcp || true
./venv_optimized/bin/python3 app.py
