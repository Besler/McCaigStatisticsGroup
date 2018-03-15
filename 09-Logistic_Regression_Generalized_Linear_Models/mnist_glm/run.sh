#! /usr/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

xhost +
docker run \
  --rm \
  -v $DIR:/project \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -e DISPLAY=unix:0\
  mipcpu python3 /project/main.py
