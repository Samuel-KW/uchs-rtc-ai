@echo off
title Robotics AI Training

cd src

set TF_USE_CUDNN=0

py main.py
pause >nul

exit