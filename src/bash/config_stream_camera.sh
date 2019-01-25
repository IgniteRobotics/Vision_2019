#!/bin/bash

v4l2-ctl --device /dev/streamcam -c exposure_auto_priority=0
# v4l2-ctl --device /dev/streamcam -c focus_auto=0
v4l2-ctl --device /dev/streamcam -c exposure_absolute=150
v4l2-ctl --device /dev/streamcam -c exposure_auto=1

