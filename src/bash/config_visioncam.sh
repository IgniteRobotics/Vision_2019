#!/bin/bash

v4l2-ctl --device /dev/visioncam -c exposure_auto=1
v4l2-ctl --device /dev/visioncam -c exposure_auto_priority=0
v4l2-ctl --device /dev/visioncam -c exposure_absolute=3
v4l2-ctl --device /dev/visioncam -c brightness=0