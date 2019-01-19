#!/bin/bash

# this is a script to sort the cams into predictable values.  
# the cam serial number must be in the CAM_MAP.

declare -A CAM_MAP
CAM_MAP[046d_0821_4804CA90]=cam0  # logitech
CAM_MAP[HD_Camera_Manufacturer_USB_2.0_Camera]=cam1  #fisheye.  no really.


for cam in $(ls /dev/video*); do
        echo camera: $cam
        #sudo udevadm info --query=all /dev/video0 | grep 'VENDOR_ID\|MODEL_ID\|SERIAL_SHORT'
        #cam_info=$(sudo udevadm info --query=all $cam | grep 'SERIAL_SHORT' | cut -d'=' -f2)
        cam_info=$(sudo udevadm info --query=all $cam | grep 'SERIAL=' | cut -d'=' -f2)
        echo $cam_info
        sudo ln -s $cam /dev/${CAM_MAP[$cam_info]}   
done