#!/bin/bash

# this is a script to sort the cams into predictable values.  
# the cam serial number must be in the CAM_MAP.

declare -A CAM_MAP
CAM_MAP[046d_0821_4804CA90]=visioncam  # logitech c910
CAM_MAP[046d_HD_Pro_Webcam_C920_B778927F]=visioncam # c920
CAM_MAP[046d_HD_Pro_Webcam_C920_7F11382F]=visioncam # c920
CAM_MAP[HD_Camera_Manufacturer_USB_2.0_Camera]=streamcam  #fisheye.  no really.


for cam in $(ls /dev/video*); do
        #sudo udevadm info --query=all /dev/video0 | grep 'VENDOR_ID\|MODEL_ID\|SERIAL_SHORT'
        #cam_info=$(sudo udevadm info --query=all $cam | grep 'SERIAL_SHORT' | cut -d'=' -f2)
        cam_info=$(sudo udevadm info --query=all $cam | grep 'SERIAL=' | cut -d'=' -f2)
        echo "camera $cam" 
        if [[ -v CAM_MAP[$cam_info] ]]; then
            echo "found $cam : $cam_info";
            sudo rm /dev/${CAM_MAP[$cam_info]}
            sudo ln -s $cam /dev/${CAM_MAP[$cam_info]}
        fi        
done