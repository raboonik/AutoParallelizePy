
#!/bin/bash

: '
    Uninstallation script for AutoParallelizePy.
    Remove libraries and update python the environment.
'

if [ ! $EUID -ne 0 ]
then 
    echo Please do not run as root and try again without sudo!
    exit
fi

if [ -f "/home/abbas/synced_XPS_iMac/python_codes/python_workplace/AutoParallelizePy/add2path.sh" ]
then
    rm "/home/abbas/synced_XPS_iMac/python_codes/python_workplace/AutoParallelizePy/add2path.sh"
fi

if [ -d "/home/abbas/.local/lib/AutoParallelizePy/" ]
then
    rm -r /home/abbas/.local/lib/AutoParallelizePy/*
    export PYTHONPATH=${PYTHONPATH#":/home/abbas/.local/lib/AutoParallelizePy/libs"}
    echo AutoParallelizePy uninstallation successfully!
else
    echo AutoParallelizePy not installed or not found at /home/abbas/.local/lib/AutoParallelizePy/!
fi

sed -i "/\/AutoParallelizePy\//d" /home/abbas/.bashrc
source /home/abbas/.bashrc

rm /home/abbas/synced_XPS_iMac/python_codes/python_workplace/AutoParallelizePy/uninstall.sh

