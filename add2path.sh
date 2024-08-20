
#!/bin/bash

: '
    Add AutoParallelizePy library to .bashrc and source it.
'

if [ ! $EUID -ne 0 ]
then 
    echo Please do not run as root and try again without sudo!
    exit
fi

if [ ! -f "/home/abbas/.bashrc" ]
then
    touch "/home/abbas/.bashrc"
fi

addLine="export PYTHONPATH=':/home/abbas/.local/lib/AutoParallelizePy/libs'"
if grep -Fxq $addLine $HOME/.bashrc
then
    echo .bashrc file already updated
else
    echo Adding the AutoParallelizePy library to PYTHONPATH in .bashrc
    echo $addLine >> $HOME/.bashrc
    source $HOME/.bashrc
fi

