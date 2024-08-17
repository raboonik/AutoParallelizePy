#!/bin/bash

: '
    Installation script for AutoParallelizePy.
    Author:  Abbas Raboonik
    Contact: raboonik@gmail.com
    GitID:   https://github.com/raboonik
'

if [ ! "$EUID" -ne 0 ]
then 
    echo "Please do not run as root and try again without sudo!"
    exit
fi

sourceDir=$(pwd)
installationDir=$HOME/.local/lib/AutoParallelizePy/

if [ $# -eq 0 ]
then
    echo "Installing in the default directory under $installationDir"
else
    installationDir=$1
    if [ ! -d $installationDir ]
    then
        echo "The input directory does not exist!"
        exit
    else
        echo "Installing in $installationDir"
        installationDir=$installationDir"AutoParallelizePy"
    fi
fi

IFS='/' read -r -a array <<< "$installationDir"

installationDir=""
for subDir in "${array[@]}"
do
    installationDir=${installationDir}${subDir}"/"
    if [ ! -d "$installationDir" ]; then
        mkdir $installationDir
    fi
done

cp -r $sourceDir"/libs/" $installationDir

if [ -f $sourceDir"/add2path.sh" ]
then
    rm $sourceDir"/add2path.sh"
fi

touch $sourceDir"/add2path.sh"

echo "
#!/bin/bash

: '
    Add AutoParallelizePy library to .bashrc and source it.
'

if [ ! "\$EUID" -ne 0 ]
then 
    echo "Please do not run as root and try again without sudo!"
    exit
fi

if [ ! -f \"$HOME"/.bashrc"\" ]
then
    touch \"$HOME"/.bashrc"\"
fi

addLine=\"export PYTHONPATH='${PYTHONPATH}:${installationDir}libs'\"
if grep -Fxq \$addLine \$HOME"/.bashrc"
then
    echo ".bashrc file already updated"
else
    echo "Adding the AutoParallelizePy library to PYTHONPATH in .bashrc"
    echo "\$addLine" >> \$HOME/.bashrc
    source \$HOME/.bashrc
fi
" >> $sourceDir"/add2path.sh"

chmod +x $sourceDir"/add2path.sh"

if [ -f $sourceDir"/uninstall.sh" ]
then
    rm $sourceDir"/uninstall.sh"
fi

touch $sourceDir"/uninstall.sh"

echo "
#!/bin/bash

: '
    Uninstallation script for AutoParallelizePy.
    Remove libraries and update python the environment.
'

if [ ! "\$EUID" -ne 0 ]
then 
    echo "Please do not run as root and try again without sudo!"
    exit
fi

if [ -f \"$sourceDir"/add2path.sh"\" ]
then
    rm \"$sourceDir"/add2path.sh"\"
fi

if [ -d \"$installationDir\" ]
then
    rm -r $installationDir"\*"
    export PYTHONPATH=\${PYTHONPATH#"\":${installationDir}libs"\"}
    echo "AutoParallelizePy uninstallation successfully!"
else
    echo "AutoParallelizePy not installed or not found at ${installationDir}!"
fi

sed -i \""/\\/AutoParallelizePy\\//d"\" $HOME/.bashrc
source $HOME/.bashrc

rm $sourceDir"/uninstall.sh"
" >> $sourceDir"/uninstall.sh"

chmod +x $sourceDir"/uninstall.sh"

echo "Installation done successfully!"