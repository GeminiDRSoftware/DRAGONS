#!/bin/bash

sourcedir=$1

if [ ! -e conf.py ]; then
    ln -s ../../usermanuals/${sourcedir}/*.rst .
    ln -s ../../usermanuals/${sourcedir}/primitives_pages .
    rm index.rst
    rm index-latex.rst
else
    echo 'cd to the instrument directory'
fi

