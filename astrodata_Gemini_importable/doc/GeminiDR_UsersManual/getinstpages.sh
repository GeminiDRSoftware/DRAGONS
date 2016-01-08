#!/bin/bash

sourcedir=$1

if [ ! -e conf.py ]; then
    ln -s ../../usermanuals/${sourcedir}/*.rst .
    mkdir primitives_pages
    cd primitives_pages
    ln -s ../../../usermanuals/${sourcedir}/primitives_pages/*.rst .
    cd ..
    rm index.rst
    rm index-latex.rst
else
    echo 'cd to the instrument directory'
fi

