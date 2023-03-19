#!/bin/bash

#cur_dir=`pwd`
#echo $cur_dir
mkdir -p ./build/dist
cd ./build/dist
rm -rf *
cd -
ls |grep -vw build |grep -vw "package.sh" |grep -v grep |xargs cp -r -t ./build/dist
cd ./build/dist
find . -name "__pycache__" -type d -exec rm -rf {} \;
python3 -m compileall -b .
find . -name "*.py" -exec rm -rf {} \;
find . -name "*.pid" -exec rm -rf {} \;
cd -
