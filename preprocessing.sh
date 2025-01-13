#!/usr/bin/bash
# file name: preprocessing.sh

# change to directory
echo "Current directory: $(pwd)"
cd /home/ubuntu/7008_project
echo "Current directory: $(pwd)"
#echo "Directory changed"
 
#echo "Installing required packages..."
#pip install -r requirements.txt
 
# run python script
echo "Execution start"
/usr/bin/python3 /scripts/preprocessing.py --source /temp/rawData.csv
#/usr/bin/python3 test.py
echo "Execution complete"
