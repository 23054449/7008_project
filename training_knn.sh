#!/usr/bin/bash
# file name: training_knn.sh

# change to directory
echo "Current directory: $(pwd)"
cd /home/ubuntu/7008_project
echo "Current directory: $(pwd)"
#echo "Directory changed"

# run python script
echo "Execution start"
/usr/bin/python3 /home/ubuntu/7008_project/scripts/training.py /home/ubuntu/7008_project/temp/cleanTrain.csv /home/ubuntu/7008_project/temp/cleanTest.csv knn
echo "Execution complete"
