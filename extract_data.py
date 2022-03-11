from os import listdir
from os.path import isfile, join
import os
import subprocess
import sys


old_path = os.path.join("audio_dataset") #for all OS
new_path = os.path.join("new_audio_dataset")

files = []
data_folders = os.listdir(old_path) #Lists all folders in the directory

for i in data_folders: #search in each folder
	temp_join = join(old_path,i)
	for j in listdir(temp_join):
		if isfile(join(temp_join,j)):
			files.append(j)
			temp = j.split('.')[0].split('-')
			subprocess.call("move %s %s" % (join(temp_join,j),join(new_path,temp[2])),shell=True)