import sys
import glob
from pydub import AudioSegment
import os

#Path to all mp3 files
files_path = os.path.join("audio_dataset","Actor_01",'/*.mp3')
print(files_path)

for file in sorted(glob.iglob(files_path)):
    name = file.split('.mp3')[0]
    segment=AudioSegment.from_mp3(file)
    segment.export(name + '.wav',format='wav')
