import speech_recognition as sr #Google api
import soundfile as sf #
import speech_recognition as sr
from os import walk

r = sr.Recognizer()
#optional
#r.energy_threshold = 300

def startConvertion(path = 'test.wav', lang = 'en-IN'):
    with sr.AudioFile(path) as source:
        #print('Fetching File')
        audio_file = r.record(source)
        print(r.recognize_google(audio_file, language=lang))
startConvertion()

def transcribe(audio_file, loops, starting_increment):
    audio_text1 = []
    increment = starting_increment
    for i in range(loops):

        r = sr.Recognizer()
        with sr.AudioFile(audio_file) as source:
            audio = r.record(source, offset = (increment) , duration= 10)
        string = r.recognize_google(audio)
        audio_text1.append(string)
        increment += 10
    return audio_text1


audio_file = 'test.wav' #input audio file
f = sf.SoundFile(audio_file) #read in audio file
vid_len = int(len(f) / f.samplerate) + (len(f) % f.samplerate > 0)    #round up to make sure we get the entire file

loop_num = int(vid_len/10) + (vid_len % 10 > 0) #calculate the number of required loops
if loop_num == 0:
    loop_range= 2
else:
    loop_range = int((loop_num/2))

audio_text1 = transcribe(audio_file, loop_range, 0)
audio_text2 = transcribe(audio_file, loop_range, loop_range)
#concadination
audio_half1 = " ".join(audio_text1)
audio_half2 = " ".join(audio_text2)
full_text = audio_half1 + " " + audio_half2
print(full_text)
words=full_text.split()
counts={}
for i in words:
    if i not in counts:
        counts[i] = 0
    counts[i] += 1
print(counts)
import speech_recognition as sr
from os import walk

r = sr.Recognizer()
#optional
#r.energy_threshold = 300
