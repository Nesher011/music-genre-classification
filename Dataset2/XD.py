import os
import shutil
from pydub import AudioSegment

source_folder = r"C:\Users\jedrz\Documents\Repositories\music-genre-classification\Dataset2\Data\genres\world\\"
destination_folder = r"C:\Users\jedrz\Documents\Repositories\music-genre-classification\Dataset2\Data\genres\world\\"

counter=1
# fetch all files
for song in os.listdir(source_folder):
    # construct full file path
    source = source_folder+ song

    destination = destination_folder + str(counter)+".wav"
    # copy only files
    print(source)
    sound = AudioSegment.from_mp3(source)
    sound.export(destination, format="wav")
    print(source)
    #shutil.move(source, destination)
    #print('copied', song)
    counter=counter+1