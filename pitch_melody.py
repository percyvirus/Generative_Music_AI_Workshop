import time
import math
import numpy as np
import essentia.standard as estd
from essentia.pytools.spectral import hpcpgram

# list_original_songs_path = '/Users/percywbm/Desktop/PERCY/MÀSTER/DATASETS/COVERS80/coversongs/covers32k/list1.list'
# list_cover_songs_paths = '/Users/percywbm/Desktop/PERCY/MÀSTER/DATASETS/COVERS80/coversongs/covers32k/list2.list'
# output_folder = '/Users/percywbm/Desktop/PERCY/MÀSTER/DATASETS/PROPIOS/COVERS80'

class PitchMelody():
    
    def __init__(self):
        pass
    
    def extract_pitch_melody(self, song_path, song_audio_features):
        
        frame_size = song_audio_features[1] * 0.1
        frame_size = int(2 ** math.ceil(math.log2(frame_size)))
        hop_size = int(frame_size/2)
        
        loader = estd.EqloudLoader(filename=song_path, sampleRate=song_audio_features[1])
        audio = loader()
        
        pitch_extractor = estd.PredominantPitchMelodia(frameSize=frame_size, hopSize=hop_size)
        pitch_values, pitch_confidence = pitch_extractor(audio)
        pitch_times = np.linspace(0.0,len(audio)/song_audio_features[1],len(pitch_values) )
        
        
        return pitch_values, pitch_confidence, pitch_times, hop_size, audio
