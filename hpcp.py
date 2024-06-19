import math

import essentia.standard as estd
from essentia.pytools.spectral import hpcpgram

# list_original_songs_path = '/Users/percywbm/Desktop/PERCY/MÀSTER/DATASETS/COVERS80/coversongs/covers32k/list1.list'
# list_cover_songs_paths = '/Users/percywbm/Desktop/PERCY/MÀSTER/DATASETS/COVERS80/coversongs/covers32k/list2.list'
# output_folder = '/Users/percywbm/Desktop/PERCY/MÀSTER/DATASETS/PROPIOS/COVERS80'

class HPCP():
    
    def __init__(self):
        pass
    
    def extract_HPCPs(self, song_audio, song_audio_features, num_bins=12, min_frequency=50, max_frequency=5000):
        
        # Load song with MonoLoader algorithm, it returns:
        # · [0] audio (vector_real) - the audio signal
        
        frame_size = song_audio_features[1] * 0.1
        frame_size = int(2 ** math.ceil(math.log2(frame_size)))
        hop_size = int(frame_size/2)
        
        # Extracting HPCP features from audio loaded with AudioLoader algorithm
        hpcp = hpcpgram(song_audio, sampleRate=song_audio_features[1], frameSize=frame_size, hopSize=hop_size, numBins=num_bins, minFrequency=min_frequency, maxFrequency=max_frequency)
        
        return hpcp
