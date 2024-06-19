import time
import math

import essentia.standard as estd
from essentia.pytools.spectral import hpcpgram

# list_original_songs_path = '/Users/percywbm/Desktop/PERCY/MÀSTER/DATASETS/COVERS80/coversongs/covers32k/list1.list'
# list_cover_songs_paths = '/Users/percywbm/Desktop/PERCY/MÀSTER/DATASETS/COVERS80/coversongs/covers32k/list2.list'
# output_folder = '/Users/percywbm/Desktop/PERCY/MÀSTER/DATASETS/PROPIOS/COVERS80'

class BeatDetection():
    
    def __init__(self):
        pass
    
    def detect_beat(self, audio):
        
        # Compute beat positions and BPM.
        rhythm_extractor = estd.RhythmExtractor2013(method="multifeature")
        bpm, beats, beats_confidence, _, beats_intervals = rhythm_extractor(audio)
        
        return bpm, beats, beats_confidence, _, beats_intervals
