import essentia.standard as estd

class BeatDetection():
    
    def __init__(self):
        pass
    
    def detect_beat(self, audio):
        
        # Compute beat positions and BPM.
        rhythm_extractor = estd.RhythmExtractor2013(method="multifeature")
        bpm, beats, beats_confidence, _, beats_intervals = rhythm_extractor(audio)
        
        return bpm, beats, beats_confidence, _, beats_intervals
