import essentia.standard as estd

class Note_segmentation_and_converting_to_MIDI():
    
    def __init__(self):
        pass
    
    def extract_pitch_MIDI(self, pitch_values, audio, hop_size):
        
        onsets, durations, notes = estd.PitchContourSegmentation(hopSize=hop_size)(pitch_values, audio)

        return onsets, durations, notes
