import essentia.standard as estd
import numpy as np
import math
import mido
import pretty_midi

class AudioFileToMidiFile():
    
    def __init__(self):
        pass
    
    def execute(self, audio_file_path, audio_file_features):
        
        frame_size = audio_file_features[1] * 0.1
        frame_size = int(2 ** math.ceil(math.log2(frame_size)))
        hop_size = int(frame_size/2)
        # Load audio file.
        # It is recommended to apply equal-loudness filter for PredominantPitchMelodia.
        loader = estd.EqloudLoader(filename=audio_file_path, sampleRate=audio_file_features[1])
        audio = loader()

        # Extract the pitch curve
        # PitchMelodia takes the entire audio signal as input (no frame-wise processing is required).

        pitch_extractor = estd.PredominantPitchMelodia(frameSize=frame_size, hopSize=hop_size)
        pitch_values, pitch_confidence = pitch_extractor(audio)

        # Pitch is estimated on frames. Compute frame time positions.
        pitch_times = np.linspace(0.0,len(audio)/audio_file_features[1],len(pitch_values) )
        
        onsets, durations, notes = estd.PitchContourSegmentation(hopSize=hop_size)(pitch_values, audio)
        
        PPQ = 96 # Pulses per quarter note.
        BPM = 120 # Assuming a default tempo in Ableton to build a MIDI clip.
        tempo = mido.bpm2tempo(BPM) # Microseconds per beat.

        # Compute onsets and offsets for all MIDI notes in ticks.
        # Relative tick positions start from time 0.
        offsets = onsets + durations
        silence_durations = list(onsets[1:] - offsets[:-1]) + [0]

        mid = mido.MidiFile()
        track = mido.MidiTrack()
        mid.tracks.append(track)

        # Filter out NaN values from notes, onsets, durations, and silence_durations
        valid_indices = ~np.isnan(notes)
        filtered_notes = notes[valid_indices]
        filtered_onsets = onsets[valid_indices]
        filtered_durations = durations[valid_indices]
        filtered_silence_durations = np.array(silence_durations)[valid_indices]

        for note, onset, duration, silence_duration in zip(list(notes), list(onsets), list(durations), silence_durations):
            track.append(mido.Message('note_on', note=int(note), velocity=64,
                                    time=int(mido.second2tick(duration, PPQ, tempo))))
            track.append(mido.Message('note_off', note=int(note),
                                    time=int(mido.second2tick(silence_duration, PPQ, tempo))))

        
        return pitch_values, pitch_confidence, pitch_times, track
