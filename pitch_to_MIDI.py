import essentia.standard as estd
import mido
import numpy as np

class Pitch_to_MIDI():
    
    def __init__(self):
        pass
    
    def save_pitch_to_MIDI(self, file_path, onsets, durations, notes):
        
        PPQ = 96 # Pulses per quarter note.
        BPM = 60 # Assuming a default tempo in Ableton to build a MIDI clip.
        tempo = mido.bpm2tempo(BPM) # Microseconds per beat.
        
        # Clean the data by removing NaN and infinite values
        valid_indices = ~np.isnan(onsets) & ~np.isnan(durations) & ~np.isnan(notes)
        valid_indices &= ~np.isinf(onsets) & ~np.isinf(durations) & ~np.isinf(notes)
        onsets = onsets[valid_indices]
        durations = durations[valid_indices]
        notes = notes[valid_indices]

        # Compute onsets and offsets for all MIDI notes in ticks.
        # Relative tick positions start from time 0.
        offsets = onsets + durations
        silence_durations = list(onsets[1:] - offsets[:-1]) + [0]

        mid = mido.MidiFile()
        track = mido.MidiTrack()
        mid.tracks.append(track)

        for note, onset, duration, silence_duration in zip(list(notes), list(onsets), list(durations), silence_durations):
            track.append(mido.Message('note_on', note=int(note), velocity=64,
                                    time=int(mido.second2tick(duration, PPQ, tempo))))
            track.append(mido.Message('note_off', note=int(note),
                                    time=int(mido.second2tick(silence_duration, PPQ, tempo))))

        midi_file = file_path.replace(".wav","_extracted_melody.mid")
        mid.save(midi_file)
        print("MIDI file location:", midi_file) 