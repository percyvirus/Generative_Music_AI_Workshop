import mido
import numpy as np
from scipy.stats import entropy

class EntropyCalculator:
    def __init__(self):
        pass

    def calculate_entropy_with_window(self, midi_file_path, window_size=5.0, bin_size=0.1):
        midi_file = mido.MidiFile(midi_file_path)

        # Initialize variables
        time_bins = []
        entropy_values = []
        
        current_time = 0
        current_notes = set()
        note_distribution = []

        for msg in midi_file:
            current_time += msg.time
            
            if msg.type == 'note_on' and msg.velocity > 0:
                current_notes.add(msg.note)
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                if msg.note in current_notes:
                    current_notes.remove(msg.note)

            # Calculate entropy in each time bin
            if current_time >= len(time_bins) * bin_size:
                # Flatten the set of current_notes to a sorted list of MIDI notes
                note_counts = np.bincount(list(current_notes), minlength=128)
                note_probs = note_counts / np.sum(note_counts)
                note_distribution.append(note_probs)

            # Calculate entropy with a window
            if current_time >= window_size:
                # Combine note distributions within the window
                combined_distribution = np.mean(note_distribution[-int(window_size / bin_size):], axis=0)
                entropy_value = entropy(combined_distribution)
                entropy_values.append(entropy_value)
                time_bins.append(current_time - window_size / 2)
        
        return time_bins, entropy_values
