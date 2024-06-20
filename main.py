import essentia.standard as estd
import numpy as np
from scipy.stats import entropy
import matplotlib.pyplot as plt
import os
from audio_beat_detection import AudioBeatDetection
from audio_pitch_melody import AudioPitchMelody
from note_segmentation_and_converting_to_MIDI import Note_segmentation_and_converting_to_MIDI
from pitch_to_MIDI import Pitch_to_MIDI
import csv

from midi_notes_extractor import MidiNotesExtractor
from midi_entropy_calculator import MidiEntropyCalculator
from hpcp import HPCP
from midi_notes_extractor import MidiNotesExtractor
from audio_file_to_midi_file import AudioFileToMidiFile
from changeComplexity import ChangeComplexity
import mido

import MarkovProb as mp

prints = False

midi_file_path = './data/midi_files/Complex_4.mid'
audio_file_path = "./data/audio_files/AS_escala_alba_A.wav"

input_matrix_path = "/Users/percywbm/Desktop/PERCY/Generative Music AI Workshop/generativemusicaicourse/Generative_Music_AI_Workshop/output.csv"

# Initialize an empty list to store the data
list_of_lists = []

with open(input_matrix_path, mode='r') as file:
    for line in file:
        # Split each line by the tab character and append it to the list_of_lists
        row = line.strip().split("\t")
        list_of_lists.append(row)


# Initialize an empty dictionary
note_dict = {}

# Iterate over each sublist in data
for sublist in list_of_lists:
    # Extract the note name and numerical values
    note_data = sublist[0].split(',')
    note_name = note_data[0]
    numerical_values = [float(value) for value in note_data[1:]]
    
    # Store in the dictionary
    note_dict[note_name] = numerical_values

# MIDI file
midi_notes_extractor = MidiNotesExtractor()
# (Midi note, duration in ticks, duration in seconds)
#original_midi_file_pitches, original_midi_file_durations, original_midi_durations_in_seconds, original_midi_file_onsets, original_midi_onsets_in_seconds = midi_notes_extractor.extract_midi_notes(midi_file_path, channel=0)
original_midi_file_onsets, original_midi_file_durations, original_midi_file_pitches, original_midi_onsets_in_seconds, original_midi_durations_in_seconds, original_midi_file_note_names = midi_notes_extractor.extract_midi_notes(midi_file_path, channel=0)
original_time_midi_file_pitches = np.cumsum(original_midi_file_durations)

print(f"Original midi pitches: {original_midi_file_note_names}")
print(f"Original midi onsets: {original_midi_file_onsets}")
print(f"Original midi cumsum durations: {original_time_midi_file_pitches}")
print(f"Original midi durations: {original_midi_file_durations}")
print(f"Total original midi pitches duration (ticks): {sum(original_midi_file_durations)}")
print(f"Total original midi pitches duration (seconds): {sum(original_midi_durations_in_seconds)}")
print(f"Total original midi pitches duration (seconds): {original_midi_durations_in_seconds}")

change_complexity = ChangeComplexity()

#new_midi_durations_in_seconds, new_midi_file_onsets, new_midi_onsets_in_seconds 
new_midi_file_pitches, new_midi_file_pitches_with_octaves, new_midi_file_durations = change_complexity.execute(original_midi_file_note_names, original_midi_file_durations, original_midi_file_onsets, note_dict)
#new_midi_file_pitches, new_midi_file_pitches_with_octaves, new_midi_file_durations = change_complexity.execute(original_midi_file_note_names, original_midi_file_durations, original_midi_file_onsets, note_dict)

print(f"New midi pitches: {new_midi_file_pitches}")
print(f"New midi with octave pitches: {new_midi_file_pitches_with_octaves}")
print(f"New midi with octave durations: {new_midi_file_durations}")
print(f"Total new midi with octave pitches duration (ticks): {sum(new_midi_file_durations)}")
#print(f"Total original midi pitches duration (seconds): {sum(midi_durations_in_seconds)}")

# Save the results to a CSV file
output_file_path = "midi_notes.csv"
with open(output_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Pitch", "Duration", "Duration in seconds"])
    for pitch, duration, duration_in_seconds in zip(original_midi_file_note_names, original_midi_file_durations, original_midi_durations_in_seconds):
        writer.writerow([pitch, duration, duration_in_seconds])

#print(f"Results saved to {output_file_path}")
# (ticks, nats)
midi_entropy_calculator = MidiEntropyCalculator()
midi_file_time_bins, midi_file_entropy_values = midi_entropy_calculator.calculate_midi_entropy_with_window(midi_file_path)

# Scatter plot of entropy values vs time
plt.figure(figsize=(10, 6))
plt.scatter(midi_file_time_bins, midi_file_entropy_values, marker='o', color='blue', alpha=0.7, s=1.5)
plt.title('Entropy vs Time')
plt.xlabel('Time (seconds)')
plt.ylabel('Entropy')
plt.grid(True)

# Audio file

# Load query audio with AudioLoader algorithm, it returns:
# · [0] audio (vector_stereosample) - the input audio signal
# · [1] sampleRate (real) - the sampling rate of the audio signal [Hz]
# · [2] numberChannels (integer) - the number of channels
# · [3] md5 (string) - the MD5 checksum of raw undecoded audio payload
# · [4] bit_rate (integer) - the bit rate of the input audio, as reported by the decoder codec
# · [5] codec (string) - the codec that is used to decode the input audio
audio_file_features = estd.AudioLoader(filename = audio_file_path)()

# Load song with MonoLoader algorithm, it returns:
# · [0] audio (vector_real) - the audio signal
audio_file_audio = estd.MonoLoader(filename = audio_file_path, resampleQuality = 0, sampleRate = audio_file_features[1])()

# Compute beat positions and BPM.
audio_beat_detector = AudioBeatDetection()
# (bpm, seconds, [0,1], seconds)
audio_file_bpm, audio_file_beats, audio_file_beats_confidence, _, audio_file_beats_intervals = audio_beat_detector.detect_beat(audio_file_audio)

# Computes beats per minute histogram and its statistics for the highest
# and second highest peak. 
# · firstPeakBPM (real) - value for the highest peak [bpm]
# · firstPeakWeight (real) - weight of the highest peak
# · firstPeakSpread (real) - spread of the highest peak
# · secondPeakBPM (real) - value for the second highest peak [bpm]
# · secondPeakWeight (real) - weight of the second highest peak
# · secondPeakSpread (real) - spread of the second highest peak
# · histogram (vector_real) - bpm histogram [bpm]

peak1_bpm, peak1_weight, peak1_spread, peak2_bpm, peak2_weight, peak2_spread, audio_file_histogram_bpm = \
    estd.BpmHistogramDescriptors()(audio_file_beats_intervals)

"""print("Overall BPM (estimated before): %0.1f" % audio_file_bpm)
print("First histogram peak: %0.1f bpm" % peak1_bpm)
print("Second histogram peak: %0.1f bpm" % peak2_bpm)"""

audio_file_bpm_entropy = entropy(audio_file_histogram_bpm)

print(f"Audio file BPM entropy: {audio_file_bpm_entropy}")

"""resample_quality = 0
num_bins = 12
min_frequency = 50
max_frequency = 5000"""

hpcp = HPCP()
audio_file_hpcp = hpcp.extract_HPCPs(audio_file_audio, audio_file_features)

audio_file_to_midi_file = AudioFileToMidiFile()
audio_file_pitch_values, audio_file_pitch_confidence, audio_file_pitch_times, audio_file_midi_track = audio_file_to_midi_file.execute(audio_file_path, audio_file_features)

"""audio_file_pitch_melody = AudioPitchMelody()
audio_file_pitch_values, audio_file_pitch_confidence, audio_file_pitch_times, audio_file_hop_size, audio = audio_file_pitch_melody.extract_audio_pitch_melody(audio_file_path, audio_file_features)

pitch_MIDI = Note_segmentation_and_converting_to_MIDI()
onsets, durations, notes = pitch_MIDI.extract_pitch_MIDI(audio_file_pitch_values, audio_file_audio, audio)

pitch_to_MIDI = Pitch_to_MIDI()
pitch_to_MIDI.save_pitch_to_MIDI(audio_file_path, onsets, durations, notes)
"""
# Filter pitch_values where pitch_confidence is not 0
filtered_pitch_values = audio_file_pitch_values[audio_file_pitch_confidence > 0]
filtered_pitch_times = audio_file_pitch_times[audio_file_pitch_confidence > 0]

# Variance of Pitch
pitch_variance = np.var(filtered_pitch_values)

# Range of Pitch
pitch_range = np.max(filtered_pitch_values) - np.min(filtered_pitch_values)

# Number of Pitch Changes
pitch_changes = np.sum(np.diff(filtered_pitch_values) != 0)

# Entropy of Pitch
pitch_probabilities, _ = np.histogram(filtered_pitch_values, bins=50, density=True)
pitch_entropy = entropy(pitch_probabilities)

# Rhythm and Duration of Notes
note_durations = np.diff(filtered_pitch_times)

print(f"Pitch Entropy: {pitch_entropy}")

#onsets, durations, notes = estd.PitchContourSegmentation(hopSize=128)(audio_file_pitch_values, audio)


"""def create_transition_matrix(notes, note_range=(0, 127)):
    num_notes = note_range[1] - note_range[0] + 1
    transition_matrix = np.zeros((num_notes, num_notes))
    
    for i in range(len(notes) - 1):
        current_note = notes[i] - note_range[0]
        next_note = notes[i + 1] - note_range[0]
        transition_matrix[current_note, next_note] += 1
    
    # Normalize the matrix to convert counts to probabilities
    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    transition_matrix = np.divide(transition_matrix, row_sums, where=row_sums != 0)
    
    return transition_matrix"""

# Plot
note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Convert the durations to cumulative time
time_midi_file_pitches = np.cumsum(original_midi_file_durations)
time_new_melody = np.cumsum(new_midi_file_durations)


octave_note_names = ['C3', 'C#3', 'D3', 'D#3', 'E3', 'F3', 'F#3', 'G3', 'G#3', 'A3', 'A#3', 'B3',
                     'C4', 'C#4', 'D4', 'D#4', 'E4', 'F4', 'F#4', 'G4', 'G#4', 'A4', 'A#4', 'B4']

octave_note_positions = np.arange(36, 97)  # MIDI note numbers from C3 to B6
time_octave_melodies = np.cumsum(new_midi_file_durations)

# Create the plot
plt.figure(figsize=(12, 6))

# Plot for midi_file_pitches
plt.subplot(2, 1, 1)
plt.plot(time_midi_file_pitches, original_midi_file_note_names, marker='o', markersize=4, linestyle='-', color='blue', label='Original Melody')
plt.plot(time_new_melody, new_midi_file_pitches, marker='o', markersize=4, linestyle='-', color='red', label='New Melody')
plt.yticks(range(12), note_names)
plt.title('Original MIDI Pitches')
plt.xlabel('Time (cumulative duration)')
plt.ylabel('Note')
plt.grid()
plt.legend()

# Plot for octave_melodies
plt.subplot(2, 1, 2)
plt.plot(time_octave_melodies, new_midi_file_pitches_with_octaves, marker='o', markersize=4, linestyle='-', color='black', label='Octave Melodies Pitches')
plt.yticks(range(len(octave_note_names)), octave_note_names)
plt.title('Octave Melodies Pitches')
plt.xlabel('Time (cumulative duration)')
plt.ylabel('Note')
plt.grid()
plt.legend()

# Adjust layout and display plot
plt.tight_layout()

plt.show()

octave_note_names = ['C3', 'C#3', 'D3', 'D#3', 'E3', 'F3', 'F#3', 'G3', 'G#3', 'A3', 'A#3', 'B3',
                     'C4', 'C#4', 'D4', 'D#4', 'E4', 'F4', 'F#4', 'G4', 'G#4', 'A4', 'A#4', 'B4',
                     'C5', 'C#5', 'D5', 'D#5', 'E5', 'F5', 'F#5', 'G5', 'G#5', 'A5', 'A#5', 'B5',
                     'C6', 'C#6', 'D6', 'D#6', 'E6', 'F6', 'F#6', 'G6', 'G#6', 'A6', 'A#6', 'B6']

if prints:
    # Scatter plot of entropy values vs time
    plt.figure(figsize=(10, 6))
    plt.scatter(midi_file_time_bins, midi_file_entropy_values, marker='o', color='blue', alpha=0.7, s=1.5)
    plt.title('Entropy vs Time')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Entropy')
    plt.grid(True)

    # Plot histogram for pitches
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.hist(new_midi_file_pitches, bins=range(21, 109), edgecolor='black', alpha=0.7)
    plt.xlabel('MIDI Note Number')
    plt.ylabel('Frequency')
    plt.title('Histogram of MIDI Note Pitches')

    # Plot histogram for durations
    plt.subplot(2, 1, 2)
    plt.hist(new_midi_file_durations, bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel('Duration (ticks)')
    plt.ylabel('Frequency')
    plt.title('Histogram of Note Durations')

    plt.tight_layout()

    # Plot 2D histogram for pitch vs duration
    plt.figure(figsize=(12, 6))
    plt.hist2d(new_midi_file_pitches, new_midi_file_durations, bins=[range(21, 109), 30], cmap='viridis')
    plt.colorbar(label='Frequency')
    plt.xlabel('MIDI Note Number')
    plt.ylabel('Duration (ticks)')
    plt.title('2D Histogram of Pitch and Duration')

    plt.tight_layout()

    """# Create the transition matrix
    transition_matrix = create_transition_matrix(notes, note_range=(21, 108))
    # Plot the transition matrix
    plt.figure(figsize=(12, 10))
    plt.imshow(transition_matrix, cmap='viridis', aspect='auto', origin='lower')
    plt.colorbar(label='Transition Probability')
    plt.xlabel('Next Note')
    plt.ylabel('Current Note')
    plt.title('Transition Probability Matrix')

    # Set tick labels to MIDI note numbers
    note_labels = np.arange(21, 109)
    plt.xticks(ticks=np.arange(len(note_labels)), labels=note_labels, rotation=90)
    plt.yticks(ticks=np.arange(len(note_labels)), labels=note_labels)"""

    plt.show()

    #
    note_names = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']

    fig, ax = plt.subplots()
    ax.bar(range(len(audio_file_histogram_bpm)), audio_file_histogram_bpm, width=1)
    ax.set_xlabel('BPM')
    ax.set_ylabel('Frequency of occurrence')
    plt.title("BPM histogram")
    ax.set_xticks([20 * x + 0.5 for x in range(int(len(audio_file_histogram_bpm) / 20))])
    ax.set_xticklabels([str(20 * x) for x in range(int(len(audio_file_histogram_bpm) / 20))])

    #plt.rcParams['figure.figsize'] = (15, 6)
    plt.figure(figsize=(10, 6))
    plt.plot(audio)
    for onset, note in zip(onsets, notes):
        plt.axvline(x=onset*audio_file_features[1], color='red')
        plt.text(onset * audio_file_features[1], np.max(audio), str(note), color='red', fontsize=12, ha='center', va='bottom')
    plt.xlabel('Time (samples)')
    plt.title("Audio waveform and the estimated onsets")

    #print("MIDI notes:", notes) # Midi pitch number
    #print("MIDI note onsets:", onsets)
    #print("MIDI note durations:", durations)

    #print("MIDI notes:", notes) # Midi pitch number
    #print("MIDI note onsets:", onsets)
    #print("MIDI note durations:", durations)

    f, axarr = plt.subplots(2, sharex=True)
    f.suptitle(f"'{os.path.basename(audio_file_path)}' (audio 1) Melody", fontsize=15)
    axarr[0].scatter(audio_file_pitch_times, audio_file_pitch_values, s=1.5)
    axarr[0].set_title('estimated pitch [Hz]')
    axarr[1].scatter(audio_file_pitch_times, audio_file_pitch_confidence, s=1.5)
    axarr[1].set_title('pitch confidence')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    f.savefig(audio_file_path.replace(".wav","_melody.png"))

    plt.figure(figsize=(10, 6))
    plt.hist(filtered_pitch_values, bins=50, color='g', alpha=0.7)
    plt.title('Histogram of pitch values with confidence > 0', fontsize=15)
    plt.xlabel('Pitch Value [Hz]')
    plt.ylabel('Frequency')

    # Guardar el histograma
    plt.savefig(audio_file_path.replace(".wav","_histogram.png"))

    """plt.figure(figsize=(8, 4))
    plt.title(f"'{os.path.basename(file_path)}' (audio 1) HPCP", fontsize=15)
    plt.imshow(hpcp1.T, aspect='auto', origin='lower', interpolation='none', extent=[0, hpcp1.shape[0], -0.5, 11.5])
    plt.xlabel('Frames', fontsize=15)
    plt.xticks(fontsize=15)
    plt.ylabel('HPCP Index (musical notes)', fontsize=15)
    plt.yticks(range(12), note_names, fontsize=15)
    plt.savefig(file_path.replace(".wav","_hpcp.png"))"""

    plt.show()

    #complexity = estd.MusicExtractor(file_path)

    #print(complexity)
    