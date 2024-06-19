import essentia.standard as estd
import numpy as np
from scipy.stats import entropy
import matplotlib.pyplot as plt
import os
from beat_detection import BeatDetection
from audio_pitch_melody import AudioPitchMelody
from note_segmentation_and_converting_to_MIDI import Note_segmentation_and_converting_to_MIDI
from pitch_to_MIDI import Pitch_to_MIDI

from midi_notes_extractor import MidiNotesExtractor
from midi_entropy_calculator import MidiEntropyCalculator
from hpcp import HPCP
from midi_notes_extractor import MidiNotesExtractor
from audio_file_to_midi_file import AudioFileToMidiFile

prints = False

midi_file_path = './data/midi_files/471 Stage 1 - "The Jungle".mid'
audio_file_path = "./data/AS_escala_alba_A.wav"

# MIDI file

midi_notes_extractor = MidiNotesExtractor()
# (Midi note, ticks)
midi_file_pitches, midi_file_durations = midi_notes_extractor.extract_midi_notes(midi_file_path, channel=0)

# (ticks, nats)
midi_entropy_calculator = MidiEntropyCalculator()
midi_file_time_bins, midi_file_entropy_values = midi_entropy_calculator.calculate_midi_entropy_with_window(midi_file_path)

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
beat_detector = BeatDetection()
# (bpm, seconds, [0,1], seconds)
audio_file_bpm, audio_file_beats, audio_file_beats_confidence, _, audio_file_beats_intervals = beat_detector.detect_beat(audio_file_audio)

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

onsets, durations, notes = estd.PitchContourSegmentation(hopSize=128)(audio_file_pitch_values, audio)


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
    plt.hist(midi_file_pitches, bins=range(21, 109), edgecolor='black', alpha=0.7)
    plt.xlabel('MIDI Note Number')
    plt.ylabel('Frequency')
    plt.title('Histogram of MIDI Note Pitches')

    # Plot histogram for durations
    plt.subplot(2, 1, 2)
    plt.hist(midi_file_durations, bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel('Duration (ticks)')
    plt.ylabel('Frequency')
    plt.title('Histogram of Note Durations')

    plt.tight_layout()

    # Plot 2D histogram for pitch vs duration
    plt.figure(figsize=(12, 6))
    plt.hist2d(midi_file_pitches, midi_file_durations, bins=[range(21, 109), 30], cmap='viridis')
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