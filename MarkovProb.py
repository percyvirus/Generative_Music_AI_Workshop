from music21 import converter, note, chord
from hmmlearn import hmm 
import numpy as np
import os
from collections import defaultdict
import pretty_midi

def read_file(path): 
    midi_directory = path
    midi_files = [os.path.join(midi_directory,f) for f in os.listdir(midi_directory)]
    return midi_files

def get_pitch(midi_file): 
    """Parse a MIDI file and convert notes to pitch notations"""
    midi_data = converter.parse(midi_file)
    pitches = [note.pitch for note in midi_data.flat.notes]
    return pitches

def count_pitch(pitches): 
    """Count the number of each pitch notation"""
    pitch_dic = defaultdict(int)
    for pitch in pitches: 
        pitch_dic[pitch]+=1
    print(pitch_dic)
    return pitch_dic

def pitch_to_number(pitches):
    """Convert a pitch (e.g., 'C4', 'D#5') to a numerical value."""
    pitch_n = [pitch.ps for pitch in pitches]
    return pitch_n

def aggregate_pitches(midi_files):
    """Aggregate pitch data from multiple MIDI files."""
    all_pitches_n = []
    for midi_file in midi_files:
        pitches = get_pitch(midi_file)
        all_pitches_n.extend(pitches)
    return all_pitches_n

def compute_state_probs(all_pitches): 
    """Compute the state probabilities"""
    pitch_dic = count_pitch(all_pitches)
    pitch_counts = list(pitch_dic.values())
    total_pitch = sum(np.array(pitch_counts))
    state_probs = defaultdict(float)
    for pitch in pitch_dic.keys():
        state_probs[pitch] = pitch_dic[pitch]/total_pitch
    return pitch_counts, state_probs

def train_hmm(pitches, pitch_counts):
    """Train an HMM on the given pitch data."""
    # Define the HMM
    n_components = len(pitch_counts)
    model = hmm.GaussianHMM(n_components, covariance_type='diag', n_iter=100)
    # Train the HMM
    pitches = pitch_to_number(pitches)
    pitches = np.array(pitches).reshape(-1, 1)
    model.fit(pitches)
    return model

def compute_transition_probs(model):
    """Get transition probabilities from the HMM model"""
    return model.transmat_

def compute_transition_matrix(midi_file): 
    midi_data = pretty_midi.PrettyMIDI(midi_file)
    transition_matrix = midi_data.instruments[0].get_pitch_class_transition_matrix()
    avgs = transition_matrix.sum(axis=1)
    for i in range(len(avgs)):
        transition_matrix[i] = transition_matrix[i]/avgs[i]
    return transition_matrix

# ----- Test -----
# midi_files = read_file('/Users/wendy/Documents/GenAI/Nexity/dataset')
# all_pitches = aggregate_pitches(midi_files)
# pitch_counts, state_probs = compute_state_probs(all_pitches)
# model = train_hmm(all_pitches, pitch_counts)
# transition_probs = compute_transition_probs(model)
# print(transition_probs)