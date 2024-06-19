import numpy as np
import random
import MarkovProb as mp
from music21 import converter

def get_melody(path): 
    """Get the input melody and change it into pitch notation"""
    midi_data = converter.parse(path)
    melody_pitch = [note.pitch.name for note in midi_data.flat.notes]
    return melody_pitch

def choose_random_index(n, a, b):
    """Choose some indexes"""
    # return [random.randint(a, b) for _ in range(n)]
    return random.sample(range(a, b), n)

def add_index(scale, input_melody): 
    """Add random index into the melody line according to the input scale"""
    len_input = len(input_melody)
    print(len_input)
    n_note_added = round(0.1*scale*len_input)
    new_melody_length = n_note_added+len(input_melody)
    print(new_melody_length)

    # Generate and sort the index list
    added_index = choose_random_index(n_note_added, 1, new_melody_length-1)
    added_index = sorted(added_index)
    return added_index, new_melody_length

def insert_notes(added_index, new_length): 
    """Create new melody line to store the position of added index"""
    new_melody_line_no_pitch = [0 for _ in range(new_length)]
    # Using 1 to indicate positions needed to add notes
    for index in added_index: 
        new_melody_line_no_pitch[index] = 1
    return new_melody_line_no_pitch

def create_new_melody_line(new_melody, melody): 
    """Create new melody line to store the position of added index"""
    counter = 0
    for note in melody: 
        while new_melody[counter] != 0: 
            counter += 1
        new_melody[counter] = note
    return new_melody

def matrix_to_dic(transition_matrix): 
    pitch_class_prob_dic = {}
    keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    for row in range(len(transition_matrix)): 
        pitch_class = keys[row]
        pitch_class_prob_dic[pitch_class] = transition_matrix[row]
    return keys, pitch_class_prob_dic

def generate_new_state(melody, pitch_class_prob_dic, keys): 
    for idx, note in enumerate(melody): 
        if note == 1: 
            prev_state = melody[idx-1]
            probabilities = pitch_class_prob_dic[prev_state]
            melody[idx] = random.choices(keys, probabilities)[0]
    return melody

# def change_duration(scale, input_melody, ): 
    
#     return duration_changed_melody

path = './data/Complex_4.mid'
melody = get_melody(path)
# print(melody)
added_index, new_melody_length = add_index(5, melody) # 1 - scale
print(added_index)
new_melody_line_no_pitch = insert_notes(added_index, new_melody_length)
new_melody_line = create_new_melody_line(new_melody_line_no_pitch, melody)
#print(new_melody_line)
transition_matrix = mp.compute_transition_matrix('./data/midi_score.mid')
# print(transition_matrix)
pitch_class, pitch_class_prob_dic = matrix_to_dic(transition_matrix)
# print(pitch_class_prob_dic)
new_melody = generate_new_state(new_melody_line, pitch_class_prob_dic, pitch_class)
print(new_melody)