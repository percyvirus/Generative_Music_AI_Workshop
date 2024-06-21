import numpy as np
import random
import MarkovProb as mp
from music21 import converter, pitch
import csv

class ChangeComplexity():
    
    def __init__(self):
        pass
    
    def execute(self, midi_file_pitches, midi_file_durations, midi_file_onsets, note_dict):
        transition_matrix = mp.compute_transition_matrix('./data/midi_files/midi_score.mid')
        pitch_class, pitch_class_prob_dic = self.matrix_to_dic(transition_matrix)
        
        added_index, new_melody_length = self.add_index(5, midi_file_pitches)
        
        new_melody_line_no_pitches = self.insert_notes(added_index, new_melody_length)
        new_melody_line_no_new_pitches = self.create_new_melody_line(new_melody_line_no_pitches, midi_file_pitches)
        new_melody_pitches = self.generate_new_state(new_melody_line_no_new_pitches, pitch_class_prob_dic, pitch_class)
        
        
        new_melody_pitches_midi = self.pitch_to_midi_note(new_melody_pitches)
        
        new_melody_line_no_duration = self.insert_notes(added_index, new_melody_length)
        new_melody_line_no_new_durations = self.create_new_melody_line(new_melody_line_no_duration, midi_file_durations)
        new_melody_durations = self.generate_durations(added_index, new_melody_line_no_new_durations)
        
        print(f"Total length original: {sum(midi_file_durations)}")
        print(f"Total length new: {sum(new_melody_durations)}")
        print(f"Difference: {sum(new_melody_durations)-sum(midi_file_durations)}")
        
        print(f"Total silence: {sum(new_melody_durations)-midi_file_onsets[-1]-midi_file_durations[-1]}")
        
        onsets = [0]
        for duration in new_melody_durations[:-1]:  # Itera hasta el penúltimo elemento
            onsets.append(onsets[-1] + duration)

        """
        new_melody_line_no_onsets = self.insert_notes(added_index, new_melody_length)
        new_melody_line_no_new_onsets = self.create_new_melody_line_onset(new_melody_line_no_onsets, midi_file_onsets)
        print(new_melody_line_no_new_onsets)
        new_melody_onsets = self.generate_onsets(added_index, new_melody_line_no_new_onsets)
        """
        octave_melodies = self.generate_octave(note_dict, new_melody_pitches, added_index)
        
        """print(f"Onsets: {onsets}")
        print(f"Durations: {new_melody_durations}")
        print(f"Pitches: {octave_melodies}")"""
        
        ticks_per_beat = 480
        tempo_bpm = 120 
        
        onsets = self.convert_ticks_to_seconds(onsets, ticks_per_beat, tempo_bpm)
        new_melody_durations = self.convert_ticks_to_seconds(new_melody_durations, ticks_per_beat, tempo_bpm)

        
        #return new_melody_pitches, octave_melodies, new_melody_durations
        return onsets, new_melody_durations, octave_melodies, new_melody_pitches_midi
    
    def execute_bis(self, midi_file_pitches, midi_file_durations, midi_file_onsets, note_dict):
        added_index, new_melody_length = self.add_index(5, midi_file_pitches)
        new_melody_line_no_pitch = self.insert_notes(added_index, new_melody_length)
        new_melody_line = self.create_new_melody_line(new_melody_line_no_pitch, midi_file_pitches)
        transition_matrix = mp.compute_transition_matrix('./data/midi_files/midi_score.mid')
        pitch_class, pitch_class_prob_dic = self.matrix_to_dic(transition_matrix)

        new_melody = self.generate_new_state(new_melody_line, pitch_class_prob_dic, pitch_class)
        durations_new_melody_line_no_duration = self.insert_notes(added_index, new_melody_length)
        durations_new_melody_line = self.create_new_melody_line(durations_new_melody_line_no_duration, midi_file_durations)
        durations_new_melody = self.generate_durations_bis(added_index, durations_new_melody_line, midi_file_onsets)
        octave_melodies = self.generate_octave(note_dict, new_melody)
        new_midi = self.pitch_to_midi_note(new_melody_with_octave   )
        
        return new_melody, octave_melodies, durations_new_melody
    
    def change_note(element): # ADDED
        change_dic = {'C-':'B', 'D-':'C#', 'E-':'D#', 'F-':'E', 'G-':'F#', 'A-':'G#', 'B-':'A#'}
        if element in change_dic: 
            element = change_dic[element]
        return element
    
    def get_melody(self, path): 
        """Get the input melody and change it into pitch notation
        midi_data = converter.parse(path)
        melody_pitch = [note.pitch.name for note in midi_data.flat.notes]
        return melody_pitch"""
        midi_data = converter.parse(path)
        melody_pitch_with_octave = []
        melody_pitch_no_octave = []
        for element in midi_data.flat.notes: 
            if element.isNote: 
                pitch_with_octave = change_note(element.pitch.nameWithOctave)
                pitch_no_octave = change_note(element.pitch.name)
                melody_pitch_with_octave.append(pitch_with_octave)
                melody_pitch_no_octave.append(pitch_no_octave)
            elif element.isChord: 
                chord_pitches_with_octave = [change_note(pitch.nameWithOctave) for pitch in element.pitches]
                chord_pitches_no_octave = [change_note(pitch.name) for pitch in element.pitches]
                melody_pitch_with_octave.append(chord_pitches_with_octave)
                melody_pitch_no_octave.append(chord_pitches_no_octave)
        return melody_pitch_with_octave, melody_pitch_no_octave


    def choose_random_index(self, n, a, b):
        """Choose some indexes"""
        # return [random.randint(a, b) for _ in range(n)]
        return random.sample(range(a, b), n)

    def add_index(self, scale, input_melody): 
        """Add random index into the melody line according to the input scale"""
        len_input = len(input_melody)
        # print(len_input)
        n_note_added = round(0.1*scale*len_input)
        new_melody_length = n_note_added+len(input_melody)
        # print(new_melody_length)

        # Generate and sort the index list
        added_index = self.choose_random_index(n_note_added, 1, new_melody_length-1)
        added_index = sorted(added_index)
        return added_index, new_melody_length

    def insert_notes(self, added_index, new_length): 
        """Create new melody line to store the position of added index"""
        new_melody_line_no_pitch = [0 for _ in range(new_length)]
        # Using 1 to indicate positions needed to add notes
        for index in added_index: 
            new_melody_line_no_pitch[index] = 1
        return new_melody_line_no_pitch

    def create_new_melody_line(self, new_melody_line_no_pitch, midi_file_pitches):
        new_melody = new_melody_line_no_pitch.copy()
        """Create new melody line to store the position of added index"""
        counter = 0
        for note in midi_file_pitches:
            while new_melody[counter] != 0: 
                counter += 1
            new_melody[counter] = note
        return new_melody
    
    def create_new_melody_line_onset(self, new_melody_line_no_onset, midi_file_onsets):
        new_melody = new_melody_line_no_onset.copy()
        """Create new melody line to store the position of added index"""
        counter = 1
        for onset in midi_file_onsets[1:]:
            while new_melody[counter] != 0: 
                counter += 1
            new_melody[counter] = onset
        return new_melody

    def matrix_to_dic(self, transition_matrix): 
        pitch_class_prob_dic = {}
        keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        for row in range(len(transition_matrix)): 
            pitch_class = keys[row]
            pitch_class_prob_dic[pitch_class] = transition_matrix[row]
        return keys, pitch_class_prob_dic

    """def generate_new_state(self, melody_line, pitch_class_prob_dic, pitches_class):
        melody = melody_line.copy()
        for idx, note in enumerate(melody): 
            if note == 1: 
                prev_state = melody[idx-1]
                probabilities = pitch_class_prob_dic[prev_state]
                melody[idx] = random.choices(pitches_class, probabilities)[0]
        return melody"""
    
    def generate_new_state(self, melody, pitch_class_prob_dic, pitch_classes): # CHANGED
        for idx, note in enumerate(melody): 
            if note == 1: 
                prev_state = melody[idx-1]
                # print(prev_state)
                # Check if the previous state is a note or a chord
                if isinstance(prev_state,str): 
                    if len(prev_state)==1: 
                        probabilities = pitch_class_prob_dic[prev_state]
                    elif prev_state[1]=='#' and len(prev_state)==2: 
                        probabilities = pitch_class_prob_dic[prev_state]
                    elif prev_state[1]!='#' and len(prev_state)==2: 
                        probabilities = pitch_class_prob_dic[prev_state[:-1]]
                    elif len(prev_state)>2: 
                        probabilities = pitch_class_prob_dic[prev_state[:-1]]
                else: 
                    # Choose the root note of the chord and get corresponding transition probability
                    previous_state = prev_state[0]
                    if len(previous_state)==1: 
                        probabilities = pitch_class_prob_dic[previous_state]
                    elif previous_state[1]=='#' and len(previous_state)==2: 
                        probabilities = pitch_class_prob_dic[previous_state]
                    elif previous_state[1]!='#' and len(previous_state)==2: 
                        probabilities = pitch_class_prob_dic[previous_state[:-1]]
                    elif len(previous_state)>2: 
                        probabilities = pitch_class_prob_dic[previous_state[:-1]]
                new_pitch = random.choices(pitch_classes, probabilities)[0]

                # Randomly generate a note or a chord
                if random.choice([True, False]): 
                    melody[idx] = new_pitch
                else: 
                    num_notes = random.randint(2, 5)
                    pitches = [new_pitch]
                    while len(pitches)<num_notes: 
                        pitches.append(random.choice(pitch_classes))
                    melody[idx] = pitches
        return melody

    def generate_onsets(self, idx_list, melody_line_no_new_onsets):
        
        onsets_new_melody_line = melody_line_no_new_onsets.copy()
        print(onsets_new_melody_line) 

        for idx in idx_list:
            prev_index = idx-1
            prev_onset = onsets_new_melody_line[prev_index]
            
            next_index = idx+1
            while onsets_new_melody_line[next_index] == 1: 
                next_index += 1
            next_onset = onsets_new_melody_line[next_index]
            
            gap_onset = next_onset - prev_onset
            gap_index = next_index - prev_index
            
            current_onset = int(prev_onset + random.randint(0, gap_onset)/(gap_index-1))
            
            onsets_new_melody_line[idx] = current_onset
        
        return onsets_new_melody_line
    
    def pitch_to_midi_note(self, melody): # ADDED
        midi_element = []
        for element in melody: 
            if isinstance(element, list): 
                midi_chord_note = []
                for chord_note in element: 
                    midi_chord_note.append(pitch.Pitch(chord_note).midi)
                midi_element.append(midi_chord_note)
            else: 
                midi_element.append(pitch.Pitch(element).midi)
        return midi_element
    
    
    def convert_ticks_to_seconds(self, ticks_list, ticks_per_beat, tempo_bpm):
        # Calcula el número de segundos por beat
        seconds_per_beat = 60 / tempo_bpm
        # Calcula el número de segundos por tick
        seconds_per_tick = seconds_per_beat / ticks_per_beat
        # Convierte la lista de ticks a segundos
        return [tick * seconds_per_tick for tick in ticks_list]

    """

        pitch_class = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        for idx, note in enumerate(melody): 
            if note == 1: 
                prev_state = melody[idx-1]

                # Check if the previous state is a note or a chord
                if len(prev_state)>1: 

                    # Choose the root note of the chord and get corresponding transition probability
                    probabilities = pitch_class_prob_dic[prev_state[0]]
                else: 
                    probabilities = pitch_class_prob_dic[prev_state]
                new_pitch = random.choices(keys, probabilities)[0]

                # Randomly generate a note or a chord
                if random.choice([True, False]): 
                    melody[idx] = new_pitch
                else: 
                    num_notes = random.randint(2, 5)
                    pitches = [new_pitch]
                    while len(pitches)<num_notes: 
                        pitches.append(random.choice(pitch_class))
                    melody[idx] = pitches
        return melody"""

    def get_duration(self, duration_file): 

        columns = {}
        # Open the CSV file
        with open(duration_file, mode='r', newline='') as file:
            # Create a CSV DictReader object
            csv_reader = csv.DictReader(file)
        
            # Initialize the dictionary with column names as keys
            for column in csv_reader.fieldnames:
                columns[column] = []
        
            # Iterate over each row in the CSV file and populate the dictionary
            for row in csv_reader:
                for column in csv_reader.fieldnames:
                    columns[column].append(row[column])
            durations_in_tick = columns['Duration']
            durations_in_tick = [int(duration) for duration in durations_in_tick]
        return durations_in_tick

    def generate_durations(self, idx_list, melody_line_no_new_durations):
        
        durations_new_melody_line = melody_line_no_new_durations.copy() 

        for idx in idx_list:
            prev_duration = durations_new_melody_line[idx-1]
            next_index = idx+1
            while durations_new_melody_line[next_index] == 1: 
                next_index += 1
            next_duration = durations_new_melody_line[next_index]
            if prev_duration == next_duration: 
                current_duration = prev_duration
            else: 
                current_duration = random.randint(min(prev_duration,next_duration), max(prev_duration,next_duration))
            total_duration = prev_duration+next_duration
            total_duration_added = prev_duration+current_duration+next_duration
            durations_new_melody_line[idx-1] = round(prev_duration/total_duration_added*total_duration)
            durations_new_melody_line[idx] = round(current_duration/total_duration_added*total_duration)
            durations_new_melody_line[idx+1] = round(next_duration/total_duration_added*total_duration)
        return durations_new_melody_line
    
    def generate_durations_bis(self, idx_list, durations_new_melody_line, onsets_new_melody_line): 

        for idx in idx_list:
            prev_duration = durations_new_melody_line[idx-1]
            prev_onset = onsets_new_melody_line[idx-1]
            next_index = idx+1
            while durations_new_melody_line[next_index] == 1: 
                next_index += 1
            next_duration = durations_new_melody_line[next_index]
            next_onset = onsets_new_melody_line[next_index]
            if prev_duration == next_duration: 
                current_duration = prev_duration
            else: 
                current_duration = random.randint(min(prev_duration,next_duration), max(prev_duration,next_duration))
            total_duration = prev_duration+next_duration
            total_duration_added = prev_duration+current_duration+next_duration
            durations_new_melody_line[idx-1] = round(prev_duration/total_duration_added*total_duration)
            durations_new_melody_line[idx] = round(current_duration/total_duration_added*total_duration)
            durations_new_melody_line[idx+1] = round(next_duration/total_duration_added*total_duration)
        return durations_new_melody_line

    """def generate_octave(self, octave_prob_dic, melody, added_index): 
        octaves = ['3','4','5','6']
        for idx in added_index: 
            note = melody[idx]
            if isinstance(note, str):
                probabilities = octave_prob_dic[note]
                melody[idx] = note + random.choices(octaves, probabilities)[0]
            else:
                probabilities = octave_prob_dic[note[0]]
                root_note = note[0]+random.choices(octaves, probabilities)[0]
                # for note_i in note: 
                #     probabilities = octave_prob_dic[note_i]
                #     chord.append(note_i+random.choices(octaves, probabilities)[0])
                chord = self.generate_harmonic_chord(root_note)
                melody[idx] = chord
        return melody"""
    def generate_octave(self, octave_prob_dic, melody, added_index): # CHANGED
        octaves = ['3','4','5','6']
        for idx in added_index: 
            note = melody[idx]
            if isinstance(note, str):
                probabilities = octave_prob_dic[note]
                melody[idx] = note + random.choices(octaves, probabilities)[0]
            else: 
                chord = []
                for note_i in note: 
                    probabilities = octave_prob_dic[note_i]
                    chord.append(note_i+random.choices(octaves, probabilities)[0])
                melody[idx] = chord
        return melody


"""melody_path = '/Users/wendy/Documents/GenAI/Nexity/dataset/Complex_4.mid'
melody = get_melody(melody_path)
# print(melody)
added_index, new_melody_length = add_index(5, melody) # 1 - scale
# print(added_index)
new_melody_line_no_pitch = insert_notes(added_index, new_melody_length)
new_melody_line = create_new_melody_line(new_melody_line_no_pitch, melody)
# print(new_melody_line)
transition_matrix = mp.compute_transition_matrix('midi_score.mid')
# print(transition_matrix)
pitch_class, pitch_class_prob_dic = matrix_to_dic(transition_matrix)
# print(pitch_class_prob_dic)
new_melody = generate_new_state(new_melody_line, pitch_class_prob_dic, pitch_class)
# print(new_melody)

duration_path = '/Users/wendy/Documents/GenAI/Nexity/midi_notes.csv'
durations_in_tick = get_duration(duration_path)
durations_new_melody_line_no_duration = insert_notes(added_index, new_melody_length)
durations_new_melody_line = create_new_melody_line(durations_new_melody_line_no_duration, durations_in_tick)
durations_new_melody = generate_durations(added_index,durations_new_melody_line)
# print(durations_new_melody)"""