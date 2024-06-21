import mido
import pandas as pd
from collections import defaultdict

class ExtractNoteTransitions():
    
    def __init__(self):
        pass
    
    def extract_note_transitions(self, midi_file):
        # Cargar el archivo MIDI
        mid = mido.MidiFile(midi_file)
        
        note_transitions = defaultdict(lambda: defaultdict(int))
        previous_note = None

        for track in mid.tracks:
            for msg in track:
                if msg.type == 'note_on' and msg.velocity > 0:
                    current_note = msg.note
                    if previous_note is not None:
                        note_transitions[previous_note][current_note] += 1
                    previous_note = current_note
                elif msg.type == 'note_off':
                    previous_note = None  # Reseteamos después de una nota off para evitar notas sostenidas

        return note_transitions

    def calculate_transition_probabilities(self, transitions):
        probabilities = {}

        for note, next_notes in transitions.items():
            total = sum(next_notes.values())
            probabilities[note] = {next_note: count / total for next_note, count in next_notes.items()}

        return probabilities

    def save_probabilities_to_csv(self, probabilities, csv_file):
        all_notes = sorted(set(probabilities.keys()).union({n for nexts in probabilities.values() for n in nexts.keys()}))

        df = pd.DataFrame(index=all_notes, columns=all_notes).fillna(0.0)

        for note, next_notes in probabilities.items():
            for next_note, prob in next_notes.items():
                df.at[note, next_note] = prob

        df.to_csv(csv_file)
