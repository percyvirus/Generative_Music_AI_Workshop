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
        all_notes = list(range(128))  # Incluir todas las notas MIDI posibles (0-127)

        # Crear DataFrame con todas las notas posibles
        df = pd.DataFrame(index=all_notes, columns=all_notes).fillna(0.0)

        # Asignar probabilidades calculadas a la matriz
        for note, next_notes in probabilities.items():
            for next_note, prob in next_notes.items():
                df.at[note, next_note] = prob

        # Asignar probabilidad de 1 a la transición de una nota a sí misma si no aparece
        for note in all_notes:
            if note not in probabilities:
                df.at[note, note] = 1.0

        # Guardar el DataFrame en un archivo CSV
        df.to_csv(csv_file, float_format='%.8f')
