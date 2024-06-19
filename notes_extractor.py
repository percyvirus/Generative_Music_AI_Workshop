import mido
import numpy as np

class NotesExtractor:
    def __init__(self):
        pass

    def extract_notes(self, midi_file_path, channel=0):
        self.midi_file = mido.MidiFile(midi_file_path, clip=True)
        pitches = []
        durations = []
        
        for track in self.midi_file.tracks:
            note_on_events = {}
            current_time = 0
            
            for msg in track:
                if msg.type in ['note_on', 'note_off'] and msg.channel == channel:
                    current_time += msg.time
                    if msg.type == 'note_on' and msg.velocity > 0:
                        note_on_events[msg.note] = current_time
                    elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                        if msg.note in note_on_events:
                            note_on_time = note_on_events.pop(msg.note)
                            duration = current_time - note_on_time
                            pitches.append(msg.note)
                            durations.append(duration)
        
        return pitches, durations

