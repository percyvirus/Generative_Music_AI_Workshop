import mido
import numpy as np

class MidiNotesExtractor:
    def __init__(self):
        pass
    
    def extract_midi_notes(self, midi_file_path, channel=0):
        self.midi_file = mido.MidiFile(midi_file_path, clip=True)
        pitches = []
        durations = []
        onsets = []
        ticks_per_beat = self.midi_file.ticks_per_beat
        tempo = None
        
        for track in self.midi_file.tracks:
            note_on_events = {}
            current_time = 0
            
            for msg in track:
                if msg.type == 'set_tempo':
                    tempo = msg.tempo
                
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
                            onsets.append(note_on_time)
        
        if tempo is None:
            tempo = mido.bpm2tempo(120)  # default to 120 BPM if no tempo message is found

        # Convert durations and onsets from ticks to seconds
        durations_in_seconds = [mido.tick2second(d, ticks_per_beat, tempo) for d in durations]
        onsets_in_seconds = [mido.tick2second(o, ticks_per_beat, tempo) for o in onsets]

        # Convert MIDI note numbers to note names
        note_names = [self.midi_note_to_name(pitch) for pitch in pitches]
        
        return onsets, durations, pitches, onsets_in_seconds, durations_in_seconds, note_names

    @staticmethod
    def midi_note_to_name(midi_note):
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        octave = (midi_note // 12) - 1
        note_name = note_names[midi_note % 12]
        return f"{note_name}"

