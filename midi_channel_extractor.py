import os
import mido

class MidiChannelExtractor:
    def __init__(self, output_folder):
        self.output_folder = output_folder
        os.makedirs(output_folder, exist_ok=True)

    def get_channels(self, midi_file_path):
        midi_file = mido.MidiFile(midi_file_path)
        channels = set()
        
        for track in midi_file.tracks:
            for msg in track:
                if not msg.is_meta:
                    channels.add(msg.channel)
                    
        return channels

    def save_channel_midi(self, input_file_path, channel):
        # Load the original MIDI file
        midi_file = mido.MidiFile(input_file_path)

        # Create a new MIDI file
        new_midi = mido.MidiFile()

        for track in midi_file.tracks:
            new_track = mido.MidiTrack()
            new_midi.tracks.append(new_track)
            
            for msg in track:
                # Filter messages by the specified channel
                if not msg.is_meta and msg.channel == channel:
                    new_track.append(msg)
                elif msg.is_meta:
                    # Add meta messages to the new track
                    new_track.append(msg)

        # Construct the output file path
        file_name = os.path.basename(input_file_path).replace('.mid', f'_channel_{channel}.mid')
        output_file_path = os.path.join(self.output_folder, file_name)

        # Save the new MIDI file
        new_midi.save(output_file_path)
        print(f"New MIDI file saved to: {output_file_path}")

# Usage example
midi_folder = './data/midi_files'
output_folder = './data/midi_files/generated_midi_files'

midi_channel_extractor = MidiChannelExtractor(output_folder)

for file_name in os.listdir(midi_folder):
    if file_name.endswith('.mid'):
        input_file_path = os.path.join(midi_folder, file_name)
        channels = midi_channel_extractor.get_channels(input_file_path)
        
        if len(channels) > 1:
            for channel in channels:
                midi_channel_extractor.save_channel_midi(input_file_path, channel)
