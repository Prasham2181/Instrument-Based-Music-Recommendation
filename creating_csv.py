import os
import yaml
import csv


def extract_slakh_metadata(root_dir, output_csv):
    """
    Extracts metadata from all YAML files in the Slakh2100 dataset directory and writes to a CSV file.

    Args:
    root_dir (str): Path to the root directory of the Slakh2100 dataset.
    output_csv (str): Path to the output CSV file.
    """
    try:
        # Open the CSV file for writing
        with open(output_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)

            # Write header row
            writer.writerow([
                'Folder Name', 'UUID', 'Track Name', 'Instrument Class',
                'MIDI Program Name', 'Integrated Loudness', 'Is Drum',
                'Plugin Name', 'Program Number'
            ])

            # Traverse the dataset directory
            for folder in os.listdir(root_dir):
                folder_path = os.path.join(root_dir, folder)
                if os.path.isdir(folder_path):
                    # Check for the YAML file in the folder
                    yaml_file_path = os.path.join(folder_path, 'metadata.yaml')
                    if os.path.exists(yaml_file_path):
                        # Process the YAML file
                        with open(yaml_file_path, 'r') as file:
                            try:
                                data = yaml.safe_load(file)

                                # Extract folder name and UUID
                                uuid = data.get('UUID', 'Unknown')
                                stems = data.get('stems', {})

                                # Extract information for each stem (track)
                                for track_name, track_info in stems.items():
                                    writer.writerow([
                                        folder,  # Folder name
                                        uuid,  # UUID
                                        track_name + ".wav",  # Track name (e.g., S00, S01)
                                        track_info.get('inst_class', 'Unknown'),  # Instrument class
                                        track_info.get('midi_program_name', 'Unknown'),  # MIDI program name
                                        track_info.get('integrated_loudness', 'Unknown'),  # Integrated loudness
                                        track_info.get('is_drum', 'Unknown'),  # Is drum
                                        track_info.get('plugin_name', 'Unknown'),  # Plugin name
                                        track_info.get('program_num', 'Unknown')  # Program number
                                    ])
                            except yaml.YAMLError as e:
                                print(f"Error reading YAML file: {yaml_file_path}, Error: {e}")
        print(f"Data successfully written to {output_csv}")
    except Exception as e:
        print(f"Error: {e}")


# Example usage
if __name__ == "__main__":
    # Root directory of Slakh2100 dataset
    slakh_root_dir = 'RawData'

    # Output CSV file path
    output_csv_path = 'slakh2100_metadata.csv'

    # Extract metadata and write to CSV
    extract_slakh_metadata(slakh_root_dir, output_csv_path)
