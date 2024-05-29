import argparse
import json
import glob
import numpy as np
import os
import pandas as pd
import re
import shutil
from sklearn.model_selection import train_test_split
from typing import List, Union

from pydub import AudioSegment
from tools.filelist_to_manifest import main as filelist_to_manifest
from tools.json_handler import JSONHandler
from tools.progress_bar_handler import ProgressBarHandler
from tools.time_handler import TimeHandler

class Preprocessor:
    """
    A class to preprocess audio sessions given audio and diarization JSON files
    to create a dataset for speaker identification.
    """
    
    
    @staticmethod
    def summarize(main_dir: str) -> pd.DataFrame:
        """
        Summarizes the dataset by creating a CSV file containing the
        speakers' names and the time spoken in seconds. It also returns the same
        data in the form of a pandas DataFrame.

        Args:
            main_dir: Directory containing the dataset.
        
        Returns:
            pd.DataFrame: DataFrame containing names of speakers and time spoken.
        """
        
        # Use glob to get a list of directories
        directories = glob.glob(os.path.join(main_dir, '*'))

        # Filter out non-directories and hidden directories to get the list of speakers
        directories = [d for d in directories
                       if os.path.isdir(d) and not os.path.basename(d).startswith('.')]
        directories.sort()
        
        print(f"\n{len(directories)} speakers were found.")
        progress_bar = ProgressBarHandler(len(directories), f"Characterizing {len(directories)} speakers...")
        progress_bar.open()
        
        speakers_df = pd.DataFrame(columns=['Speaker', 'Time spoken'])
        speakers_df['Speaker'] = [os.path.basename(s) for s in directories]
        
        # Get time spoken by speaker
        time_spoken = []
        for dir in directories:
            durations = []
            for file in glob.glob(os.path.join(dir, '*wav')):
                try:
                    audio = AudioSegment.from_file(os.path.join(dir, file))
                    d = len(audio)
                    durations += [d]
                except:
                    pass
            durations = np.array(durations)
            time_spoken += [int(sum(durations) / 1000)]  # Time spoken in seconds
            progress_bar.update(1)

        speakers_df['Time spoken'] = time_spoken
        speakers_df.to_csv(os.path.join(main_dir, 'set_summary.csv'), index=False)
        
        progress_bar.close()
        
        # Find the index of the row with the maximum 'time_spoken'
        idxmax = speakers_df['Time spoken'].idxmax()
        idxmin = speakers_df['Time spoken'].idxmin()
        
        print(f"\nMaximum speak time: {speakers_df.loc[idxmax, 'Speaker']} - {speakers_df.loc[idxmax, 'Time spoken']} sec.")
        print(f"Minimum speak time: {speakers_df.loc[idxmin, 'Speaker']} - {speakers_df.loc[idxmin, 'Time spoken']} sec.")
        print(f"\nFor more information check 'set_summary.csv' in '{main_dir}'.\n")
        return speakers_df
    
    
    @staticmethod
    def segment(chop_dir: str, duration: int) -> str:
        """
        Segments the audios in a directory into subsegments. The maximum duration of all segments
        is given by `duration`.

        Args:
            chop_dir: Directory containing the audios to segment.

        Returns:
            str: Path to the new directory containing all segments.
        """
        # Create main directory f"dataset_{duration}s"
        dataset_name = os.path.basename(chop_dir)
        segments_dir = os.path.join(os.path.dirname(chop_dir), f"{dataset_name}_{duration}")
        try:
            os.makedirs(segments_dir, exist_ok=False)
        except FileExistsError:
            print(f"Directory '{segments_dir}' already exists.")
        
        listdir = os.listdir(chop_dir)
        speakers = [entry for entry in listdir if os.path.isdir(os.path.join(chop_dir, entry))]
        
        duration *= 1000  # Convert to milliseconds
        
        for speaker in speakers:
            speaker_dir = os.path.join(chop_dir, speaker)
            speaker_files = glob.glob(f"{speaker_dir}/*.wav")
            new_speaker_dir = os.path.join(segments_dir, speaker)
            os.makedirs(new_speaker_dir)
            
            for file in speaker_files:
                audio = AudioSegment.from_file(os.path.join(speaker_dir, file))
                D = len(audio)
                if D > duration:  # If the audio is longer than duration it must be segmented
                    slices = D // duration
                    file_basename = os.path.splitext(os.path.basename(file))[0]
                    for i in range(slices):
                        segment = audio[i*duration : (i+1)*duration]
                        path = f"{os.path.join(new_speaker_dir, file_basename)}_{i}.wav"
                        segment.export(path, format='wav')
                    # if D % duration > 1000:  # If the remaining segment is longer than 1 s
                    #     segment = audio[slices*duration:]
                    #     path = f"{os.path.join(new_speaker_dir, file_basename)}_{slices}.wav"
                    #     segment.export(path, format='wav')
                    
                # else:  # If the audio is shorter than duration it is copied as is
                #     shutil.copy(file, new_speaker_dir)
                    
        print(f"Segments of {int(duration/1000)} sec created at {segments_dir}")
        
        # Preprocessor.summarize(segments_dir)``
        return segments_dir
    
    @staticmethod
    def create_manifest(chopped_dir: str, output_dir: str) -> str:
        """
        Creates a manifest from a directory containing chopped audios.
        
        Args:
            chop_dir: Directory containing the chopped audios.
            output_dir: Directory where the manifest file will be saved.
        """
        
        # Get the list of all audio files and save it to a .txt file
        filelist = glob.glob(os.path.join(chopped_dir, '**/*.wav'), recursive=True) 
        filelist_path = os.path.join(chopped_dir, f'{os.path.basename(chopped_dir)}.txt')
        with open(filelist_path, 'w') as file:
            for item in filelist:
                file.write(item + '\n')
            
        # Create a manifest from the .txt file
        all_manifest = os.path.join(output_dir, f'{os.path.basename(chopped_dir)}.json')
        
     
        filelist_to_manifest(filelist=filelist_path,
                             manifest=None,
                             id=-2,
                             out=all_manifest,
                             segments_dir=chopped_dir,
                             min_count=0,
        )
        
        # Sort the manifest by directory name and by the numeric part of the filename
        def extract_numeric_part(filepath):
            """Extract the numeric part from the filename in the audio_filepath."""
            match = re.search(r'_(\d+)\.wav$', filepath)
            if match:
                return int(match.group(1))
            return None

        def sort_key(item):
            """Create a sorting key: first by directory name, then by the numeric part of the filename."""
            parts = item['audio_filepath'].rsplit('/', 2)
            directory = parts[1]
            numeric_part = extract_numeric_part(parts[2])
            return (directory, numeric_part)
        
        with open(all_manifest, 'r') as f:
            lines = f.readlines()
            data = [json.loads(line) for line in lines]

        # Sort the list using the custom sort key
        sorted_data = sorted(data, key=sort_key)

        # Write the sorted list back to a file
        with open(all_manifest, 'w') as f:
            for item in sorted_data:
                f.write(json.dumps(item) + '\n')

        return all_manifest
    
    @staticmethod
    def df_to_manifest(df: pd.DataFrame, output_path: str) -> None:
        """
        Creates and saves a manifest file from a DataFrame for speaker identification.

        Parameters:
        - path: Path where the manifest file will be saved.
        - df  : DataFrame from which data is loaded.

        Returns:
        - None
        """
        
        if not output_path.endswith('.json'):
            raise ValueError("`path` should end with the extension '.json'")

        df.to_json(output_path, orient='records', lines=True)
        # Read the contents of the file and replace escaped "/" characters
        with open(output_path, 'r') as file:
            data = file.read()

        # Replace escaped "/" with "/"
        data = data.replace(r'\/', '/')

        # Write the modified content back to the file
        with open(output_path, 'w') as file:
            file.write(data)
            
    
    @staticmethod
    def split(manifest: str, output_dir: str, 
              train_size: float, val_size: float, test_size: float = 0.0) -> List[str]:
        """
        Splits a DataFrame into train, validation and (optionally) test sets, and saves them
        as manifest files.

        Args:
        - manifest (str): Path of the manifest to split.
        - output_dir (str): Directory where the manifest files will be saved.
        - train_size (float): Proportion of the dataset that will be used for training.
        - val_size (float): Proportion of the dataset that will be used for validation.
        - test_size (float): Proportion of the dataset that will be used for testing.

        Returns:
        - str: Path to the training manifest.
        - str: Path to the validation manifest.
        - str: Path to the training manifest.
        """
        
        if not os.path.isdir(output_dir):
            raise ValueError(f"The value provided as `output_dir` {output_dir} is not a valid directory.")
        
        if not 0 <= train_size <= 1 or not 0 <= val_size <= 1:
            raise ValueError(f"Both train and validation sizes must be between 0 and 1.")
        
        if train_size + val_size + test_size != 1:
            raise ValueError(f"The sum of all set sizes must be 1.")

        df = pd.read_json(manifest, lines=True)

        dataset_name = os.path.basename(manifest).split('.')[0]
        
        # Features (X) and labels (y)
        X = df.drop(columns=['label'])  # Features
        y = df['label']  # Labels
        
        if test_size:
            if not 0 <= test_size <= 1:
                raise ValueError(f"Test size must be between 0 and 1.")
            
            # Split the data into trainval and testing sets
            X_trainval, X_test, y_trainval, y_test = train_test_split(X, y,
                                                                      stratify=y,
                                                                      test_size=test_size,
                                                                      random_state=620)
            # Split the data into training and validation sets
            X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval,
                                                              stratify=y_trainval,
                                                              test_size=val_size/(1-test_size),
                                                              random_state=42)
        else:
            # Split the data into training and validation sets
            X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                            stratify=y,
                                                            test_size=val_size,
                                                            random_state=42)


        # Create new DataFrames for the sets
        train_df = pd.concat([X_train, y_train], axis=1)
        val_df = pd.concat([X_val, y_val], axis=1)

        # Save manifests
        train_manifest = os.path.join(output_dir, f'{dataset_name}_enrollment.json')
        val_manifest = os.path.join(output_dir, f'{dataset_name}_verification.json')

        Preprocessor.df_to_manifest(train_df, train_manifest)
        Preprocessor.df_to_manifest(val_df, val_manifest)
        
        if test_size:
            test_df = pd.concat([X_test, y_test], axis=1)
            test_manifest = os.path.join(output_dir, 'test_manifest.json')
            Preprocessor.df_to_manifest(test_df, test_manifest)
            
            return train_manifest, val_manifest, test_manifest
        
        else:
            return train_manifest, val_manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audios_dir",
        help="Path to directory containing audios to chop",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--drz_dir",
        help="path to directory containing diarization files of audios to chop",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        help="Path to where the main directory 'set' containing all segments will be created",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--summarize",
        help="Whether to summarize the dataset or not",
        type=bool,
        required=False,
        default=True,
    )
    
    args = parser.parse_args()

    Preprocessor.chop(
        args.audios_dir, args.drz_dir, args.output_dir, args.summarize,
    )

