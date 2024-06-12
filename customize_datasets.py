"""
This script preprocesses the original speaker recognition datasets to create custom 
datasets depending on the maximun length of utterances and total time spoken by speaker.

Args:
    '-d', '--dataset_dir' (str): Path to the directory of the dataset.
    '-s', '--segment_duration' (int): Maximum duration of audio segments by speaker (in seconds).
    
Returns:
    Tuple[str, float]: Path to the checkpoint file and duration of the training process.
        or
    Tuple[str, float, str]: Path to the checkpoint file, duration of the training process
                            and path to the base dataset directory.  

Stages and possible use cases:

    1) Creation of custom dataset based on `segment_duration`.
    
    2) Creation of a main manifest file.
                
    3) Separation of manifest file into enrollment and verification sets.
"""

import argparse

from tools.preprocessor import Preprocessor


def main(dataset_dir: str, segment_duration: int, manifest_dir: str ) -> None:

    # Creation of custom dataset based on `segment_duration`.
    segments_dir = Preprocessor.segment(dataset_dir, duration=segment_duration)

    # Creation of main manifest file.
    manifest = Preprocessor.create_manifest(segments_dir, manifest_dir)
    print(f"\nMain manifest file created at: {manifest}")
    
  
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d',
        '--dataset_dir',
        help="Path to directory of main dataset",
        type=str,
    )
    parser.add_argument(
        '-s',
        '--segment_duration',
        help="Path to directory containing labeled diarization files of audios to create base dataset",
        type=int,
    )
    parser.add_argument(
        '-m',
        '--manifest_dir',
        help="Path to directory to save manifest files",
        type=str,
    )
    
    args = parser.parse_args()
    
    main(args.dataset_dir, args.segment_duration, args.manifest_dir)