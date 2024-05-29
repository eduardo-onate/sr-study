import json
import os
from typing import Any

class JSONHandler:
    """
    A handler class to facilitate reading from and writing to JSON files.
    """

    @staticmethod
    def read(json_path: str) -> Any:
        """
        Read data from a JSON file.

        Parameters:
        - json_path: Path to the JSON file.
        :return: Parsed data from the JSON file.

        Raises exceptions if:
        - The JSON file doesn't exist.
        - The JSON file is not a valid JSON format.
        """
        if not json_path.lower().endswith('.json'):
            raise ValueError(f"The file '{json_path}' is not a valid JSON file.")
        
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"The JSON file '{json_path}' doesn't exist.")
        
        with open(json_path, 'r') as json_file:
            return json.load(json_file)

    @staticmethod
    def write(data: Any, json_path: str, indent: int = 4) -> None:
        """
        Write data to a JSON file.

        Parameters:
        - data: Data to be written to the JSON file.
        - json_path: Path to the JSON file.
        - indent: Number of spaces for indentation in the JSON file.

        Raises exceptions if:
        - The JSON file is not a valid JSON format.
        - There's an issue writing to the file.
        """
        if not json_path.lower().endswith('.json'):
            raise ValueError(f"The file '{json_path}' is not a valid JSON file.")
        
        with open(json_path, 'w') as json_file:
            json.dump(data, json_file, indent=indent)
