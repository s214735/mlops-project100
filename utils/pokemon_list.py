import json
import os

import typer


def write_folder_names_to_json(path: str, output_file: str):
    """
    Write the names of all folders within the specified directory to a JSON file.

    Args:
        path (str): Path to the directory containing folders.
        output_file (str): Path to the output JSON file where folder names will be written.
    """
    try:
        # Get a list of all items in the directory
        items = os.listdir(path)

        # Filter to include only directories
        folders = [item for item in items if os.path.isdir(os.path.join(path, item))]

        # Write folder names to the output JSON file
        with open(output_file, "w") as file:
            json.dump(folders, file, indent=4)

        print(f"Folder names written to {output_file} in JSON format.")

    except FileNotFoundError:
        print(f"The specified path '{path}' does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    typer.run(write_folder_names_to_json)
