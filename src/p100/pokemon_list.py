import os
import typer

def write_folder_names_to_file(path: str, output_file: str):
    """
    Write the names of all folders within the specified directory to a text file.

    Args:
        path (str): Path to the directory containing folders.
        output_file (str): Path to the output text file where folder names will be written.
    """
    try:
        # Get a list of all items in the directory
        items = os.listdir(path)

        # Filter to include only directories
        folders = [item for item in items if os.path.isdir(os.path.join(path, item))]

        # Write folder names to the output file
        with open(output_file, 'w') as file:
            file.write("\n".join(folders))

        print(f"Folder names written to {output_file}")

    except FileNotFoundError:
        print(f"The specified path '{path}' does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    typer.run(write_folder_names_to_file)