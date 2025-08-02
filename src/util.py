import os
import pandas as pd

SOURCE_FOLDER = "/content/drive/MyDrive/gdp_nowcasting_seminar/Data/pickles"


def save_all_dfs_to_drive(data_frames, target_folder=SOURCE_FOLDER):
    """
    Saves each DataFrame in the data_frames dictionary as a pickle file to the specified Google Drive folder.

    Args:
        data_frames (dict): Dictionary of DataFrames where keys are file names (without extension).
        target_folder (str): Folder path in Google Drive to save the pickle files.
    """
    # Create the folder if it doesn't exist
    os.makedirs(target_folder, exist_ok=True)

    for name, df in data_frames.items():
        filename = os.path.join(target_folder, f"{name}.pkl")
        df.to_pickle(filename)
        print(f"Saved {name} to {filename}")


def load_all_pickles_from_drive(source_folder=SOURCE_FOLDER):
    """
    Loads all .pkl files from a specified Google Drive folder into a dictionary of DataFrames.

    Args:
        source_folder (str): Path to the folder containing .pkl files.

    Returns:
        dict: Dictionary of DataFrames with keys as file names (without .pkl extension).
    """
    data_frames = {}

    if not os.path.exists(source_folder):
        raise FileNotFoundError(f"Folder not found: {source_folder}")

    for filename in os.listdir(source_folder):
        if filename.endswith(".pkl"):
            filepath = os.path.join(source_folder, filename)
            df_name = filename.replace(".pkl", "")
            data_frames[df_name] = pd.read_pickle(filepath)
            print(f"Loaded {df_name} from {filepath}")

    return data_frames
