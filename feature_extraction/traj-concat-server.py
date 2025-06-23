import os
import glob

def read_lammps_traj_frames(filepath):
    """
    Reads a LAMMPS trajectory file and splits it into individual frames.
    Each frame is a list of lines. Returns None if file not found.
    """
    frames = []
    current_frame = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                # A new frame starts with "ITEM: TIMESTEP"
                # Ensure current_frame is not empty before appending (handles first frame)
                if line.strip() == "ITEM: TIMESTEP" and current_frame:
                    frames.append(list(current_frame)) # Add a copy of the completed frame
                    current_frame = [] # Reset for the new frame
                current_frame.append(line)
            if current_frame: # Add the last frame in the file
                frames.append(list(current_frame))
    except FileNotFoundError:
        print(f"Error: File not found {filepath}")
        return None
    return frames

def get_timestep_from_frame(frame_lines):
    """
    Extracts the timestep value from a frame (list of lines).
    Assumes the line after "ITEM: TIMESTEP" is the timestep value.
    """
    if not frame_lines or len(frame_lines) < 2:
        return None
    # Expect "ITEM: TIMESTEP" then the timestep value
    if frame_lines[0].strip() == "ITEM: TIMESTEP":
        try:
            return int(frame_lines[1].strip())
        except ValueError:
            print(f"Error: Could not parse timestep from frame starting with: {frame_lines[0:2]}")
            return None
    return None # Should not happen if frames are structured correctly

def modify_frame_timestep(frame_lines, new_timestep):
    """
    Modifies the timestep in a frame (list of lines) and returns a new list of lines (frame).
    The input frame_lines is not modified in place.
    """
    modified_frame = list(frame_lines) # Make a copy to avoid modifying the original
    if not modified_frame or len(modified_frame) < 2:
        print("Error: Cannot modify timestep in an empty or too short frame.")
        return frame_lines # Return original if error

    if modified_frame[0].strip() == "ITEM: TIMESTEP":
        modified_frame[1] = str(new_timestep) + "\n"
    else:
        print("Error: Frame does not start with 'ITEM: TIMESTEP', cannot modify timestep.")

    return modified_frame

def process_folder(folder_path):
    """
    Processes a single configf_### folder to create concat-traj.lammpstrj.
    """
    short_traj_filename = "full_traj_short.lammpstrj"
    long_traj_filename = "full_traj_long.lammpstrj"
    output_traj_filename = "concat-traj.lammpstrj"

    short_traj_path = os.path.join(folder_path, short_traj_filename)
    long_traj_path = os.path.join(folder_path, long_traj_filename)
    output_traj_path = os.path.join(folder_path, output_traj_filename)

    print(f"\nProcessing folder: {folder_path}")

    if not os.path.exists(short_traj_path):
        print(f"  Skipping: {short_traj_filename} not found.")
        return
    if not os.path.exists(long_traj_path):
        print(f"  Skipping: {long_traj_filename} not found.")
        return

    short_frames_all = read_lammps_traj_frames(short_traj_path)
    long_frames_all = read_lammps_traj_frames(long_traj_path)

    if short_frames_all is None or long_frames_all is None:
        print(f"  Failed to read one or both trajectory files. Skipping.")
        return
    
    if not short_frames_all:
        print(f"  Warning: {short_traj_filename} is empty or unreadable. Skipping.")
        return
    if not long_frames_all:
        print(f"  Warning: {long_traj_filename} is empty or unreadable. Skipping.")
        return

    # --- Select frames from short trajectory ---
    # Target timesteps: 0, 5100, 9900, 15000, 20100
    target_short_timesteps = [0, 5100, 9900, 15000, 20100]
    
    # Create a dictionary of {timestep: frame_content} for quick lookup
    short_frames_map = {}
    for frame in short_frames_all:
        ts = get_timestep_from_frame(frame)
        if ts is not None:
            short_frames_map[ts] = frame
        else:
            print(f"  Warning: Could not read timestep for a frame in {short_traj_filename}.")

    final_selected_short_frames = []
    for ts in target_short_timesteps:
        if ts in short_frames_map:
            final_selected_short_frames.append(short_frames_map[ts])
        else:
            print(f"  Warning: Target timestep {ts} not found in {short_traj_filename}.")
            
    if len(final_selected_short_frames) != len(target_short_timesteps):
         print(f"  Note: Expected {len(target_short_timesteps)} frames from {short_traj_filename}, but found {len(final_selected_short_frames)} matching target timesteps.")


    # --- Prepare to write output ---
    total_frames_written = 0
    frames_written_short = 0
    frames_written_long = 0

    with open(output_traj_path, 'w') as outfile:
        # Write selected frames from short trajectory (original timesteps)
        for frame_content in final_selected_short_frames:
            for line in frame_content:
                outfile.write(line)
            frames_written_short += 1
        
        if frames_written_short > 0:
            print(f"  Wrote {frames_written_short} frames from {short_traj_filename}.")
        else:
            print(f"  No frames written from {short_traj_filename}.")


        # Write all frames from long trajectory, adjusting timesteps
        timestep_offset_long = 25000 # Actual start step of the long simulation
        for frame_content in long_frames_all:
            original_timestep = get_timestep_from_frame(frame_content)
            if original_timestep is not None:
                new_timestep = original_timestep + timestep_offset_long
                modified_frame = modify_frame_timestep(frame_content, new_timestep)
                for line in modified_frame:
                    outfile.write(line)
                frames_written_long += 1
            else:
                print(f"  Warning: Could not parse timestep for a frame in {long_traj_filename}. Skipping that frame.")
        
        if frames_written_long > 0:
            print(f"  Wrote {frames_written_long} frames from {long_traj_filename} (timesteps adjusted by +{timestep_offset_long}).")
        else:
             print(f"  No frames written from {long_traj_filename}.")

    total_frames_written = frames_written_short + frames_written_long
    
    # --- Summary ---
    ideal_long_frames_count = len(long_frames_all)
    expected_ideal_total = len(target_short_timesteps) + ideal_long_frames_count # Ideal is 5 + 96 = 101

    print(f"  Successfully created {output_traj_filename}")
    print(f"  Total frames written to {output_traj_filename}: {total_frames_written}")
    
    user_target_total_frames = 100
    if total_frames_written != user_target_total_frames:
        print(f"  Note: User's specified target was {user_target_total_frames} frames.")
        print(f"        The generated file has {total_frames_written} frames.")
        if total_frames_written == expected_ideal_total:
             print(f"        This count ({total_frames_written}) matches the ideal expectation of {len(target_short_timesteps)} selected short frames and all {ideal_long_frames_count} long frames.")
        elif frames_written_short == len(target_short_timesteps) and frames_written_long == ideal_long_frames_count:
             # This case should be covered by the above if total_frames_written == expected_ideal_total
             pass
        else:
             print(f"        This count ({total_frames_written}) differs from the ideal sum ({expected_ideal_total}) possibly due to missing target timesteps in source files or parsing issues.")
    else: # total_frames_written == user_target_total_frames (i.e. 100)
        print(f"  This matches the user's specified target of {user_target_total_frames} frames.")
        if total_frames_written != expected_ideal_total:
            print(f"  Note: This required a deviation from using all {ideal_long_frames_count} long frames or the {len(target_short_timesteps)} specified short frames (ideal total: {expected_ideal_total}).")


def main():
    # Assumes the script is run from the parent directory of configf_* folders
    base_dir = "." 
    # Find directories like configf_69, configf_322, configf_1394, etc.
    folders_to_process = sorted([
        d for d in glob.glob(os.path.join(base_dir, "configf_*"))
        if os.path.isdir(d) and os.path.isfile(os.path.join(d, "stable.txt"))
    ])

    if not folders_to_process:
        print(f"No 'configf_*' folders found in the current directory ('{os.getcwd()}').")
        print("Please run this script from the directory that contains your 'configf_...' folders.")
        return

    print(f"Found {len(folders_to_process)} 'configf_*' folders to process: {folders_to_process}")

    for folder in folders_to_process:
        process_folder(folder)

    print("\nScript finished.")

if __name__ == "__main__":
    main()