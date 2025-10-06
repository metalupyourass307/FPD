import numpy as np
import matplotlib.pyplot as plt

def calculate_distance_from_dump(filename, box_size=128.0):
    """
    Parses a LAMMPS-like dump file to calculate the distance between two colloids over time,
    accounting for Periodic Boundary Conditions (PBC).

    Args:
        filename (str): The path to the dump file.
        box_size (float): The size of the simulation box (assuming a cubic box).

    Returns:
        tuple: A tuple containing two lists (timesteps, distances).
    """
    timesteps = []
    distances = []

    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found. Please make sure the filename in the script matches the actual data file.")
        return None, None

    i = 0
    while i < len(lines):
        if lines[i].strip() == "ITEM: TIMESTEP":
            timestep = int(lines[i+1].strip())
            
            # Find the start of the atom data
            atom_data_start_index = -1
            for j in range(i, i + 10): # Search within the next 10 lines
                if lines[j].strip().startswith("ITEM: ATOMS"):
                    atom_data_start_index = j + 1
                    break
            
            if atom_data_start_index != -1:
                try:
                    # Read the coordinates of the two particles
                    line1_data = lines[atom_data_start_index].strip().split()
                    pos1 = np.array([float(line1_data[3]), float(line1_data[4]), float(line1_data[5])])

                    line2_data = lines[atom_data_start_index + 1].strip().split()
                    pos2 = np.array([float(line2_data[3]), float(line2_data[4]), float(line2_data[5])])

                    # Calculate distance with PBC
                    delta = pos1 - pos2
                    delta = delta - box_size * np.round(delta / box_size)
                    distance = np.linalg.norm(delta)
                    
                    timesteps.append(timestep)
                    distances.append(distance)
                    
                    # Move index to the next TIMESTEP block
                    i = atom_data_start_index + 2 
                except (IndexError, ValueError) as e:
                    print(f"Warning: Could not parse atom data at timestep {timestep}. Error: {e}")
                    i += 1
            else:
                i += 1
        else:
            i += 1
            
    return timesteps, distances

# --- Main script ---
if __name__ == "__main__":
    # ==================== FIX: Corrected filenames to match your files ====================
    cc_dump_file = 'dump_cc.dat'
    cr_dump_file = 'dump_cr.dat'
    # =======================================================================================
    
    # Process both files
    print(f"Processing {cc_dump_file}...")
    cc_time, cc_dist = calculate_distance_from_dump(cc_dump_file)
    print(f"Processing {cr_dump_file}...")
    cr_time, cr_dist = calculate_distance_from_dump(cr_dump_file)

    # Plotting
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    if cc_time and cc_dist:
        ax.plot(cc_time, cc_dist, marker='o', markersize=4, linestyle='-', label='Constant Charge (CC) at pH=4')
        print(f"Found {len(cc_time)} data points for CC simulation.")
    else:
        print("No data found or processed for CC simulation.")
    
    if cr_time and cr_dist:
        ax.plot(cr_time, cr_dist, marker='s', markersize=4, linestyle='--', label='Charge Regulation (CR) at pH=4')
        print(f"Found {len(cr_time)} data points for CR simulation.")
    else:
        print("No data found or processed for CR simulation.")

    # Formatting the plot
    ax.set_title('Colloid Aggregation Behavior Comparison at pH=4', fontsize=16)
    ax.set_xlabel('Time (Simulation Steps)', fontsize=12)
    ax.set_ylabel('Distance Between Colloids', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True)
    
    # Set a logical y-axis limit, starting from a value slightly less than 2*Radius
    # 2 * RADI = 2 * 3.2 = 6.4
    if (cc_dist or cr_dist):
        all_dists = (cc_dist if cc_dist is not None else []) + (cr_dist if cr_dist is not None else [])
        if all_dists:
            ax.set_ylim(bottom=min(all_dists) * 0.95, top=max(all_dists) * 1.05)
    
    # Save the figure
    output_filename = 'distance_comparison_ph4.png'
    plt.savefig(output_filename, dpi=300)
    
    print(f"Plot has been saved to '{output_filename}'")