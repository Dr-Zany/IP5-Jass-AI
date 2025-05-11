import argparse
import h5py

def make_flate(input_path, output_path):
    with h5py.File(input_path, 'r') as f, h5py.File(output_path, 'w') as g:
        total_states = f.attrs["total_states_saved"]
        groups = list(f.keys())
        state_length = f[groups[0]]["state"].shape[1]
        g.attrs["total_states_saved"] = total_states

        dset_states = g.create_dataset("states", (total_states, state_length), dtype=f[groups[0]]["state"].dtype)
        dset_actions = g.create_dataset("actions", (total_states,1), dtype=f[groups[0]]["action"].dtype)

        num_states = 0
        print(f"starting to copy {total_states} states from {len(groups)} groups to {output_path}")
        for group in groups:
            group_data = f[group]
            num_group_states = group_data["state"].shape[0]
            dset_states[num_states:num_states + num_group_states] = group_data["state"]
            dset_actions[num_states:num_states + num_group_states] = group_data["action"]
            num_states += num_group_states

        print(f"Copied {num_states}/{total_states} states from {len(groups)} groups to {output_path}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge and flatten all datasets in an HDF5 file.")
    parser.add_argument('-i', '--input', required=True, help='Input HDF5 file')
    parser.add_argument('-o', '--output', required=True, help='Output HDF5 file')
    args = parser.parse_args()
    make_flate(args.input, args.output)