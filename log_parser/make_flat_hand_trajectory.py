import argparse
import h5py

HAND_SIZE = 36

def make_flat_hand_trajectory(input_path, output_path, hand_size: int = HAND_SIZE):
    """Flatten grouped datasets into trajectories of fixed length."""
    with h5py.File(input_path, 'r') as f, h5py.File(output_path, 'w') as g:
        groups = list(f.keys())
        state_length = f[groups[0]]["state"].shape[1]
        g_attrs_total = 0
        for group in groups:
            num_states = f[group]["state"].shape[0]
            g_attrs_total += num_states // hand_size

        dset_states = g.create_dataset(
            "states", (g_attrs_total, hand_size, state_length),
            dtype=f[groups[0]]["state"].dtype
        )
        dset_actions = g.create_dataset(
            "actions", (g_attrs_total, hand_size, 1),
            dtype=f[groups[0]]["action"].dtype
        )

        hand_idx = 0
        print(f"starting to copy {g_attrs_total} hands from {len(groups)} groups to {output_path}")
        for group in groups:
            group_data = f[group]
            num_states = group_data["state"].shape[0]
            full_hands = num_states // hand_size
            for i in range(full_hands):
                start = i * hand_size
                end = start + hand_size
                dset_states[hand_idx] = group_data["state"][start:end]
                dset_actions[hand_idx] = group_data["action"][start:end]
                hand_idx += 1

        g.attrs["total_states_saved"] = hand_idx

        print(f"Copied {hand_idx}/{g_attrs_total} hands from {len(groups)} groups to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge groups into trajectories of fixed length.")
    parser.add_argument('-i', '--input', required=True, help='Input HDF5 file')
    parser.add_argument('-o', '--output', required=True, help='Output HDF5 file')
    parser.add_argument('--hand-size', type=int, default=HAND_SIZE, help='Number of states per hand')
    args = parser.parse_args()
    make_flat_hand_trajectory(args.input, args.output, args.hand_size)