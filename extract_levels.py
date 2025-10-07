import os
import argparse
import h5py
from tifffile import imsave

def extract_levels_from_h5(levels, dataset_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    raw_nuc_path = os.path.join(output_folder, 'nuc/')
    raw_cyto_path = os.path.join(output_folder, 'cyto/')
    os.makedirs(raw_nuc_path, exist_ok=True)
    os.makedirs(raw_cyto_path, exist_ok=True)

    # Traverse HDF5 files in the dataset folder
    for file in os.listdir(dataset_folder):
        if file.endswith(".h5") or file.endswith(".hdf5"):
            file_path = os.path.join(dataset_folder, file)
            specimen_name = os.path.splitext(file)[0]

            with h5py.File(file_path, 'r') as f:
                for level in levels:
                    nuclei = f['t00000/s00/0/cells'][:, level, :].astype('uint16')
                    cyto = f['t00000/s01/0/cells'][:, level, :].astype('uint16')

                    nuc_name = f"nuc_{specimen_name}_{level}.tiff"
                    cyto_name = f"cyto_{specimen_name}_{level}.tiff"

                    nuc_out_path = os.path.join(raw_nuc_path, nuc_name)
                    cyto_out_path = os.path.join(raw_cyto_path, cyto_name)

                    if not (os.path.exists(nuc_out_path) and os.path.exists(cyto_out_path)):
                        imsave(nuc_out_path, nuclei)
                        imsave(cyto_out_path, cyto)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract 2D levels from 3D HDF5 pathology volumes.")
    parser.add_argument("--levels", type=int, nargs='+', required=True, help="List of levels to extract")
    parser.add_argument("--dataset_folder", type=str, required=True, help="Path to the folder containing .h5 files.")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to the output folder to store extracted TIFFs.")

    args = parser.parse_args()

    extract_levels_from_h5(args.levels, args.dataset_folder, args.output_folder)