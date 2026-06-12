"""
Run this script to print the exact structure of your HDF5 file.
Usage: python inspect_hdf5.py
"""
import h5py
import sys

path = 'data/train_images.hdf5'

def print_structure(name, obj):
    indent = '  ' * name.count('/')
    if isinstance(obj, h5py.Dataset):
        print(f"{indent}DATASET '{name}': shape={obj.shape}, dtype={obj.dtype}")
    else:
        print(f"{indent}GROUP   '{name}'")

with h5py.File(path, 'r') as f:
    print('=== Full HDF5 structure ===')
    f.visititems(print_structure)
    print()
    print('=== Top-level keys ===')
    for k in f.keys():
        item = f[k]
        print(f"  '{k}' -> {type(item).__name__}", end='')
        if isinstance(item, h5py.Dataset):
            print(f"  shape={item.shape} dtype={item.dtype}")
        else:
            subkeys = list(item.keys())[:5]
            print(f"  subkeys (first 5): {subkeys}")
