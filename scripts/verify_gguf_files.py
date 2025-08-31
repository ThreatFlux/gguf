#!/usr/bin/env python3
"""Verify GGUF files in the data directory"""

import struct
import os
import sys
from pathlib import Path

def read_gguf_header(filepath):
    """Read and verify GGUF file header"""
    with open(filepath, 'rb') as f:
        # Read magic number
        magic = f.read(4)
        if magic != b'GGUF':
            return None, f"Invalid magic: {magic}"
        
        # Read version
        version = struct.unpack('<I', f.read(4))[0]
        
        # Read tensor count
        tensor_count = struct.unpack('<Q', f.read(8))[0]
        
        # Read metadata count  
        metadata_count = struct.unpack('<Q', f.read(8))[0]
        
        return {
            'magic': magic.decode('ascii'),
            'version': version,
            'tensor_count': tensor_count,
            'metadata_count': metadata_count,
            'file_size_mb': os.path.getsize(filepath) / (1024 * 1024)
        }, None

def main():
    data_dir = Path('data')
    gguf_files = list(data_dir.glob('*.gguf'))
    
    print(f"Found {len(gguf_files)} GGUF files in data/")
    print("=" * 60)
    
    results = []
    for filepath in sorted(gguf_files):
        header, error = read_gguf_header(filepath)
        if header:
            results.append((filepath.name, header))
            print(f"\n✓ {filepath.name}")
            print(f"  Size: {header['file_size_mb']:.1f} MB")
            print(f"  Version: {header['version']}")
            print(f"  Tensors: {header['tensor_count']}")
            print(f"  Metadata: {header['metadata_count']}")
        else:
            print(f"\n✗ {filepath.name}: {error}")
    
    print("\n" + "=" * 60)
    print(f"Summary: {len(results)}/{len(gguf_files)} files are valid GGUF")
    
    # Summary statistics
    if results:
        print("\nFile Statistics:")
        print(f"  Total size: {sum(h['file_size_mb'] for _, h in results):.1f} MB")
        print(f"  Total tensors: {sum(h['tensor_count'] for _, h in results)}")
        print(f"  Average metadata entries: {sum(h['metadata_count'] for _, h in results) / len(results):.1f}")
        
        # Group by version
        versions = {}
        for name, header in results:
            v = header['version']
            if v not in versions:
                versions[v] = []
            versions[v].append(name)
        
        print(f"\nVersions found:")
        for v, files in versions.items():
            print(f"  Version {v}: {len(files)} files")

if __name__ == '__main__':
    main()