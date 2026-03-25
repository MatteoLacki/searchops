"""Print the maximum fragment m/z submitted per spectrum in an MGF file.

Usage:
    python scan_mgf_mzs.py <mgf> [--stop-above <mz>]
"""

import argparse
import os
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("mgf", help="Path to MGF file")
parser.add_argument("--stop-above", type=float, default=None,
                    metavar="MZ", help="Stop as soon as a fragment m/z exceeds this value")
args = parser.parse_args()

max_mz = 0.0
spectrum_max = 0.0
title = None
best_title = None
stopped_early = False

file_size = os.path.getsize(args.mgf)
with open(args.mgf) as f, tqdm(total=file_size, unit="B", unit_scale=True, desc=args.mgf) as bar:
    for line in f:
        bar.update(len(line.encode()))
        line = line.strip()
        if line.startswith("TITLE="):
            title = line
            spectrum_max = 0.0
        elif line == "END IONS":
            if spectrum_max > max_mz:
                max_mz = spectrum_max
                best_title = title
        else:
            parts = line.split()
            if len(parts) == 2:
                try:
                    mz = float(parts[0])
                    if mz > spectrum_max:
                        spectrum_max = mz
                    if args.stop_above is not None and mz > args.stop_above:
                        if mz > max_mz:
                            max_mz = mz
                            best_title = title
                        stopped_early = True
                        break
                except ValueError:
                    pass

if stopped_early:
    print(f"Stopped early: found m/z {max_mz:.3f} > {args.stop_above}")
else:
    print(f"Max fragment m/z: {max_mz:.3f}")
print(f"In spectrum: {best_title}")
