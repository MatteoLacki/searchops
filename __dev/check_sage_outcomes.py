import pandas as pd
import re

from pandas_ops.io import read_df
from timstofu.conversion.reformat import open_pmsms
from pathlib import Path

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 5)


sage_version = "v0.14.7"
sage_version = "devel"
path = Path(f"temp/F9477/optimal2tier/sage/{sage_version}/p12f15/human/dir")

path = Path(f"/home/matteo/tmp/sagebug/out")
frags = read_df(path/"matched_fragments.sage.tsv")
prec = read_df(path/"results.sage.tsv")



weird = frags.iloc[[frags.fragment_mz_experimental.argmax()]]

prec[prec.psm_id == int(weird.psm_id.iloc[0])]

found_prec = 

pmsms = open_pmsms("temp/F9477/optimal2tier/pmsms.mmappet")
pmsms.data.mz.max()



capture = False
lines = []

with open("temp/F9477/optimal2tier/mgf.mgf") as f:
    for line in f:
        if line.startswith('BEGIN IONS'):
            capture = False  # reset

        if 'TITLE="idx=1080501 frame=8515 scan=130 tof=268640 iim=1.1697 I=374"' in line:
            capture = True
            lines.append("BEGIN IONS\n")  # include header explicitly if needed
            lines.append(line)
            continue

        if capture:
            lines.append(line)
            if line.startswith("END IONS"):
                break

spectrum = "".join(lines)
print(spectrum)
print(spectrum[:1000])

with open("/home/matteo/tmp/sagebug/tiny.mgf", "w") as f:
    f.write(spectrum)