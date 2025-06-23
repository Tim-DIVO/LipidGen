#!/usr/bin/env python
import subprocess
import numpy as np

def run_assembly(n_lipids=10000):
    chol_values = [0.3, 0.4, 0.5]
    combos = ["equal", "u70", "u30"]  # for how unsat/sat is split

    i = 1

    for chol in chol_values:
        # We'll define 3 splits for (unsat, sat) => (U, S)
        # total = 1 - chol
        # combos:
        #   "equal" => each 0.5*(1-chol)
        #   "u70"   => unsat=0.7*(1-chol), sat=0.3*(1-chol)
        #   "u30"   => unsat=0.3*(1-chol), sat=0.7*(1-chol)

        remain = 1.0 - chol
        splits = {
            "equal": (0.5*remain, 0.5*remain),
            "u70":   (0.7*remain, 0.3*remain),
            "u30":   (0.3*remain, 0.7*remain),
        }

        chol_str = f"{int(round(chol * 100)):02d}"

        for c in combos:
            uFrac, sFrac = splits[c]

            # Outer => typeOne=uFrac, typeTwo=sFrac, typeFive=chol
            outOne = uFrac
            outTwo = sFrac
            outFive= chol

            # Inner => typeThr=uFrac, typeFour=sFrac, typeFive=chol
            inThr  = uFrac
            inFour = sFrac
            inFive = chol

            # We'll build a unique name for the data file
            out_name = f"bilayer_chol{chol_str}_{c}.data"
            # Command line:
            cmd = [
                "python", "assembly.py",
                str(n_lipids),
                f"{outOne:.4f}", f"{outTwo:.4f}", f"{outFive:.4f}",
                f"{inThr:.4f}", f"{inFour:.4f}", f"{inFive:.4f}",
                f"{i}"
            ]
            print("Running:", " ".join(cmd))
            subprocess.run(cmd)

            # Move or rename the bilayer.data => out_name
            subprocess.run(["mv", "bilayer.data", out_name])

if __name__ == "__main__":
    run_assembly()
