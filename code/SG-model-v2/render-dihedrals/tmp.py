
from BuildStructure import placePeptide
from chimera import runCommand as rc
aa = "W"

# Build a sequence
placePeptide(aa, [(-50, 70)] * len(aa), model="peptide")
# Add hydrogens automatically
rc("addh")
# Align so the CA-CB bond is going into the page
rc("align @CA @N")
# Color and highlight
rc("color grey")
rc("repr stick")
rc("shape sphere radius 0.5 center @CD1 color red")
rc("shape sphere radius 0.5 center @CG color blue")
rc("shape sphere radius 0.5 center @CB color green")
rc("shape sphere radius 0.5 center @CA color orange")
# Save
rc("copy file TRP-chi2.png supersample 3")
# rc("close all")
