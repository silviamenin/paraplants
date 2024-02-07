import os
os.mkdir('results')
os.chdir('results')
os.system('python ../main.py run_files/protein.pdb run_files/plants.conf run_files/ref_ligand.mol2 2')