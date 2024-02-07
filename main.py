import multiprocessing
from sys import argv, exit
import os
import MDAnalysis as mda
from rdkit import DataStructs, Chem
import rdkit
from parallelizer import parallelizer
import prolif
import numpy as np
import pandas as pd
from mol2supplier import mol2_supplier
import time



helper = '''
Fingerprint docker
Use: python main.py <protein.pdb> <plants.conf> <reference_ligand.mol2> <n_jobs>
'''

def file_finder(file, extension=None):
    if os.path.exists(file):
        if not extension == None:
            if str(file).endswith(f'.{extension}'):
                pass
            else:
                exit(f'ERROR\t{file} is not .{extension}')
        
        return os.path.abspath(file)
    
    else:
        exit(f'ERROR\t{file} not found')
    


def mp_setup(ligand, n_proc, plants_config_lines, ligand_line, output_line, protein_pdb, reference_ligand):
    mp_args = []

    ### create a folder for multiprocessing operations
    if not os.path.exists('chunks'):
        os.mkdir('chunks')
    wd = os.path.abspath('chunks')

    blocks = []
    mol2 = mol2_supplier(ligand).get_molecules(rdkit=False)

    n_ligand = len(mol2)
    chunk_size = n_ligand // n_proc

    for i in range(0, len(mol2), chunk_size):
        chunk = mol2[i:i+chunk_size]
        text = ''
        for mol in chunk:
            for line in mol:
                text += line
        
        output_path = f'{wd}/output_{i}/'
        ligand_path = f'{wd}/chunk_{i}.mol2'

        with open(ligand_path, 'w') as f:
            f.write(text)

        ### write modified lines for ligand and output
        new_ligand_line = f'ligand_file {ligand_path}\n'
        new_output_line = f'output_dir {output_path}\n'

        plants_config_lines[ligand_line] = new_ligand_line
        plants_config_lines[output_line] = new_output_line

        ### write plants.conf
        config_path = f'{wd}/plants_{i}.conf'
        plants_config_text = ''.join(plants_config_lines)
        with open(config_path, 'w') as conf:
            conf.write(plants_config_text)

        mp_args.append((i, ligand_path, config_path, output_path, protein_pdb, reference_ligand))

    return mp_args



### execute docking
def docking(i, chunk, plants_config, output, protein_pdb, reference_ligand):
    start = time.time()
    os.system(f'cd {os.path.commonpath([chunk, output])}')
    os.system(f'plants1.2 --mode screen {plants_config} > chunks/stdout_{i}')

    docked_file = f'{output}/docked_ligands.mol2'
    score_file = f'{output}/features.csv'
    docking_score = []
    with open(score_file, 'r') as s:
        for line in s.readlines()[1:]:
            docking_score.append(line.split(',')[1])

    ### FINGERPRINT CALCULATIONS
    protein_mda = mda.Universe(protein_pdb).atoms
    protein_ifp = prolif.Molecule.from_mda(protein_mda)

    fp = prolif.Fingerprint()
    ligands_iter = prolif.mol2_supplier(docked_file)
    fp.run_from_iterable(ligands_iter, protein_ifp, residues='all', n_jobs=1, progress=False)

    df = fp.to_dataframe(index_col="Pose")

    ### POSE RESCORING
    ref_lig_ifp = prolif.mol2_supplier(reference_ligand)
    fp_ref = prolif.Fingerprint()
    fp_ref.run_from_iterable(ref_lig_ifp, protein_ifp, residues='all', n_jobs=1, progress=False)
    df_ref = fp_ref.to_dataframe(index_col="Pose")

    # set the "pose index" to -1
    df_ref.rename(index={0: -1}, inplace=True)
    # set the ligand name to be the same as poses
    df_ref.rename(columns={str(protein_ifp[0].resid): df_ref.columns.levels[0][0]}, inplace=True)

    # concatenate both dataframes
    df_ref_poses = (
        pd.concat([df_ref, df])
        .fillna(False)
        .sort_index(
            axis=1,
            level=1,
            key=lambda index: [prolif.ResidueId.from_string(x) for x in index],
        )
    )

    bitvectors = prolif.to_bitvectors(df_ref_poses)
    tanimoto_score = DataStructs.BulkTanimotoSimilarity(bitvectors[0], bitvectors[1:])
    
    result_file = f'{output}/result_{i}.sdf'
    with Chem.SDWriter(result_file) as w:
        for molecule, docking_score, tanimoto_score in zip(mol2_supplier(docked_file).get_molecules(), docking_score, tanimoto_score):
            molecule.GetProp('_Name')
            molecule.SetProp('Docking_score', str(docking_score))
            molecule.SetProp('Tanimoto_score', str(tanimoto_score))
            w.write(molecule)

    end = time.time()
    print(f'time for parallel process {i}: {end-start}')

    return result_file





def main():
    ### PARSE and SETUP work
    
    if argv[1] == '-h' or argv[1] == '--help':
        print(helper)
        exit(0)

    else:
        protein_pdb = file_finder(argv[1])
        plants_config = file_finder(argv[2])
        reference_ligand = file_finder(argv[3])
        n_proc = int(argv[4])
        

    with open(plants_config, 'r') as config:
        plants_config_lines = config.readlines()
        for i, line in enumerate(plants_config_lines):
            if line.startswith('protein_file'):
                protein_mol2 = file_finder(line.split()[1], 'mol2')
                plants_config_lines[i] = f'protein_file {protein_mol2}\n'

            elif line.startswith('ligand_file'):
                ligand_line = i
                ligand_mol2 = file_finder(line.split()[1], 'mol2')

            elif line.startswith('output'):
                output_line = i


    ### DOCKING
    docking_poses = os.path.abspath('docking_poses.mol2')
    docking_args = mp_setup(ligand_mol2, n_proc, plants_config_lines, ligand_line, output_line, protein_pdb, reference_ligand)

    results = parallelizer.run(docking_args, docking, n_proc, 'Calculation progress')
    
    files = ''
    for file in results:
        files += f'{file} '

    os.system(f'cat {files} > results.sdf')


    


if __name__ == '__main__':
    start = time.time()
    multiprocessing.set_start_method('spawn')
    main()
    end = time.time()
    print(f'time elapsed: {end-start}')