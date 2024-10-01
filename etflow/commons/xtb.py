import os
import subprocess

from rdkit import Geometry
from rdkit.Chem import rdmolfiles

from etflow.commons import build_conformer

my_dir = f"/tmp/{os.getpid()}"
if not os.path.isdir(my_dir):
    os.mkdir(my_dir)

BOHR_TO_ANGSTROM = 0.529177


def xtb_energy(mol, path_xtb, water=False, dipole=False):
    path = f"/tmp/{os.getpid()}.xyz"
    rdmolfiles.MolToXYZFile(mol, path)
    cmd = [path_xtb, path, "--iterations", str(1000)]
    if water:
        cmd += ["--alpb", "water"]
    if dipole:
        cmd += ["--dipole"]
    n_tries = 5
    result = {}
    for i in range(n_tries):
        try:
            out = subprocess.check_output(
                cmd, stderr=open("/dev/null", "w"), cwd=my_dir
            )
            break
        except subprocess.CalledProcessError:
            if i == n_tries - 1:
                print("xtb_energy did not converge")
                return result  # print(e.returncode, e.output)
    if dipole:
        dipole = [line for line in out.split(b"\n") if b"full" in line][1]
        result["dipole"] = float(dipole.split()[-1])

    runtime = out.split(b"\n")[-8].split()
    result["runtime"] = (
        float(runtime[-2])
        + 60 * float(runtime[-4])
        + 3600 * float(runtime[-6])
        + 86400 * float(runtime[-8])
    )

    energy = [line for line in out.split(b"\n") if b"TOTAL ENERGY" in line]
    result["energy"] = 627.509 * float(energy[0].split()[3])

    gap = [line for line in out.split(b"\n") if b"HOMO-LUMO GAP" in line]
    result["gap"] = 23.06 * float(gap[0].split()[3])

    return result


def xtb_optimize(mol, level, path_xtb):
    in_path = f"{my_dir}/xtb.xyz"
    out_path = f"{my_dir}/xtbopt.xyz"
    if os.path.exists(out_path):
        os.remove(out_path)
    try:
        rdmolfiles.MolToXYZFile(mol, in_path)
        cmd = [path_xtb, in_path, "--opt", level]
        out = subprocess.check_output(cmd, stderr=open("/dev/null", "w"), cwd=my_dir)
        runtime = out.split(b"\n")[-12].split()
        runtime = (
            float(runtime[-2])
            + 60 * float(runtime[-4])
            + 3600 * float(runtime[-6])
            + 86400 * float(runtime[-8])
        )
        out = open(out_path).read().split("\n")[2:-1]
        coords = []
        for line in out:
            _, x, y, z = line.split()
            coords.append([float(x), float(y), float(z)])

        conf = mol.GetConformer()

        for i in range(mol.GetNumAtoms()):
            x, y, z = coords[i]
            conf.SetAtomPosition(i, Geometry.Point3D(x, y, z))
        return runtime
    except Exception as e:
        print(e)
        return None


def worker_fn(job):
    i, xtb_path, mol, positions, e0 = job

    pos = positions / BOHR_TO_ANGSTROM  # Coordinates in Bohr

    num_tries = 5
    for _ in range(num_tries):
        conf = build_conformer(pos)
        mol.RemoveAllConformers()
        mol.AddConformer(conf)
        success = xtb_optimize(mol, "normal", path_xtb=xtb_path)
        if success:
            break

    if not success:
        (i, None, None, None, None)

    res = xtb_energy(mol, dipole=True, path_xtb=xtb_path)
    energy, dipole, gap = (
        res["energy"],
        res["dipole"],
        res["gap"],
    )

    if e0 is not None:
        energy -= e0

    optimized_positions = mol.GetConformer().GetPositions().reshape(1, -1, 3)
    return i, optimized_positions, energy, dipole, gap
