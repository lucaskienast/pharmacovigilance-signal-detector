"""
vina_helpers_bio.py — Tiny wrappers for BLAST+, Meeko, AutoDock Vina, and SDF generation.

All usage is documented inline in each function docstring. Keep this file in your project
and import what you need. No README necessary.

Requires command-line tools in PATH:
- BLAST+: blastp, blastdbcmd      https://blast.ncbi.nlm.nih.gov/
- Meeko: mk_prepare_ligand.py, mk_prepare_receptor.py   https://github.com/forlilab/Meeko
- AutoDock Vina: vina             http://vina.scripps.edu/
- (Optional) Open Babel: obabel   https://openbabel.org/
- (Optional) RDKit (Python)       https://www.rdkit.org/

Tip: set BLASTDB env to your database dir, e.g. `export BLASTDB=/home/ubuntu/blastdb`
"""

from __future__ import annotations
import csv
import os
import re
import shutil
import subprocess
import tempfile
import textwrap
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

__all__ = [
    "BLAST_OUTFMT",
    "blastp_from_seq",
    "meeko_ligand",
    "meeko_receptor",
    "smiles_to_sdf_3d",
    "vina_score_only",
    "vina_dock",
    "clean_pdb_for_docking"
]

# ---- Biopython config dir: keep cache inside workspace ----
_BIO_CFG = os.environ.get("BIOPYTHON_CONFIG_DIR")
if not _BIO_CFG:
    # Prefer explicit WORKSPACE_DIR if present, else default to /workspace
    _ws = os.environ.get("WORKSPACE_DIR", "/workspace")
    cfg = Path(_ws) / ".config" / "biopython" / "Bio" / "Entrez" / "DTDs"
    cfg.mkdir(parents=True, exist_ok=True)
    os.environ["BIOPYTHON_CONFIG_DIR"] = str(cfg.parent.parent)  # .../.config/biopython

def _run(cmd: Sequence[str]) -> str:
    """Run a subprocess and return stdout; include stdout/stderr on failure."""
    try:
        cp = subprocess.run(list(cmd), check=True, capture_output=True, text=True)
        return cp.stdout
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\nSTDOUT:\n{e.stdout}\n\nSTDERR:\n{e.stderr}"
        ) from e


# ----------------------------- BLAST+ ----------------------------------------

BLAST_OUTFMT = "6 sacc pident length evalue bitscore sscinames"


def blastp_from_seq(
    seq: str,
    db: str = "pdbaa",
    *,
    max_targets: int = 10,
    threads: int = 2,
    blastdb: str | None = None,
) -> List[Dict[str, str]]:
    """Run BLASTP against a local DB and return tabular hits as dicts.

    Args:
        seq: Protein sequence (FASTA body).
        db: BLAST protein DB name (e.g., "pdbaa", "swissprot").
        max_targets: Max target sequences.
        threads: Number of CPU threads.
        blastdb: Optional BLASTDB directory to export for this call.

    Returns:
        List of dictionaries with keys: sacc, pident, length, evalue, bitscore, sscinames.

    Notes:
        - Ensure BLAST+ is installed and DBs are in $BLASTDB; taxonomy fields require taxdb.*.
        - CPU-bound; set `threads` as appropriate.

    Example:
        >>> hits = blastp_from_seq("MTEYKLVVVGAGGVGKSALTIQLIQNHF...",
        ...                         db="pdbaa", max_targets=5, threads=4,
        ...                         blastdb="/home/ubuntu/blastdb")
        >>> [(h["sacc"], h["pident"]) for h in hits][:3]
        [('sp|P01112|', '95.1'), ('sp|Q61PZ0|', '93.7'), ...]
    """
    if blastdb:
        os.environ["BLASTDB"] = blastdb
    with tempfile.NamedTemporaryFile("w", suffix=".faa", delete=False) as fh:
        fh.write(textwrap.dedent(f">q\n{seq}\n"))
        q = fh.name
    out = _run(
        [
            "blastp",
            "-query",
            q,
            "-db",
            db,
            "-max_target_seqs",
            str(max_targets),
            "-num_threads",
            str(threads),
            "-outfmt",
            BLAST_OUTFMT,
        ]
    )
    reader = csv.DictReader(
        out.splitlines(),
        fieldnames=["sacc", "pident", "length", "evalue", "bitscore", "sscinames"],
        delimiter="\t",
    )
    return list(reader)


# ----------------------------- PDB cleaning ----------------------------------

def clean_pdb_for_docking(
    in_pdb: str | os.PathLike,
    out_pdb: str | os.PathLike,
    *,
    keep_chains: Sequence[str] | None = None,
    drop_resnames: Sequence[str] = ("HOH", "WAT", "NAG", "BMA", "MAN", "FUC", "GAL", "GLC", "FLC"),
    drop_hetatm: bool = True,
    drop_hydrogens: bool = False,
) -> None:
    """
    Create a receptor-clean PDB that keeps only protein ATOM records (optionally only
    specified chains), drops waters/glycans/ligands, and (optionally) hydrogens.

    Args:
        in_pdb: Input PDB path.
        out_pdb: Output cleaned PDB path.
        keep_chains: If provided, only these chain IDs are kept (e.g., ["A"]).
        drop_resnames: Residue names to remove (case-sensitive 3-letter codes).
        drop_hetatm: If True, drop all HETATM records (ligands, ions, etc.).
        drop_hydrogens: If True, drop hydrogen atoms (atom name starts with ' H').

    Notes:
        - Keeps only lines that start with "ATOM " (and "TER" when relevant) by default.
        - Chain ID is at column 22 (0-based slice [21:22]) in PDB format.
        - Residue name is at columns 18–20 (slice [17:20]).
        - Atom name is at columns 13–16 (slice [12:16]).
    """
    in_p = Path(in_pdb)
    out_p = Path(out_pdb)

    kept: list[str] = []
    chains_set = set(keep_chains) if keep_chains else None
    drop_set = set(drop_resnames)

    with open(in_p, "r") as fh:
        for line in fh:
            rec = line[0:6]
            if rec.startswith("ATOM  "):
                resname = line[17:20].strip()
                if resname in drop_set:
                    continue
                if chains_set:
                    chain_id = line[21:22]
                    if chain_id not in chains_set:
                        continue
                if drop_hydrogens:
                    atom_name = line[12:16]
                    if atom_name.strip().startswith("H"):
                        continue
                kept.append(line)
            elif rec.startswith("HETATM"):
                # drop all HETATM unless explicitly asked to keep them (default: drop)
                if not drop_hetatm:
                    # optional selective keep if not in drop list and chain matches
                    resname = line[17:20].strip()
                    if resname in drop_set:
                        continue
                    if chains_set:
                        chain_id = line[21:22]
                        if chain_id not in chains_set:
                            continue
                    if drop_hydrogens:
                        atom_name = line[12:16]
                        if atom_name.strip().startswith("H"):
                            continue
                    kept.append(line)
            elif rec.startswith("TER   "):
                # Keep TER only if we have kept something from that chain so far
                if kept:
                    kept.append(line)

    if not kept:
        raise RuntimeError("clean_pdb_for_docking produced an empty file; check input/filters.")

    out_p.write_text("".join(kept))

# ----------------------------- Meeko -----------------------------------------


def meeko_ligand(in_sdf: str | os.PathLike, out_pdbqt: str | os.PathLike) -> None:
    """Convert a small-molecule SDF to PDBQT using Meeko.
    Args:
        in_sdf: Input .sdf path.
        out_pdbqt: Output .pdbqt path.
    Example:
        >>> meeko_ligand("lig.sdf", "lig.pdbqt")
    """
    _run(["mk_prepare_ligand.py", "-i", str(in_sdf), "-o", str(out_pdbqt)])


def meeko_receptor(
    in_pdb: str | os.PathLike,
    out_pdbqt: str | os.PathLike,
    *,
    auto_clean: bool = True,
    keep_chains: Sequence[str] | None = None,
    drop_resnames: Sequence[str] = ("HOH", "WAT", "NAG", "BMA", "MAN", "FUC", "GAL", "GLC", "FLC"),
    drop_hydrogens: bool = False,
    fallback_obabel: bool = True,
    obabel_ph: float = 7.4,
) -> None:
    """
    Prepare receptor PDBQT using Meeko with optional auto-cleaning and Open Babel fallback.
    Writes <out>.clean.pdb alongside <out>.pdbqt so you can inspect inputs to Meeko/OBabel.
    """
    import shutil
    in_p = Path(in_pdb)
    out_qt = Path(out_pdbqt)
    out_qt.parent.mkdir(parents=True, exist_ok=True)

    # 0) Prepare a visible cleaned file (never deleted)
    meeko_input = in_p
    cleaned_path = out_qt#.with_suffix(".clean.pdb")
    if auto_clean:
        clean_pdb_for_docking(
            in_p,
            cleaned_path,
            keep_chains=keep_chains,
            drop_resnames=drop_resnames,
            drop_hetatm=True,
            drop_hydrogens=drop_hydrogens,
        )
        meeko_input = cleaned_path

    # 1) Try Meeko if present
    exe = shutil.which("mk_prepare_receptor.py")
    delete_arg = ",".join(sorted(set(drop_resnames)))
    meeko_cmd = [exe or "mk_prepare_receptor.py", "-i", str(meeko_input), "-o", str(out_qt), "--delete_residues", delete_arg]

    meeko_stdout = ""
    meeko_stderr = ""
    used = []

    if exe:
        try:
            cp = subprocess.run(meeko_cmd, check=True, text=True, capture_output=True)
            meeko_stdout, meeko_stderr = cp.stdout, cp.stderr
            used.append("meeko")
        except subprocess.CalledProcessError as e:
            meeko_stdout, meeko_stderr = e.stdout, e.stderr

    # 2) If Meeko not available or produced no file, optionally fall back to OBabel
    if not (out_qt.is_file() and out_qt.stat().st_size > 0):
        if fallback_obabel:
            try:
                _receptor_to_pdbqt_via_obabel(meeko_input, out_qt, ph=obabel_ph)
                used.append("obabel")
            except Exception as ob_e:
                # Compose a super explicit error
                listing = "\n".join(sorted(p.name for p in out_qt.parent.iterdir()))
                raise RuntimeError(
                    "Receptor prep failed.\n"
                    f"Tried: {', '.join(used) if used else 'none (Meeko missing)'}\n"
                    f"Cleaned PDB used: {meeko_input}\n"
                    f"Meeko cmd: {' '.join(meeko_cmd)}\n"
                    f"Meeko STDOUT:\n{meeko_stdout}\n\nMeeko STDERR:\n{meeko_stderr}\n"
                    f"Open Babel error: {ob_e}\n"
                    f"Folder contents:\n{listing}"
                )
        else:
            listing = "\n".join(sorted(p.name for p in out_qt.parent.iterdir()))
            raise RuntimeError(
                "Meeko receptor prep produced no output and fallback_obabel=False.\n"
                f"Cleaned PDB used: {meeko_input}\n"
                f"Meeko cmd: {' '.join(meeko_cmd)}\n"
                f"Meeko STDOUT:\n{meeko_stdout}\n\nMeeko STDERR:\n{meeko_stderr}\n"
                f"Folder contents:\n{listing}"
            )

    # 3) Final guard
    if not out_qt.is_file() or out_qt.stat().st_size == 0:
        listing = "\n".join(sorted(p.name for p in out_qt.parent.iterdir()))
        raise RuntimeError(
            f"Receptor PDBQT not created: {out_qt}\n"
            f"Cleaned PDB used: {meeko_input}\n"
            f"Folder contents:\n{listing}"
        )

    if out_qt.is_file() and out_qt.stat().st_size > 0:
        try:
            _strict_receptorize_pdbqt(out_qt, out_qt)
        except Exception as s_e:
            # not fatal; but if you want hard guarantee, raise instead
            pass
    return


def _strict_receptorize_pdbqt(in_pdbqt: os.PathLike, out_pdbqt: os.PathLike) -> None:
    """
    Make a Vina-rigid receptor PDBQT:
      - Keep only ATOM/HETATM (and an optional single TER before END)
      - Drop all other records, including REMARK/ROOT/BRANCH/TORSDOF/etc.
      - Normalize line endings to LF and ensure final 'END' line.
    """
    src = Path(in_pdbqt)
    dst = Path(out_pdbqt)
    if not src.is_file() or src.stat().st_size == 0:
        raise RuntimeError(f"Cannot receptorize: missing/empty PDBQT: {src}")

    atoms: list[str] = []
    saw_ter = False

    with src.open("rb") as fh:  # read bytes to normalize newlines & strip weird chars
        raw = fh.read().decode("utf-8", errors="ignore")
    for raw_line in raw.splitlines():
        line = raw_line.rstrip("\r")  # normalize CRLF -> LF
        if line.startswith("ATOM  ") or line.startswith("HETATM"):
            atoms.append(line + "\n")
            saw_ter = False
        elif line.startswith("TER   "):
            # keep at most one TER right after an atom block
            if atoms and not saw_ter:
                atoms.append("TER   \n")
                saw_ter = True
        # else: drop EVERYTHING (REMARK, ROOT, BRANCH, TORSDOF, MODEL, ENDMDL, etc.)

    if not atoms:
        raise RuntimeError("Receptorize: no ATOM/HETATM records survived. Check the input PDBQT/PDB prep.")

    # Ensure one TER before END for safety (some Vina builds are picky)
    if not saw_ter:
        atoms.append("TER   \n")

    # Write minimal receptor
    out_lines = atoms# + ["END\n"]
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    tmp.write_text("".join(out_lines))
    tmp.replace(dst)




def _receptor_to_pdbqt_via_obabel(pdb_in: os.PathLike, pdbqt_out: os.PathLike, *, ph: float = 7.4) -> None:
    """Internal: receptor conversion via Open Babel with basic protonation/charges, then sanitize."""
    import shutil
    obabel = shutil.which("obabel")
    if not obabel:
        raise RuntimeError("Open Babel (obabel) not found in PATH.")
    pdb_in = Path(pdb_in)
    pdbqt_out = Path(pdbqt_out)
    tmp_out = pdbqt_out.with_suffix(".obabel.pdbqt")  # raw (ligand-style) PDBQT

    cmd = [
        obabel,
        "-ipdb", str(pdb_in),
        "-opdbqt",
        "-O", str(tmp_out),
        "-p", str(ph),
        "--partialcharge", "gasteiger",
    ]
    cp = subprocess.run(cmd, text=True, capture_output=True)
    if cp.returncode != 0 or not tmp_out.exists() or tmp_out.stat().st_size == 0:
        raise RuntimeError(
            "Open Babel receptor prep failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"STDOUT:\n{cp.stdout}\n\nSTDERR:\n{cp.stderr}"
        )

    # Convert ligand-style to receptor-style
    _strict_receptorize_pdbqt(tmp_out, pdbqt_out)
    try:
        tmp_out.unlink(missing_ok=True)
    except Exception:
        pass


# ----------------------------- SDF generation --------------------------------


def smiles_to_sdf_3d(smiles: str, out_sdf: str | os.PathLike) -> None:
    """Build a 3D SDF from SMILES via RDKit, falling back to Open Babel if RDKit missing.
    Args:
        smiles: SMILES string.
        out_sdf: Output .sdf path.
    Notes:
        - RDKit path (preferred): Embed with ETKDG + UFF optimize.
        - Fallback requires `obabel` in PATH: `obabel -:'<smiles>' -O out.sdf --gen3d`
    Example:
        >>> smiles_to_sdf_3d("c1ccccc1", "benzene.sdf")
    """
    try:
        from rdkit import Chem  # type: ignore
        from rdkit.Chem import AllChem  # type: ignore
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        AllChem.UFFOptimizeMolecule(mol)
        w = Chem.SDWriter(str(out_sdf))
        w.write(mol)
        w.close()
    except Exception:
        if not shutil.which("obabel"):
            raise RuntimeError("Need RDKit or Open Babel (obabel) to build 3D SDF.")
        _run(["obabel", f"-:{smiles}", "-O", str(out_sdf), "--gen3d"])


# ----------------------------- AutoDock Vina ---------------------------------

def vina_score_only(
    receptor_pdbqt: str | os.PathLike,
    ligand_pdbqt: str | os.PathLike,
    config_txt: str | os.PathLike | None = None,
) -> Tuple[float, str]:
    """Run Vina in score-only mode and return (best_affinity_kcal_per_mol, raw_stdout).
    Args:
        receptor_pdbqt: Receptor .pdbqt path.
        ligand_pdbqt: Ligand .pdbqt path.
        config_txt: Optional Vina config file (box, etc.).
    Notes:
        - Affinity is in kcal/mol (more negative is better).
        - If parsing fails, best affinity is NaN but full stdout is returned.
    Example:
        >>> best, log = vina_score_only("receptor.pdbqt", "ligand.pdbqt", "config.txt")
        >>> print(best)  # -7.4 (example)
    """
    cmd = ["vina", "--receptor", str(receptor_pdbqt), "--ligand", str(ligand_pdbqt), "--score_only"]
    if config_txt:
        cmd += ["--config", str(config_txt)]
    out = _run(cmd)
    m = re.search(r"Affinity:\s*(-?\d+(?:\.\d+)?)", out)
    best = float(m.group(1)) if m else float("nan")
    return best, out


def _validate_receptor_pdbqt_for_vina(path: os.PathLike) -> None:
    p = Path(path)
    if not p.is_file() or p.stat().st_size == 0:
        raise FileNotFoundError(f"Receptor PDBQT missing or empty: {p}")

    bad = []
    with p.open("rb") as fh:
        lines = fh.read().decode("utf-8", errors="ignore").splitlines()
    allowed = ("ATOM  ", "HETATM", "TER", "END")
    for i, ln in enumerate(lines, 1):
        if ln.startswith(allowed):
            continue
        # allow blank lines
        if ln.strip() == "":
            continue
        bad.append((i, ln))
        if len(bad) >= 3:
            break

    if bad:
        tail = "\n".join(f"{i}: {ln}" for i, ln in bad)
        tail_end = "\n".join(f"{len(lines)-k+1}: {lines[-k]}" for k in range(min(10, len(lines)), 0, -1))
        raise RuntimeError(
            "Receptor PDBQT contains disallowed tags.\n"
            f"First offending lines:\n{tail}\n\n"
            f"File tail:\n{tail_end}"
        )
    # also assert exact END line at the end
    #if lines[-1].strip() != "END":
    #    raise RuntimeError(f"Receptor PDBQT does not end with 'END'. Last line was: {lines[-1]!r}")


def vina_dock(
    receptor_pdbqt: str | os.PathLike,
    ligand_pdbqt: str | os.PathLike,
    out_pdbqt: str | os.PathLike,
    *,
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    size: Tuple[float, float, float] = (20.0, 20.0, 20.0),
    num_modes: int = 9,
    exhaustiveness: int = 8,
) -> str:
    """Dock with Vina and write poses to `out_pdbqt`. Returns Vina stdout.
    Args:
        receptor_pdbqt: Receptor .pdbqt.
        ligand_pdbqt: Ligand .pdbqt.
        out_pdbqt: Output poses .pdbqt.
        center: (cx, cy, cz) Å box center.
        size: (sx, sy, sz) Å box size.
        num_modes: Number of poses to output.
        exhaustiveness: Search thoroughness (time ↑ with value).
    Example:
        >>> log = vina_dock("receptor.pdbqt","ligand.pdbqt","poses.pdbqt",
        ...                 center=(10,12,5), size=(22,22,22), num_modes=10)
    """
    r = Path(receptor_pdbqt); l = Path(ligand_pdbqt); o = Path(out_pdbqt)
    if not r.is_file() or r.stat().st_size == 0:
        raise FileNotFoundError(f"Receptor PDBQT missing or empty: {r}")
    if not l.is_file() or l.stat().st_size == 0:
        raise FileNotFoundError(f"Ligand PDBQT missing or empty: {l}")
    o.parent.mkdir(parents=True, exist_ok=True)

    _validate_receptor_pdbqt_for_vina(receptor_pdbqt)
    
    cx, cy, cz = map(str, center)
    sx, sy, sz = map(str, size)
    cmd = [
        "vina",
        "--receptor",
        str(receptor_pdbqt),
        "--ligand",
        str(ligand_pdbqt),
        "--center_x",
        cx,
        "--center_y",
        cy,
        "--center_z",
        cz,
        "--size_x",
        sx,
        "--size_y",
        sy,
        "--size_z",
        sz,
        "--num_modes",
        str(num_modes),
        "--exhaustiveness",
        str(exhaustiveness),
        "--out",
        str(out_pdbqt),
    ]
    return _run(cmd)

