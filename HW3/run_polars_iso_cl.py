import os
import re
import subprocess
import shutil
import sys
import numpy as np

# Configuration
EXE = os.path.join("..", "HW2_iso_cl", "bin", "euler_solver")
mesh = "sc20712"
TEMPLATE = f"input_iso_cl.txt"  # template input file in this directory
OUTDIR = f"solver_outputs_iso_cl/{mesh}"
INPUTDIR = f"input_cases_iso_cl/{mesh}"
# Angles of attack (degrees) and Mach numbers to sweep. Edit as needed.
ANGLES = [0.0]
# MACHES = np.array([0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8])
MACHES = np.array([0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.8, 0.81, 0.82, 0.83, 0.84, 0.85])
QUIET = True  # when True do not print solver stdout/stderr to console

os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(INPUTDIR, exist_ok=True)

if not os.path.exists(EXE):
    found = shutil.which(os.path.basename(EXE))
    if found:
        EXE = found
    else:
        raise FileNotFoundError(f"Executable not found: {EXE}")

if not os.path.exists(TEMPLATE):
    raise FileNotFoundError(f"Template input file not found: {TEMPLATE}")


template_text = open(TEMPLATE, "r").read()

failed = []
for mach in MACHES:
    mach_str = f"{mach:.3f}"
    # per-mach grouping directory
    mach_dir = os.path.join(OUTDIR, f"mach_{mach_str}")
    os.makedirs(mach_dir, exist_ok=True)

    # per-mach input dir
    mach_input_dir = os.path.join(INPUTDIR, f"mach_{mach_str}")
    os.makedirs(mach_input_dir, exist_ok=True)

    for a in ANGLES:
        aoa_str = f"{a:.3f}"
        in_basename = f"input_aoa_{aoa_str}.txt"
        in_name = os.path.join(mach_input_dir, in_basename)

        # per-run directory under mach_dir to hold solver-generated files
        run_dir = os.path.join(mach_dir, f"aoa_{aoa_str}")
        os.makedirs(run_dir, exist_ok=True)

        # Replace the 'Mach', 'alpha', 'output_file', and 'checkpoint_file' parameters explicitly (case-insensitive).
        # If a parameter is not present, append it to the end of the file.
        lines = template_text.splitlines()
        found_mesh = False
        found_mach = False
        found_alpha = False
        found_output = False
        found_checkpoint = False

        mesh_pattern = re.compile(r"^\s*(mesh_file)\s*[:=]\s*(\S+)(\s*(#.*)?)$", re.IGNORECASE)
        mach_pattern = re.compile(r"^\s*(Mach)\s*[:=]\s*([+-]?\d+(?:\.\d+)?)(\s*(#.*)?)$", re.IGNORECASE)
        alpha_pattern = re.compile(r"^\s*(alpha)\s*[:=]\s*([+-]?\d+(?:\.\d+)?)(\s*(#.*)?)$", re.IGNORECASE)
        output_pattern = re.compile(r"^\s*(output_file)\s*[:=]\s*(\S+)(\s*(#.*)?)$", re.IGNORECASE)
        checkpoint_pattern = re.compile(r"^\s*(checkpoint_file)\s*[:=]\s*(\S+)(\s*(#.*)?)$", re.IGNORECASE)

        new_lines = []
        for ln in lines:
            m = mesh_pattern.match(ln)
            if m:
                trailing = m.group(3) or ""
                new_lines.append(f"mesh_file = ../HW2_iso_cl/mesh/{mesh}.xyz{trailing}")
                found_mesh = True
                continue

            m = mach_pattern.match(ln)
            if m:
                trailing = m.group(3) or ""
                new_lines.append(f"Mach = {mach_str}{trailing}")
                found_mach = True
                continue

            m = alpha_pattern.match(ln)
            if m:
                trailing = m.group(3) or ""
                new_lines.append(f"alpha = {aoa_str}{trailing}")
                found_alpha = True
                continue

            m = output_pattern.match(ln)
            if m:
                orig = m.group(2)
                trailing = m.group(3) or ""
                _, ext = os.path.splitext(orig)
                if not ext:
                    ext = ".q"
                new_name = os.path.join(run_dir, f"output_aoa_{aoa_str}")
                new_lines.append(f"output_file = {new_name}{trailing}")
                found_output = True
                continue

            m = checkpoint_pattern.match(ln)
            if m:
                orig = m.group(2)
                trailing = m.group(3) or ""
                _, ext = os.path.splitext(orig)
                if not ext:
                    ext = ".txt"
                new_name = os.path.join(run_dir, f"residual_history_aoa_{aoa_str}")
                new_lines.append(f"checkpoint_file = {new_name}{trailing}")
                found_checkpoint = True
                continue

            new_lines.append(ln)

        # Append missing parameters
        if not found_mach:
            if new_lines and new_lines[-1].strip() != "":
                new_lines.append("")
            new_lines.append(f"Mach = {mach_str}")
        if not found_alpha:
            if new_lines and new_lines[-1].strip() != "":
                new_lines.append("")
            new_lines.append(f"alpha = {aoa_str}")
        if not found_output:
            if new_lines and new_lines[-1].strip() != "":
                new_lines.append("")
            new_lines.append(f"output_file = {os.path.join(run_dir, f'output_aoa_{aoa_str}.q')}")
        if not found_checkpoint:
            if new_lines and new_lines[-1].strip() != "":
                new_lines.append("")
            new_lines.append(f"checkpoint_file = {os.path.join(run_dir, f'residual_history_aoa_{aoa_str}.txt')}")

        content = "\n".join(new_lines) + "\n"

        with open(in_name, "w") as f:
            f.write(content)

        # run solver quietly and save stdout/stderr per run
        proc = subprocess.run([EXE, in_name], capture_output=True, text=True)

        out_path = os.path.join(run_dir, f"out_aoa_{aoa_str}.txt")
        err_path = os.path.join(run_dir, f"err_aoa_{aoa_str}.txt")
        with open(out_path, "w") as fo:
            fo.write(proc.stdout or "")
        with open(err_path, "w") as fe:
            fe.write(proc.stderr or "")

        # record return code
        with open(os.path.join(run_dir, f"ret_aoa_{aoa_str}.txt"), "w") as fr:
            fr.write(str(proc.returncode))

        if proc.returncode != 0:
            failed.append(((mach, a), proc.returncode))

# If desired, you can inspect files under `solver_outputs/` for per-angle stdout/stderr/ret
if failed and not QUIET:
    print("Some runs failed:")
    for a, rc in failed:
        print(f" AoA={a:.3f}: return code {rc}")