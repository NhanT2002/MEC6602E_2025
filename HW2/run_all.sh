#!/usr/bin/env bash
set -euo pipefail

# Run from HW2 directory
cd "$(dirname "$0")" || exit 1

sizes=(256 512)
mach_values=("0.8")
alpha_values=("0.0" "1.25")

for M in "${mach_values[@]}"; do
  case "$M" in
    0.8) Mtag="M08" ;;
    0.5) Mtag="M05" ;;
    *) Mtag="M$(echo "$M" | tr -d '.')" ;;
  esac

  for A in "${alpha_values[@]}"; do
    case "$A" in
      0.0) Atag="A0" ;;
      1.25) Atag="A125" ;;
      *) Atag="A$(echo "$A" | tr -d '.')" ;;
    esac

    for s in "${sizes[@]}"; do
      meshfile="mesh/naca0012_${s}x${s}.xyz"
      output_file="output_${Mtag}_${Atag}_${s}.q"
      checkpoint_file="residual_history_${Mtag}_${Atag}_${s}.txt"
      input_file="input_${Mtag}_${Atag}_${s}.txt"

      cat > "$input_file" <<EOF
num_threads = 4
mesh_file = $meshfile
Mach = $M
alpha = $A
p_inf = 1E5
T_inf = 300.0
multigrid = 0
CFL_number = 7.2
residual_smoothing = 1
k2 = 2.0
k4 = 2.0
it_max = 100000
output_file = $output_file
checkpoint_file = $checkpoint_file
EOF

      echo "============================================"
      echo "Running: bin/euler_solver $input_file"
      echo " mesh: $meshfile | Mach: $M | alpha: $A | output: $output_file"
      # Execute the solver; comment the next line out to do a dry-run
      ./bin/euler_solver "$input_file"
      echo "Finished run for ${Mtag} ${Atag} ${s}"
    done
  done
done

echo "All runs finished."
