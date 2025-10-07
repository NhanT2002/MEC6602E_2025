#!/usr/bin/env bash
set -eu

usage() {
  cat <<EOF
Usage: $(basename "$0") [-e EXECUTABLE] [-i INPUT] -a ALGORITHM -c CFL [CFL ...] [-d OUTDIR]

Runs the specified executable with a modified copy of the input file for each CFL value.

Options:
  -e EXECUTABLE  Path to the wave executable (default: ./bin/wave)
  -i INPUT       Path to the input file (default: ./input.txt)
  -a ALGORITHM   Algorithm name to set in the input file (required)
  -c CFL         One or more CFL values (e.g. 0.5 0.75 1.0) (at least one required)
  -d OUTDIR      Directory where outputs will be left (default: ./output)

Example:
  $(basename "$0") -e ../wave/bin/wave -i Wave/input.txt -a tremblayTran -c 0.5 0.75 1.0 -d Wave/output

EOF
}

# defaults
EXECUTABLE="bin/wave"
INPUT_FILE="input.txt"
OUTDIR="output/"

if [ $# -eq 0 ]; then
  usage
  exit 1
fi

ALGO=""
CFLS=()

while (( "$#" )); do
  case "$1" in
    -e)
      EXECUTABLE="$2"; shift 2;;
    -i)
      INPUT_FILE="$2"; shift 2;;
    -a)
      ALGO="$2"; shift 2;;
    -c)
      shift
      # collect remaining non-option args as CFLs until next option or end
      while [[ "$#" -gt 0 && ! "$1" =~ ^- ]]; do
        CFLS+=("$1"); shift
      done
      ;;
    -d)
      OUTDIR="$2"; shift 2;;
    -h|--help)
      usage; exit 0;;
    *)
      echo "Unknown arg: $1" >&2; usage; exit 2;;
  esac
done

if [ -z "$ALGO" ]; then
  echo "Error: algorithm (-a) is required" >&2; usage; exit 2
fi
if [ ${#CFLS[@]} -eq 0 ]; then
  echo "Error: at least one CFL (-c) is required" >&2; usage; exit 2
fi

if [ ! -x "$EXECUTABLE" ]; then
  echo "Error: executable '$EXECUTABLE' not found or not executable" >&2; exit 3
fi
if [ ! -f "$INPUT_FILE" ]; then
  echo "Error: input file '$INPUT_FILE' not found" >&2; exit 3
fi

mkdir -p "$OUTDIR"

sanitize_cfl() {
  local raw="$1"
  # if integer with trailing .0 remove it
  if [[ "$raw" =~ ^([0-9]+)\.0+$ ]]; then
    echo "${BASH_REMATCH[1]}"
    return
  fi
  # replace decimal point with hyphen for filenames
  echo "${raw//./-}"
}

run_one() {
  local cfl="$1"
  local scfl
  scfl=$(sanitize_cfl "$cfl")
  local outname="${ALGO}_CFL${scfl}_output.txt"
  local tmpfile
  tmpfile=$(mktemp --suffix="_${ALGO}_CFL${scfl}.txt")

  # Update the input file: CFL, algorithm, output_filename
  # Use sed to replace the lines if present; otherwise append them
  sed -E \
    -e "s/^[[:space:]]*CFL[[:space:]]*=.*/CFL = ${cfl}/" \
    -e "s/^[[:space:]]*algorithm[[:space:]]*=.*/algorithm = ${ALGO}/" \
    -e "s/^[[:space:]]*output_filename[[:space:]]*=.*/output_filename = ${outname}/" \
    "$INPUT_FILE" > "$tmpfile"

  # If any of the three keys were not present, append them at the end
  grep -Eq "^[[:space:]]*CFL[[:space:]]*=" "$tmpfile" || echo "CFL = ${cfl}" >> "$tmpfile"
  grep -Eq "^[[:space:]]*algorithm[[:space:]]*=" "$tmpfile" || echo "algorithm = ${ALGO}" >> "$tmpfile"
  grep -Eq "^[[:space:]]*output_filename[[:space:]]*=" "$tmpfile" || echo "output_filename = ${outname}" >> "$tmpfile"

  echo "Running: $EXECUTABLE $tmpfile -> output: $OUTDIR/$outname"
  # run the executable with the modified input file
  if "$EXECUTABLE" "$tmpfile"; then
    # move output file into OUTDIR if it's created in cwd or current input dir
    # The executable should write to the filename specified in the input file.
    if [ -f "$outname" ]; then
      mv -f "$outname" "$OUTDIR/"
    else
      # try to find file in same directory as tmpfile
      local maybe="$(dirname "$tmpfile")/$outname"
      if [ -f "$maybe" ]; then
        mv -f "$maybe" "$OUTDIR/"
      fi
    fi
  else
    echo "Warning: executable returned non-zero for CFL=$cfl" >&2
  fi

  rm -f "$tmpfile"
}

for c in "${CFLS[@]}"; do
  run_one "$c"
done

echo "All runs finished. Outputs (if produced) are in: $OUTDIR"
