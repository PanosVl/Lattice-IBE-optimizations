#!/bin/bash
# Script to run IBE with timing and save output

OUTPUT_DIR="./outputs"
OUTPUT_FILE="${OUTPUT_DIR}/timing_N${1:-512}_q${2:-default}.txt"

echo "Running Lattice-IBE with timing instrumentation..."
echo "Parameters: N=$1, q=$2"
echo "=================================="

# Create outputs directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Run the program and save output
./IBE 2>&1 | tee "$OUTPUT_FILE"

echo ""
echo "Output saved to $OUTPUT_FILE"
