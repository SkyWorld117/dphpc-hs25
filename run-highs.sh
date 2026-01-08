#!/usr/bin/env bash

# Script to run all MPS files from HiGHS check instances
# and generate a summary of successful and failed runs

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Arrays to track results
declare -a successful_files
declare -a failed_files

# MPS files directory
MPS_DIR="HiGHS/check/instances"
EXECUTABLE="build/cuopt_cli"

# Check if executable exists
if [ ! -f "$EXECUTABLE" ]; then
    echo -e "${RED}Error: $EXECUTABLE not found${NC}"
    echo "Please build the project first using ./build.sh"
    exit 1
fi

# Check if directory exists
if [ ! -d "$MPS_DIR" ]; then
    echo -e "${RED}Error: Directory $MPS_DIR not found${NC}"
    exit 1
fi

echo "========================================="
echo "Running MPS files from $MPS_DIR"
echo "========================================="
echo ""

# Find all .mps files
mps_files=("$MPS_DIR"/*.mps)
total_files=${#mps_files[@]}
current=0

# Run each MPS file
for mps_file in "${mps_files[@]}"; do
    if [ -f "$mps_file" ]; then
        current=$((current + 1))
        filename=$(basename "$mps_file")
        
        echo -e "${YELLOW}[$current/$total_files]${NC} Processing: $filename"
        
        # Run the command and capture output
        if "$EXECUTABLE" --gpu "$mps_file" > /dev/null 2>&1; then
            echo -e "${GREEN}✓${NC} Success: $filename"
            successful_files+=("$filename")
        else
            echo -e "${RED}✗${NC} Failed: $filename"
            failed_files+=("$filename")
        fi
        echo ""
    fi
done

# Generate summary
echo "========================================="
echo "SUMMARY"
echo "========================================="
echo ""
echo "Total files processed: $total_files"
echo -e "${GREEN}Successful: ${#successful_files[@]}${NC}"
echo -e "${RED}Failed: ${#failed_files[@]}${NC}"
echo ""

# List successful files
if [ ${#successful_files[@]} -gt 0 ]; then
    echo -e "${GREEN}Successful files:${NC}"
    for file in "${successful_files[@]}"; do
        echo "  ✓ $file"
    done
    echo ""
fi

# List failed files
if [ ${#failed_files[@]} -gt 0 ]; then
    echo -e "${RED}Failed files:${NC}"
    for file in "${failed_files[@]}"; do
        echo "  ✗ $file"
    done
    echo ""
fi

# Exit with appropriate code
if [ ${#failed_files[@]} -gt 0 ]; then
    exit 1
else
    exit 0
fi