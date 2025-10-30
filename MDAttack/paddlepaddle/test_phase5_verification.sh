#!/bin/bash

echo "==========================================="
echo "Phase 5 Implementation Verification"
echo "==========================================="

# Check all required files exist
echo "Checking Phase 5 files..."
files=(
    "paddlepaddle_impl/attacks/__init__.py"
    "paddlepaddle_impl/attacks/attack_handler.py"
    "paddlepaddle_impl/main.py"
    "paddlepaddle_impl/collect_results.py"
)

all_exist=true
for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "✓ $file exists"
    else
        echo "✗ $file missing"
        all_exist=false
    fi
done

if [ "$all_exist" = true ]; then
    echo ""
    echo "All Phase 5 files present!"
    echo ""
    echo "Running Python verification tests..."
    python paddlepaddle_impl/test_phase5_verification.py
else
    echo ""
    echo "Some files are missing!"
    exit 1
fi