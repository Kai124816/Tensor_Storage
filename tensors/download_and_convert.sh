#!/bin/bash

# Exit immediately on error
set -e

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <link> <data_type>"
    echo "Example: $0 http://example.com/tensor.tns.gz double"
    exit 1
fi

LINK=$1
DTYPE=$2

# ============================
# Step 1: Download the file
# ============================
echo "Downloading tensor file from $LINK..."
wget -q "$LINK" -O tensor.tns.gz

# ============================
# Step 2: Unzip the .gz file
# ============================
echo "Unzipping tensor.tns.gz..."
gunzip -f tensor.tns.gz
# After this, file is tensor.tns
TENSOR_FILE="tensor.tns"

# ============================
# Step 3: Clean the tensor file
# ============================
echo "Compiling clean_tns.cc..."
g++ -std=c++17 -O2 -o clean_tns clean_tns.cc

echo "Running clean_tns on $TENSOR_FILE..."
./clean_tns "$TENSOR_FILE"

# After clean_tns runs, assume output is "tensor_clean.txt"
CLEAN_FILE="$TENSOR_FILE"

# Remove the helper
rm clean_tns

# ============================
# Step 4: Convert to binary
# ============================
echo "Compiling txt2bin.cc..."
g++ -O2 -o txt2bin txt2bin.cc

echo "Converting $CLEAN_FILE to binary (tensor.bin) with type $DTYPE..."
./txt2bin "$CLEAN_FILE" tensor.bin "$DTYPE"

# Cleanup
rm txt2bin
rm "$TENSOR_FILE"

echo "Done! Binary tensor stored in tensor.bin"
