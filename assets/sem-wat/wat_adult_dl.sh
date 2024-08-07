#!/bin/bash

# Define the base URL
BASE_URL="http://w3.usf.edu/FreeAssociation/AppendixA/Cue_Target_Pairs."

# Define the array
declare -a array=("A-B" "C" "D-F" "G-K" "L-O" "P-R" "S" "T-Z")

# Create the directory if it doesn't exist
mkdir -p adult

# Loop through the array and download each file
for element in "${array[@]}"; do
    url="${BASE_URL}${element}"
    wget -P adult "$url"
done

echo "WAT adult download completed!"
