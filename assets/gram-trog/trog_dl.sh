#!/bin/bash

# Define the URL
URL="https://api.github.com/repos/levante-framework/core-tasks/contents/assets/TROG/original"

# Create the images directory if it doesn't exist
mkdir -p images

# Download the JSON file
curl -s "$URL" -o data.json

# Extract the download URLs and download the images
grep -o '"download_url": *"[^"]*"' data.json | sed 's/"download_url": *"\([^"]*\)"/\1/' | while read -r download_url; do
    wget -P images "$download_url"
done

# Clean up
rm data.json

echo "TROG download completed!"