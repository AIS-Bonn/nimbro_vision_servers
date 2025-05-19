#!/bin/bash

SEARCH_DIR=/cache/sam2_realtime

mkdir -p "$SEARCH_DIR"

# Define the list of URLs
urls=(  
  "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"
  "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt"
  "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt"
  "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
)

# Loop through the URLs array
for url in "${urls[@]}"; do
  # Extract the filename from the URL
  filename=$(basename "$url")

  # Construct the full path for the file
  filepath="$SEARCH_DIR/$filename"

  # Check if the file already exists
  if [ ! -f "$filepath" ]; then
    # File doesn't exist, download it
    echo "Downloading $filename to $SEARCH_DIR..."
    wget -P "$SEARCH_DIR" "$url"
  else
    echo "$filename already exists in $SEARCH_DIR, skipping..."
  fi
done

exec "$@"