#!/bin/bash

# This script downloads ONLY the metadata (.lbl, .cat, .txt, .tab) 
# and folder structure from the SELENE PDS3 archive.
# It explicitly excludes the heavy image and document files (.img, .jpg, .pdf)

echo "Starting download of SELENE metadata..."

lftp <<EOF
set net:max-retries 10
set net:timeout 30
set xfer:clobber off

# We use -X (exclude-glob) to skip massive files while keeping the directory structure intact.
# The metadata files like .lbl, .txt, .tab, .cat will still be downloaded.
mirror --continue --verbose -X "*.img" -X "*.IMG" -X "*.jpg" -X "*.JPG" -X "*.png" -X "*.PNG" -X "*.pdf" -X "*.PDF" -X "*.bsp" -X "*.bpc" -X "*.bc" -X "*.tsc" https://data.darts.isas.jaxa.jp/pub/pds3/sln-l-tc-4-dtm-ortho-v3.0/ ./selene_metadata

bye
EOF

echo "Metadata download complete!"
