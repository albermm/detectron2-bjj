#!/bin/bash

# Exit on error
set -e

# Create new directories
mkdir -p api config/model_configs data/inputs models scripts tests utils

# Move API-related files
mv -f app.py lambda_function.py API_definition_BJJ_Processing.yaml api/ 2>/dev/null || true

# Move configuration files
mv -f config.yaml config/ 2>/dev/null || true
mv -f model_configs/* config/model_configs/ 2>/dev/null || true

# Move input data
mv -f inputs/* data/inputs/ 2>/dev/null || true

# Move model file
mv -f trained_model.joblib models/ 2>/dev/null || true

# Move scripts
mv -f download-models.sh scripts/ 2>/dev/null || true

# If dynamodb_scripts exists, move its contents; otherwise, create it
if [ -d "dynamodb_scripts" ]; then
    mv dynamodb_scripts scripts/
else
    mkdir -p scripts/dynamodb_scripts
fi

# Move test files
mv -f test_*.py tests/ 2>/dev/null || true

# Move utility files
mv -f utils/* utils/ 2>/dev/null || true

# Create empty README and .gitignore if they don't exist
touch README.md
if [ ! -f ".gitignore" ]; then
    echo "__pycache__" > .gitignore
fi

# Clean up empty directories
find . -type d -empty -delete

echo "Project restructured successfully!"



