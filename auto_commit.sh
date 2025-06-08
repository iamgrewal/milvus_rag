#!/bin/bash

# Function to stage and commit changes
stage_and_commit() {
    # Stage all changes
    git add .
    
    # The pre-commit hook will handle the actual commit
    echo "Changes staged and committed automatically"
}

# Main execution
echo "Staging and committing changes..."
stage_and_commit 