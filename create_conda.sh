#!/bin/bash

set -e  # Exit immediately if a command fails
set -o pipefail  # Ensure pipeline errors are also caught

if ! conda info --envs | grep -q "^milvus_env "; then
  if ! grep -q "conda initialize" ~/.bashrc; then
    conda init bash
  fi
  conda init bash
else
  echo "Environment 'milvus_env' already exists. Skipping creation."
fi
conda init bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate milvus_env
pip install -U pip
if ! pip show uv > /dev/null 2>&1; then
  pip install uv
fi
pip install -U pip
pip install uv

# Step 3: Install packages using uv
uv pip install -r requirements.txt
uv pip install -e .
# Step 4: Common dev dependencies (optional)
# These dependencies are useful for development tasks such as building packages,
# managing documentation, and handling advanced Python features. They are optional
# and should be installed if you plan to contribute to the project or need these tools.
# Step 4: Common dev dependencies (optional)
uv pip install -U \
  setuptools \
  wheel \
  twine \
  mkdocs-material \
  python-multipart \
  pydantic \
  pydantic-settings \
  structlog \
  tenacity


uv pip install -r dev-requirements.txt

# On Debian-based systems (e.g., Ubuntu), you can install `exuberant-ctags` using:
# sudo apt install exuberant-ctags
# On other systems, you might need to install `universal-ctags` if available.
# Install ctags
# Step 5: Optional machine check
# sudo apt install exuberant-ctags  # or `universal-ctags` if available
# Generate ctags for Python files
if ! command -v ctags &> /dev/null; then
  echo "Error: ctags is not installed. Please install it and try again."
  exit 1
fi
ctags -R --languages=Python --python-kinds=-iv -f .tags .
# Generate ctags for Python files
ctags -R --languages=Python --python-kinds=-iv -f .tags .

# Install ctags for python files
uv pip install -U ctags
# Generate ctags for Python files
ctags -R --languages=Python --python-kinds=-iv -f .tags .

# Install ctags for python files
uv pip install -U ctags
# Generate ctags for Python files
ctags -R --languages=Python --python-kinds=-iv -f .tags .

# Install ctags for python files
echo "Tags generated in .tags file"

# Install ctags for python files
echo "Tags generated in .tags file"
echo "Detected CPU Architecture: $(python -c 'import platform; print(platform.machine())')"
# Step 6: Optional machine check
echo "CPU Arch: $(python -c 'import platform; print(platform.machine())')"