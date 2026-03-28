#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Load .env variables needed for conda and paths
if [ -f "$SCRIPT_DIR/.env" ]; then
    export HOMEDIR=$(grep '^HOMEDIR=' "$SCRIPT_DIR/.env" | cut -d'=' -f2- | sed "s/^['\"]//;s/['\"]$//")
    export ENV_NAME=$(grep '^ENV_NAME=' "$SCRIPT_DIR/.env" | cut -d'=' -f2- | sed "s/^['\"]//;s/['\"]$//")
    export CONDA_PREFIX=$(grep '^CONDA_PREFIX=' "$SCRIPT_DIR/.env" | cut -d'=' -f2- | sed "s/^['\"]//;s/['\"]$//")
    export FIREFOX_PROFILE_PATH=$(grep '^FIREFOX_PROFILE_PATH=' "$SCRIPT_DIR/.env" | cut -d'=' -f2- | sed "s/^['\"]//;s/['\"]$//")
fi

cd "${HOMEDIR:?HOMEDIR environment variable not set}"
echo "Working directory: $HOMEDIR"
echo "Using Firefox profile: $FIREFOX_PROFILE_PATH"

echo "Activating conda environment: $ENV_NAME"
source "$CONDA_PREFIX/etc/profile.d/conda.sh"
conda activate "$ENV_NAME" || {
    echo "Failed to activate conda environment: $ENV_NAME"
    exit 1
}

# Run under caffeinate on Mac to prevent sleep during long workflows
echo "Running run_agent.py $*"
caffeinate -dimsu python "$HOMEDIR/run_agent.py" "$@" > "$HOMEDIR/run_agent.out" 2>&1
