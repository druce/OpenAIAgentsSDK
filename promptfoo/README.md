# Promptfoo Testing Framework

This directory contains prompt evaluation tests and configurations for our AI agent workflow components.

## Installation

Install promptfoo globally:

on Mac:

```bash
brew install promptfoo
```

on Linux:

```bash
sudo apt install promptfoo
```

Or use with npx (no installation required):

```bash
npx promptfoo --version
```

## Directory Structure

```bash
promptfoo/
├── README.md                    # This file
├── prompts/                     # Prompt templates and configurations
│   ├── headline_classifier_chat.json
│   ├── headline_classifier_v1.yaml
│   └── [future prompt files...]
├── tests/                       # Test configuration files
│   ├── headline_classifier.yaml
│   └── [future test configs...]
└── results/                     # Test results (auto-generated)
    ├── headline_classifier/
    │   ├── promptfoo-results.json
    │   └── [timestamped results...]
    └── [future results directories...]
```

## Current File Structure

### Prompts Directory

- **`headline_classifier_chat.json`** - Chat-based prompt for AI headline classification
- **`headline_classifier_v1.yaml`** - YAML configuration for headline classifier prompt

### Tests Directory

- **`headline_classifier.yaml`** - Test configuration for headline classification component

## Running Evaluations

### Basic Usage

Run a specific test configuration:

```bash
promptfoo eval -c headline_classifier.yaml
```

### Run with Verbose Output

```bash
promptfoo eval -c headline_classifier.yaml --verbose
```

## Viewing Results

### Interactive Web UI

View the most recent results (with browser auto-open):

```bash
promptfoo view -y
```

### View Specific Results

```bash
# View specific result file
promptfoo view -y results/headline_classifier/results.json

# View multiple results
promptfoo view -y "results/headline_classifier/results.json,results/topic_extractor/results.json"
```

### Export Results

Generate HTML report:

```bash
promptfoo view -y --output results/report.html results/headline_classifier/results.json
```

Export as CSV:

```bash
promptfoo export results/headline_classifier/results.json --format csv
```

### Batch Testing (Future)

When you have multiple components, create a master script:

```bash
#!/bin/bash
# run-all-tests.sh

echo "Running headline classifier tests..."
promptfoo eval -c headline_classifier.yaml -o results/headline_classifier/

echo "Running topic extractor tests..."
promptfoo eval -c topic_extractor.yaml -o results/topic_extractor/

echo "Running story summarizer tests..."
promptfoo eval -c story_summarizer.yaml -o results/story_summarizer/

echo "Opening combined results..."
promptfoo view -y "results/headline_classifier/results.json,results/topic_extractor/results.json,results/story_summarizer/results.json"
```

### Organization

- prompts in `prompts/` directory
- make .json file from system prompt and user prompt
- version all prompt files
- tests in `tests/` directory, versioned
- generate results in `results/` directory based on top level yaml config file
- Add `results/` to `.gitignore` (results are ephemeral)

### Debug Commands

```bash
# Check promptfoo version
promptfoo --version

# Validate configuration file
promptfoo eval -c tests/headline_classifier.yaml --dry-run

# Run with maximum verbosity
promptfoo eval -c tests/headline_classifier.yaml --verbose --log-level debug
```
