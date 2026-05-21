# DMOs Pipeline

Python pipeline for processing bilateral foot-mounted IMU recordings and preparing stride-level outputs for Digital Mobility Outcome extraction.

The pipeline includes:

- gaitmap-based gait sequence detection
- stride segmentation
- gait event detection
- temporal and spatial gait parameter extraction
- recording and wearing-time estimation
- event and stride-level quality control

## Current workflow

1. Run the gaitmap-based processing pipeline.
2. Save gait events, stride-level parameters, and logs.
3. Run quality control on the exported events and parameters.

A complete example is available in:

```text
examples/run_gaitmap_and_quality_check.py

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/YOUR_USERNAME/DMOs_pipeline.git
cd DMOs_pipeline
pip install -r requirements.txt