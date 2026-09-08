#!/usr/bin/env python3
"""
Command-Line Interface for MEEGFlow Preprocessing Pipeline.

This module provides a command-line interface for running the MEEG preprocessing
pipeline on datasets. It supports both BIDS-formatted datasets and custom glob
pattern matching for flexible file discovery.

Main Functions
--------------
main() : Entry point for the CLI
    Parses command-line arguments, configures logging, initializes the pipeline
    with the specified reader, and executes preprocessing on specified subjects/sessions/tasks.

_parse_args() : Argument parser
    Defines and parses all command-line arguments including:
    - Reader selection (BIDS or glob)
    - Dataset location (BIDS root or data root)
    - Subject/session/task filters
    - Configuration file path
    - Logging options

Command-Line Usage
------------------
Basic BIDS usage (default):
    python src/cli.py --bids-root /path/to/bids --config config.yaml

With subject and task filtering:
    python src/cli.py --bids-root /path/to/bids --subjects 01 02 --tasks rest

Using glob pattern matching:
    python src/cli.py --reader glob --data-root /path/to/data \\
        --glob-pattern "sub-{subject}/ses-{session}/eeg/sub-{subject}_task-{task}_eeg.vhdr" \\
        --subjects 01 02 --tasks rest

With logging to file:
    python src/cli.py --bids-root /path/to/bids --log-file pipeline.log --log-level DEBUG

Available Arguments
-------------------
Reader selection:
  --reader            Reader type: "bids" (default) or "glob"
  --bids-root         Path to BIDS root (required for BIDS reader)
  --data-root         Path to data root (required for glob reader)
  --glob-pattern      Glob pattern with {variable} placeholders (required for glob reader)
  --datatype          BIDS datatype to process: eeg, meg, ieeg or nirs
                      (BIDS reader only; detected from the dataset if omitted)

Optional filters (if not specified, all matching files are processed):
  --subjects          Subject ID(s) to process
  --sessions          Session ID(s) to process
  --tasks             Task name(s) to process
  --acquisitions      Acquisition parameter(s) to process
  --runs              Run ID(s) to process
  --extension         File extension (default: .vhdr; use .fif for MEG)

Other options:
  --output-root       Custom output path (default: bids-root/derivatives/meegflow)
  --config            Path to YAML configuration file
  --log-file          Path to log file (default: console output)
  --log-level         Logging level: DEBUG, INFO, WARNING, ERROR (default: INFO)

See README.md for detailed examples and documentation.
"""
import argparse
import json
import yaml
from pathlib import Path
from mne.utils import logger, set_log_file, set_log_level
from .pipeline import MEEGFlowPipeline
from .utils import NpEncoder

def _parse_args():
    parser = argparse.ArgumentParser(description='Run MEEG preprocessing pipeline on one or more subjects.')
    parser.add_argument('--bids-root', required=False, help='Path to BIDS root (required for BIDS reader).')
    parser.add_argument('--output-root', required=False, help='Path to output derivatives root.')
    
    # Reader selection
    parser.add_argument(
        '--reader',
        type=str,
        default='bids',
        choices=['bids', 'glob'],
        help='Reader type: "bids" for BIDS datasets or "glob" for glob pattern matching (default: bids).'
    )
    parser.add_argument(
        '--data-root',
        type=str,
        required=False,
        help='Path to data root directory (required for glob reader).'
    )
    parser.add_argument(
        '--glob-pattern',
        type=str,
        required=False,
        help='Glob pattern with {variable} placeholders for glob reader, e.g., "data/sub-{subject}/ses-{session}/eeg/sub-{subject}_task-{task}_eeg.vhdr"'
    )
    
    parser.add_argument(
        '--datatype',
        type=str,
        required=False,
        choices=['eeg', 'meg', 'ieeg', 'nirs'],
        help='BIDS datatype to process (BIDS reader only). If not provided, it is detected from the dataset.'
    )

    parser.add_argument(
        '--subjects',
        nargs='+',
        required=False,
        help='Subject ID(s) to process. Provide multiple subject IDs separated by spaces e.g. --subjects 01 02. If not provided, all subjects will be processed.'
    )
    parser.add_argument(
        '--sessions',
        nargs='+',
        required=False,
        help='Session ID(s) to process. Provide multiple session IDs separated by spaces.'
    )
    parser.add_argument(
        '--tasks',
        nargs='+',
        required=False,
        help='Task(s) to process. Provide multiple task names separated by spaces.'
    )
    parser.add_argument(
        '--acquisitions',
        nargs='+',
        required=False,
        help='Acquisition parameter(s) to process.'
    )
    parser.add_argument(
        '--runs',
        nargs='+',
        required=False,
        help='Run ID(s) to process. Provide multiple run IDs separated by spaces e.g. --runs 01 02. If not provided, all runs will be processed.'
    )
    parser.add_argument(
        '--extension',
        type=str,
        default='.vhdr',
        help='File extension to process.'
    )
    parser.add_argument(
        '--io-backend',
        type=str,
        default='read_raw_bids',
        help='MNE IO backend function to read files (default: read_raw_bids).'
    )
    parser.add_argument('--config', required=False, help='Path to YAML config file with preprocessing parameters.')
    parser.add_argument('--log-file', required=False, help='Path to log file. If not specified, logs will be printed to console.')
    parser.add_argument('--log-level', required=False, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='Logging level (default: INFO).')
    return parser.parse_args()

def main():
    """Entry point for the ``meegflow`` command-line tool.

    Parses command-line arguments, configures MNE logging, builds a
    ``MEEGFlowPipeline`` from the supplied YAML config, and runs it.
    """
    args = _parse_args()
    
    # Configure logging
    set_log_level(args.log_level)
    if args.log_file:
        log_path = Path(args.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        set_log_file(str(log_path), overwrite=False)
        logger.info(f"Logging to file: {log_path}")
    
    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)

    logger.info("Starting MEEGFlow preprocessing pipeline")
    
    # Create the appropriate reader
    if args.reader == 'bids':
        if not args.bids_root:
            logger.error("--bids-root is required when using BIDS reader")
            raise ValueError("--bids-root is required when using BIDS reader")
        
        logger.info(f"Using BIDS reader")
        logger.info(f"BIDS root: {args.bids_root}")
        
        from .readers import BIDSReader
        reader = BIDSReader(args.bids_root, datatype=args.datatype)
        
    elif args.reader == 'glob':
        if not args.data_root:
            logger.error("--data-root is required when using glob reader")
            raise ValueError("--data-root is required when using glob reader")
        if not args.glob_pattern:
            logger.error("--glob-pattern is required when using glob reader")
            raise ValueError("--glob-pattern is required when using glob reader")
        
        logger.info(f"Using glob reader")
        logger.info(f"Data root: {args.data_root}")
        logger.info(f"Glob pattern: {args.glob_pattern}")
        
        from .readers import GlobReader
        reader = GlobReader(args.data_root, args.glob_pattern)
    
    if args.output_root:
        logger.info(f"Output root: {args.output_root}")
    
    # print all arguments
    logger.info("Pipeline parameters:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")
    
    # Create pipeline with reader
    pipeline = MEEGFlowPipeline(
        reader=reader,
        output_root=args.output_root, 
        config=config
    )
    results = pipeline.run_pipeline(
        subjects=args.subjects,
        sessions=args.sessions,
        tasks=args.tasks,
        acquisitions=args.acquisitions,
        runs=args.runs,
        extension=args.extension,
        io_backend=args.io_backend
    )

    # Log summary of results
    logger.info("=" * 60)
    logger.info("Pipeline execution completed")
    logger.info("=" * 60)
    
    total_recordings = sum(len(v) for v in results.values())
    total_errors = sum(1 for subj_results in results.values() for r in subj_results if 'error' in r)
    total_success = total_recordings - total_errors
    
    logger.info(f"Total recordings processed: {total_recordings}")
    logger.info(f"Successful: {total_success}")
    logger.info(f"Failed: {total_errors}")
    
    # Log detailed results in JSON format
    logger.info("\nDetailed results:")
    for subject, subject_results in results.items():
        logger.info(f"\nSubject {subject}:")
        for result in subject_results:
            if 'error' in result:
                logger.error(f"  - Error: {result['error']}")
            else:
                task = result.get('task', 'N/A')
                session = result.get('session', 'N/A')
                n_epochs = result.get('n_epochs', 'N/A')
                logger.info(f"  - Task: {task}, Session: {session}, Epochs: {n_epochs}")
    
    # Also write results to JSON file for easy processing
    output_json = Path(pipeline.dataset_root) / "derivatives" / "meegflow" / "pipeline_results.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2, cls=NpEncoder)
    logger.info(f"\nResults written to: {output_json}")


if __name__ == '__main__':
    main()
