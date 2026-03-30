#!/usr/bin/env python3
"""
Script to clean CSV files by removing files with empty 'Probability' values.
"""

import os
import csv
import sys
import argparse
from pathlib import Path

# List of target directories
TARGET_DIRS = [
    "target_hemoglobin_grade2plus",
    "target_neutrophil_grade2plus",
    "target_platelet_grade2plus",
    "target_AKI_grade2plus",
    "target_ALT_grade2plus",
    "target_AST_grade2plus",
    "target_bilirubin_grade2plus",
    "target_esas_pain_3pt_change",
    "target_esas_tiredness_3pt_change",
    "target_esas_nausea_3pt_change",
    "target_esas_depression_3pt_change",
    "target_esas_anxiety_3pt_change",
    "target_esas_drowsiness_3pt_change",
    "target_esas_appetite_3pt_change",
    "target_esas_well_being_3pt_change",
    "target_esas_shortness_of_breath_3pt_change",
    "target_death_in_30d",
    "target_death_in_365d",
    "target_ED_visit"
]

def should_delete_csv(file_path):
    """
    Check if a CSV file should be deleted based on empty 'Probability' column.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        True if file should be deleted, False otherwise
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # Read the file content
            reader = csv.DictReader(f)
            
            # Get the first (and only) data row
            row = next(reader, None)
            
            if row is None:
                print(f"  WARNING: No data rows in {file_path}")
                return True
            
            # Check if 'Probability' column exists and is empty
            if 'Probability' in row:
                prob_value = row['Probability'].strip()
                if not prob_value:
                    return True
            else:
                print(f"  WARNING: 'Probability' column not found in {file_path}")
                
    except Exception as e:
        print(f"  ERROR reading {file_path}: {e}")
        return True
    
    return False

def main(base_path):
    deleted_count = 0
    kept_count = 0
    
    # Convert to absolute path
    base_path = os.path.abspath(base_path)
    
    if not os.path.exists(base_path):
        print(f"ERROR: Base path does not exist: {base_path}")
        sys.exit(1)
    
    print(f"Base path: {base_path}\n")
    
    # Process each target directory
    for target_dir in TARGET_DIRS:
        target_dir_path = os.path.join(base_path, target_dir)
        
        if not os.path.exists(target_dir_path):
            print(f"Directory not found: {target_dir_path}")
            continue
            
        print(f"\nProcessing directory: {target_dir}")
        
        # Loop through subdirectories that start with "note"
        for subdir in os.listdir(target_dir_path):
            subdir_path = os.path.join(target_dir_path, subdir)
            
            # Check if it's a directory and starts with "note"
            if os.path.isdir(subdir_path) and subdir.startswith("note"):
                print(f"  Checking subdirectory: {subdir}")
                
                # Process CSV files that start with "mrn"
                for filename in os.listdir(subdir_path):
                    if filename.startswith("mrn") and filename.endswith(".csv"):
                        file_path = os.path.join(subdir_path, filename)
                        
                        if should_delete_csv(file_path):
                            print(f"    DELETING: {filename}")
                            os.remove(file_path)
                            deleted_count += 1
                        else:
                            kept_count += 1
    
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Files deleted: {deleted_count}")
    print(f"  Files kept: {kept_count}")
    print(f"{'='*60}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Clean CSV files by removing files with empty 'Probability' values."
    )
    parser.add_argument(
        'base_path',
        type=str,
        help='Path to the directory containing the target directories'
    )
    
    args = parser.parse_args()
    main(args.base_path)