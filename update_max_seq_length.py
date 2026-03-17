#!/usr/bin/env python3
"""
Script to update max_seq_length to max_length in notebook cells that contain Trainer classes.
Only updates cells that contain SFTTrainer, GRPOTrainer, DPOTrainer, ORPOTrainer, or other Trainer classes
AND also contain dataset_kwargs = {"skip_prepare_dataset": True}.
"""

import json
import os
import re
import glob
from pathlib import Path


def find_trainer_in_cell(cell_source):
    """
    Check if a cell contains any Trainer class instantiation AND 
    dataset_kwargs = {"skip_prepare_dataset": True}.
    Returns True if both conditions are met.
    """
    if isinstance(cell_source, list):
        source_text = ''.join(cell_source)
    else:
        source_text = str(cell_source)
    
    # Look for trainer patterns
    trainer_patterns = [
        r'SFTTrainer\s*\(',
        r'GRPOTrainer\s*\(',
        r'DPOTrainer\s*\(',
        r'ORPOTrainer\s*\(',
        r'trainer\s*=\s*\w*Trainer\s*\(',
        # Also check for trainer configuration patterns
        r'trainer\s*=.*Trainer\s*\(',
    ]
    
    has_trainer = False
    for pattern in trainer_patterns:
        if re.search(pattern, source_text, re.IGNORECASE):
            has_trainer = True
            break
    
    if not has_trainer:
        return False
    
    # Check for dataset_kwargs = {"skip_prepare_dataset": True}
    dataset_kwargs_patterns = [
        r'dataset_kwargs\s*=\s*\{\s*["\']skip_prepare_dataset["\']\s*:\s*True\s*\}',
        r'dataset_kwargs\s*=\s*\{\s*["\']+skip_prepare_dataset["\']+\s*:\s*True\s*\}',
    ]
    
    for pattern in dataset_kwargs_patterns:
        if re.search(pattern, source_text, re.IGNORECASE):
            return True
    
    return False


def update_max_seq_length_in_source(source):
    """
    Update max_seq_length to max_length in the source code.
    Handles both list and string formats.
    """
    if isinstance(source, list):
        updated_source = []
        for line in source:
            # Replace max_seq_length = with max_length =
            updated_line = re.sub(
                r'\bmax_seq_length\s*=',
                'max_length =',
                line
            )
            updated_source.append(updated_line)
        return updated_source
    else:
        # Handle string format
        return re.sub(
            r'\bmax_seq_length\s*=',
            'max_length =',
            source
        )


def process_notebook(notebook_path):
    """
    Process a single notebook file to update max_seq_length to max_length.
    Only updates cells that contain both Trainer classes AND dataset_kwargs = {"skip_prepare_dataset": True}.
    Returns the number of cells updated.
    """
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
    except Exception as e:
        print(f"Error reading {notebook_path}: {e}")
        return 0
    
    cells_updated = 0
    
    for cell in notebook.get('cells', []):
        # Only process code cells
        if cell.get('cell_type') != 'code':
            continue
            
        source = cell.get('source', [])
        if not source:
            continue
        
        # Check if this cell contains trainer code AND dataset_kwargs with skip_prepare_dataset
        if find_trainer_in_cell(source):
            # Check if max_seq_length exists in this cell
            source_text = ''.join(source) if isinstance(source, list) else str(source)
            if 'max_seq_length' in source_text:
                print(f"  Found max_seq_length in trainer cell with dataset_kwargs, updating...")
                
                # Update the source
                updated_source = update_max_seq_length_in_source(source)
                cell['source'] = updated_source
                cells_updated += 1
    
    if cells_updated > 0:
        try:
            # Write back the updated notebook
            with open(notebook_path, 'w', encoding='utf-8') as f:
                json.dump(notebook, f, indent=1, ensure_ascii=False)
            print(f"  ✓ Updated {cells_updated} cell(s) in {notebook_path}")
        except Exception as e:
            print(f"  ✗ Error writing {notebook_path}: {e}")
            return 0
    
    return cells_updated


def main():
    """
    Main function to process all notebooks in the workspace.
    """
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    
    # Find all .ipynb files recursively
    notebook_patterns = [
        str(script_dir / "**/*.ipynb"),
        str(script_dir / "nb/**/*.ipynb"),
        str(script_dir / "original_template/**/*.ipynb"),
        str(script_dir / "testing_chamber/**/*.ipynb")
    ]
    
    all_notebooks = []
    for pattern in notebook_patterns:
        all_notebooks.extend(glob.glob(pattern, recursive=True))
    
    # Remove duplicates and sort
    all_notebooks = sorted(list(set(all_notebooks)))
    
    print(f"Found {len(all_notebooks)} notebook files to process")
    print("=" * 60)
    
    total_files_updated = 0
    total_cells_updated = 0
    
    for notebook_path in all_notebooks:
        # Skip if file doesn't exist or is not readable
        if not os.path.isfile(notebook_path):
            continue
            
        print(f"Processing: {os.path.relpath(notebook_path, script_dir)}")
        
        cells_updated = process_notebook(notebook_path)
        if cells_updated > 0:
            total_files_updated += 1
            total_cells_updated += cells_updated
        else:
            print(f"  No changes needed")
    
    print("=" * 60)
    print(f"Summary:")
    print(f"  Files processed: {len(all_notebooks)}")
    print(f"  Files updated: {total_files_updated}")
    print(f"  Total cells updated: {total_cells_updated}")
    
    if total_files_updated > 0:
        print(f"\n✓ Successfully updated max_seq_length to max_length in {total_files_updated} files!")
        print(f"  Note: Only updated cells with Trainer classes AND dataset_kwargs = {{'skip_prepare_dataset': True}}")
    else:
        print(f"\nNo files needed updating.")
        print(f"  Note: Only cells with Trainer classes AND dataset_kwargs = {{'skip_prepare_dataset': True}} are updated")


if __name__ == "__main__":
    main()
