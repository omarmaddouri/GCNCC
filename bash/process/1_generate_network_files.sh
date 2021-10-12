#!/bin/bash
module load Python/3.6.6-intel-2018b
cd $SCRATCH/gcncc.bioinformatics                    # Go to project directory
source venv/bin/activate                            # Activate virtual environment (Command line should show environment name on left)
## Prepare network files
python -u scripts/reference/1_generate_ppi_ids.py
python -u scripts/reference/2_generate_ppi_edge_list.py
python -u scripts/reference/3_generate_ppi_adjacency.py
deactivate                                          # Deactivate virtual environment