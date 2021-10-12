#!/bin/bash
module load Python/3.6.6-intel-2018b                # Load Python module
cd $SCRATCH/gcncc.bioinformatics                    # Go to project directory
virtualenv venv                                     # Make a virtual environment "venv"
source venv/bin/activate                            # Activate virtual environment (Command line should show environment name on left)
pip install -r requirements.txt --upgrade           # Install and update the required packages
deactivate                                          # Deactivate virtual environment
