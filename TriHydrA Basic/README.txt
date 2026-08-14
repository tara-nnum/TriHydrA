================================================================================
                           TRIHYDRA BASIC
================================================================================

This folder contains everything needed to try TriHydrA from Python or Jupyter.
The instructions are portable: "this folder" means wherever you copied or
extracted the TriHydrA Basic files on your own computer.

FIRST-TIME INSTALLATION
-----------------------
TriHydrA requires Python 3.11 and the packages listed in requirements.txt.
You do not need an environment with a particular name. A separate environment
is recommended so that TriHydrA does not conflict with other Python projects.

Using Anaconda Prompt:

1. Change into the folder containing this README and requirements.txt. For
   example, if you extracted the files into a folder named TriHydrA Basic:
       cd /d "path\to\TriHydrA Basic"
2. Create and activate an environment:
       conda create -n trihydra-basic -c conda-forge python=3.11 pip
       conda activate trihydra-basic
3. Install the required packages:
       python -m pip install --upgrade pip
       python -m pip install -r requirements.txt
4. Register the environment for Jupyter:
       python -m ipykernel install --user --name trihydra-basic --display-name "Python (TriHydrA Basic)"

The environment name "trihydra-basic" is only a convenient example. You may
choose another name.

PLAIN PYTHON
------------
Before continuing, complete FIRST-TIME INSTALLATION above.

1. Keep these items together in the TriHydrA Basic folder:
       trihydra/
       run_trihydra_basic.py
       example_three_stations.csv
       context.csv
2. Open Anaconda Prompt.
3. Activate the environment:
       conda activate trihydra-basic
4. Change into the folder where you placed the TriHydrA Basic files:
       cd /d "path\to\TriHydrA Basic"
5. Confirm that Python is available:
       python --version
   The version should begin with Python 3.11.
6. Run TriHydrA:
       python run_trihydra_basic.py
7. Wait until this message appears:
       Finished. Open the files in: ...\results
8. Open the newly created results folder. Each station will have its own
   folder containing the TXT reports and interactive HTML diagnostics.
If Python reports that a module is missing, reactivate the environment and run:
       python -m pip install -r requirements.txt

GUIDED TWO-SERIES COMPARISON
----------------------------
Use this when you want to compare any two columns in the supplied example CSV,
such as an observation and a model series. You do not need to edit Python code.

1. Complete FIRST-TIME INSTALLATION above.
2. Open Anaconda Prompt.
3. Activate the environment:
       conda activate trihydra-basic
4. Change into the extracted TriHydrA Basic folder:
       cd /d "path\to\TriHydrA Basic"
5. Run the guided comparison:
       python run_trihydra_comparison.py
6. The script lists every available series. Type the number of the
   observation/reference series and press Enter.
7. Type the number of the model or other series you want to compare and press
   Enter. The same series cannot be selected twice.
8. Wait for the success message, then open the new comparison_results folder.
   The reference station folder contains the individual Layer 1 and Layer 2
   reports plus comparison_evidence.txt and the interactive comparison plot.

The supplied three columns are real gauge series and are included to demonstrate
the mechanism. An observation-model assessment should normally compare two
series representing the same station, in the same units.

JUPYTER NOTEBOOK
----------------
Before continuing, complete FIRST-TIME INSTALLATION above.

1. Keep these items together in the TriHydrA Basic folder:
       trihydra/
       basic_user_example.ipynb
       example_three_stations.csv
       context.csv
2. Open Anaconda Prompt.
3. Activate the environment:
       conda activate trihydra-basic
4. Change into the folder where you placed the TriHydrA Basic files:
       cd /d "path\to\TriHydrA Basic"
5. Start Jupyter:
       jupyter lab
6. Jupyter should open in your web browser. In the file list, double-click:
       basic_user_example.ipynb
7. Select the TriHydrA environment as the notebook kernel:
       Kernel → Change Kernel → Python (TriHydrA Basic)
8. Run the notebook from top to bottom:
       Run → Run All Cells
   You may also run one cell at a time by selecting a cell and pressing
   Shift+Enter.
9. Wait for the final cell to confirm that processing is complete.
10. Return to the Jupyter file list and open the newly created results folder.
    Each station will have TXT reports and interactive HTML diagnostics.
If "Python (TriHydrA Basic)" is not available in the kernel list, return to
Anaconda Prompt and run:
       conda activate trihydra-basic
       python -m ipykernel install --user --name trihydra-basic --display-name "Python (TriHydrA Basic)"

Then restart Jupyter Lab.

YOUR OWN CSV
------------
Use a wide CSV with this structure:

date,station_A,station_B,station_C
2000-01-01,1.20,2.31,0.85
2000-01-02,1.15,2.44,0.79

- The first column must contain dates.
- Each remaining column represents one station.
- Missing observations may be blank.
- Change DATA_FILE and UNIT in the script or notebook when using your data.
- Layer 3 additionally requires matching station IDs in context.csv.
- The supplied context.csv contains metadata only for the three supplied
  example gauges. Replace its rows when using different stations.
- Required context columns are station_id, longitude, latitude, river_name,
  catchment_name, catchment_area_km2, and series_type.

PORTABLE PATHS
--------------
The example script and notebook locate their input files relative to their own
folder. They do not require or contain the original author's directory. Keep
the script/notebook, the trihydra folder, the example data, and context.csv
together unless you intentionally change the paths in your own copy.

OUTPUTS
-------
The results folder is created automatically. Each station receives:

- summary.txt
- layer1_evidence.txt
- layer2_evidence.txt
- layer3_evidence.txt when assessed
- interactive HTML diagnostic plots

The top-level results folder also contains network_summary.txt.

================================================================================
