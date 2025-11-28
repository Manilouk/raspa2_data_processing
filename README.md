# raspa2_data_processing
Python script for parsing RASPA2 .data output files, extracting adsorption data, and exporting results into structured JSON and CSV format.

# RASPA2 Data Processing

This project contains a Python script for automatically parsing RASPA2 simulation output files (.data format) and extracting key adsorption properties for different MOFs and pressures. The script reads each output file, converts its contents into a dictionary, saves the raw output as .json, and finally aggregates all extracted values into a DataFrame, which is exported as data_absorption_flex.csv.

Features:
a) Average loading (absolute), Units for loading, Enthalpy of adsorption, Total energy
b) Convert simulation outputs into structured .json 
c) Supports loops over: i) multiple MOFs, ii) multiple pressures, iii) multiple simulation files




