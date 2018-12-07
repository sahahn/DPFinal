# DPFinal

This directory is dedicated to code used for my Data Privacy final project. Generating Differentially Private Synthetic Data.

-The Main script is within the notebook "Gen_Synthetic_Data", while all of the hyperparameters and privacy composition are present within 'config.py'. 

-Likewise, "old_synthetic_data" and "old_config" represent, you guessed it, older versions of both files created before a major project shift. They should essentially be ignored, I do not know why I don't delete them.

-Jupyter Notebook: Plots and Analysis, are used to evaluate synthetic data and to create visualizations of thosee results.

-FCBF_DP.py is a DP implementation of Fast correlation based filter.

-structure_learn.py is the DP structure learning script

-utils.py contains a number of useful function as well as the code designed to load the dataset, and kbin it.

-param_learn.py contains the code to generate DP conditional marginals

-generate_data.py is the script to generate synth data given conditional marginals and parents ect...



