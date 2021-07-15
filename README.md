# JSDM4POD
a Joint Species Distribution Model for Presence-Only Data

This repository contains working examples for running (and fitting) the model described in the article:
/A taxonomic-based joint species distribution model for presence-only data/ by Juan Escamilla-Molgora, Luigi Sedda, Peter Diggle and Peter Atkinson in 2020. 


## Installation
The model uses PyStan, Jupyter notebooks and Pandas. 
The best way to create a Conda environment.
The files with the necessary libraries and packages are stored as: `requirements.yml`

To install the environment (with a working Anaconda installation) do:

`conda env create -f requirements.yml`

*IMPORTANT*: The library PyStan needs to be version 2.x. The notebooks would not work with version 3.x.



## Setting up data files
Due to Github policy restrictions on file sizes, some of the data needed are compressed in the folder 'data/'. 
Please unzip these files before.


## Running the case studies
The folder 'notebooks' contains the working examples of case studies 1 and 2. 
