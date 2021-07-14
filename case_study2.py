#!/usr/bin/env python3

# A Standalone version for running the case study 2.
#
"""

Operational script for fitting multispecies model CASE study 2
==============================================================
..
This script runs the model for paper 3.
"""
__author__ = "Juan Escamilla Molgora"
__copyright__ = "Copyright 2021, JEM"
__license__ = "GPL"
__version__ = "2.2.1"

# Load libraries
import pandas as pd
import geopandas as gpd
import numpy as np
import scipy as sp
import patsy
import pystan
#import statsmodels.api as stmod
import pickle
import networkx as nt
import logging
import os
import sys,getopt

## PATH to store the pickled model (i.e. the posterior sample).
## Here it is the standard biospytial output path
PATH = "/data/output/posterior_samps/"
PATH_BASE = ''
#PATH_HOME = PATH_BASE + 'paper3code/'
PATH_HOME = PATH_BASE + ''

sys.path.append(PATH_BASE)

import models

logger = logging.getLogger('Multispecies_car-cs2')
logger.setLevel(logging.DEBUG)


def compileDataDict(data_dic,Mat,eco_formula = '~ standardize(Elevation_m) + standardize(Precipitation_m) +standardize(MeanTemp_m) -1 ',sample_formula = '~ C(wwf_mhtnam) + C(tipolgia14)'):
    """
    Prepares de data diccionary to be used by the STAN model.
    data_dic : (Geopandas) The data stacked (multilevel) and multiindexed by cell id,level, Observations and geometry.
    Mat: The matrix (type sparse matrix).

    Returns: Dictionary
    """
    logger.info("Compiling Data input for the STAN model")
    means = data_dic.mean()
    for column in ['Elevation_m', 'MeanTemp_m', 'Precipitation_m','SolarRadiation_m','Population_m','distance_to_road']:
        data_dic[column].replace(np.nan,means[column],inplace=True)
    
    ## Add an intercept to each process.
    ## HARDCODED
    ## HARDCODED

    logger.info("Starting model for TerrEcoregions and Typology ...")
    X_eco = patsy.dmatrix(eco_formula,data=data_dic,return_type='dataframe',NA_action="drop")
    X_samp = patsy.dmatrix(sample_formula,data=data_dic,return_type='dataframe',NA_action="drop")
    X = pd.concat([X_eco,X_samp],axis=1)
    
    
   
    levels = data_dic.index.get_level_values(1)
    MaxLevels = max(levels)
    NM = Mat.todense()
    N_edges = int(np.sum(NM)/2.0)
    try:
        n_miss = data_dic.groupby('Y').count().loc[2.0][0]
    except:
        n_miss = 0
    
    N = X.shape[0]
    K = X.shape[1]
    J = MaxLevels
    adjacency_matrix = NM  # Removed islands from M

    n_eco_covs = X_eco.shape[1]
    n_samp_covs = X_samp.shape[1]
    n_areas = int(X.shape[0]/J)
    y = data_dic.Y.values.astype('int')
    
    data_multilevel_CAR = {'N' : N,
            'N_ecological_covariates' : n_eco_covs,
            'N_sample_covariates' : n_samp_covs,
            'J': J,
            'N_edges' : N_edges,
            'N_areas' : n_areas,
            'level': levels, # Stan counts starting at 1
            'W' : adjacency_matrix,
            'y': y,
            'x': X,
            'N_miss' : n_miss,
            'Y_miss_array':np.array([1])
           }
    return(data_multilevel_CAR)

def prepareInputData(INPUT_DATA_FILE = 'data/case-study2/casestudy2_grid128_fullstack_env_anth_extended_selowres.gpkg',INPUT_LATTICE_DATA = 'data/case-study2/lattice128.pkl-p2'):
    """
    Prepare input data to be run by the STAN model.
    Returns a list where each element is a duple of composed of a stacked dataframe and the corresponding adjacency matrix.
    """
    
    logger.info("Preparing input data")
    ## HARDCODED
    datafile = PATH_HOME + INPUT_DATA_FILE
    ## Readfile
    data = gpd.read_file(datafile,na_values=np.nan)
    data = data.replace('N.A.', np.nan)
    data.set_index(['id','level'],inplace=True)
    data.sort_index(level=1,inplace=True,sort_remaining=True)
    NAN_VALUE_MODEL = 2
    data.Y.replace(np.nan,NAN_VALUE_MODEL,inplace=True)
    data.rename(columns={'Dist.to.road_m':'distance_to_road'},inplace=True)

    data_graph = PATH_HOME + INPUT_LATTICE_DATA
    with open(data_graph, "rb") as input_file:
        lattice_128 = pickle.load(input_file)
    
    islands_ids = list(nt.algorithms.components.connected_components(lattice_128))
    subgraphs = list(map(lambda idxs : lattice_128.subgraph(idxs),islands_ids))
    ## Order by cell ids
    subdfs = list(map(lambda island_ids :  data.loc[data.index.get_level_values(0).isin(list(island_ids))].sort_index(level=1),islands_ids ))


    # Generate adjacency matrices and order by the node id
    sub_adjmats = list(map(lambda s : nt.adjacency_matrix(s,nodelist=sorted(s.nodes())),subgraphs))
    duple_df_mat = list(zip(subdfs,sub_adjmats))
    
    return(duple_df_mat)    

def getSTANModel():
    logger.info("Compiling source code of the multispecies model")
    multispecies_model = models.multispeciesCARModel_stationaryBernoulliMissingData()
    multispecies_model = pystan.StanModel(model_code=multispecies_model)
    return(multispecies_model)


def main(argv,FILE_POST_SAMP="multispecies_case.study2-terrecoregions-typology-std-35k-lowresse.pkl"):
    try:
        opts,args = getopt.getopt(argv,"hi:n:t:p:",["niters=","nchains=","thinning=","nthreads="])
    except getopt.GetoptError as error:
        logger.error('main_multispecies.py -i <num iterations> -n <num. chains> -t <thinning> -p <num of threadsv>\n Error: %s'%error)
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            logger.info('USAGE: main_multispecies.py -i <num iterations> -n <num. chains> -t <thinning> -p <num of threads>')
            sys.exit()
        elif opt in ("-i", "--niters"):
            NITERS = arg
            logger.info("Number of requested iterations: %s"%NITERS)
        elif opt in ("-n", "--nchains"):
            NCHAINS = arg
            logger.info("Number of independent chains: %s"%NCHAINS)
        elif opt in ("-t", "--thinning"):
            THINNING = arg
            logger.info("Requested thinning: %s"%THINNING)
        elif opt in ("-p", "--nthreads"):
            NTHREADS = arg
            logger.info("Number of threads for computing: %s"%NTHREADS)


    #os.environ['STAN_NUM_THREADS'] = NTHREADS
    SEED = 12345
    duple_df_mat = prepareInputData()    
    data_d, M = duple_df_mat[0]
    ## TEST: REMOVE MISSING DATA (2.0)
    data_d.Y.replace(2.0,0.0,inplace=True)
    data_multilevel_CAR = compileDataDict(data_d,M)
    ## Load model and compile
    multispecies_model = getSTANModel()
    logger.info("Commencing posterior sampling through HMCM")
    fit3 = multispecies_model.sampling(data=data_multilevel_CAR,
            iter=int(NITERS),
            chains=int(NCHAINS),
            thin=int(THINNING),
            control={'adapt_delta':0.81},
            seed=int(SEED)) #,'max_treedepth':15})
    logger.info("Sampling finished")
    logger.info("Initiating model and posterior sampling back-up through pickling method using protocol -1")
    ## HARDCODED
    _file = PATH + FILE_POST_SAMP
    with open(_file,"wb") as f:
        logger.info("Saving pickled file")
        pickle.dump({'model':multispecies_model, 'fit': fit3},f, protocol=-1)
        
    logger.info("End!, model and posterior sample have been returned by this main function")
    return(multispecies_model,fit3) 

## To unpickle do..
#import pickle
#with open("model_fit.pkl", "rb") as f:
#    data_dict = pickle.load(f)
#    # or with a list
#    # data_list = pickle.load(f)
#fit = data_dict['fit']
## fit = data_list[1]



if __name__ == "__main__":
    main(sys.argv[1:])
