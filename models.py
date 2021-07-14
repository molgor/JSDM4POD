#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""

Stan models from the multispecies models (Paper 3)
==========
..
This module implements the models according to the 
framework defined in the paper 3.
"""
__author__ = "Juan Escamilla Mólgora"
__copyright__ = "Copyright 2019, JEM"
__license__ = "GPL"
__version__ = "2.2.1"

def univariateCARmodel():
    """
    Returns the a stan code for a univariate CAR (BYM) model.
    Usage with data:
        example:
            N_edges = nodes.shape[0]
            N = nX.shape[0]
            K = nX.shape[1]
            data = {'N' : N,
                    'K' : K,
                    'N_edges' : N_edges,
                    'node1': idx_node1,
                    'node2': idx_node2,
                    'y': y.values.flatten(),
                    'x': nX,
                    }
    """
#Univariate CAR model
    univariate_bym = """
        functions {
          real icar_normal_lpdf(vector phi, int N, int[] node1, int[] node2) {
            return -0.5 * dot_self(phi[node1] - phi[node2])
              + normal_lpdf(sum(phi) | 0, 0.001 * N);
         }
        }
        data {
          int<lower=0> N;
          int<lower=0> N_edges;
          int<lower=1> K;                // num covariates
          matrix[N, K] x;                // design matrix
          int<lower=1, upper=N> node1[N_edges];  // node1[i] adjacent to node2[i]
          int<lower=1, upper=N> node2[N_edges];  // and node1[i] < node2[i]
          int<lower=0,upper=1> y[N];              // count outcomes
        }
        
        parameters {
          vector[K] betas;       // covariates
          real<lower=0> sigma;    // spatial standard deviation
          vector[N] phi_raw;         // spatial effects
        }
        transformed parameters { 
          vector[N] phi = sigma * phi_raw;
          }
        model {
          y ~ bernoulli_logit(x*betas + phi);
          betas ~ normal(0.0, 10^2.5);
          target += -3*log(sigma) - 1/(sigma)^2;  // Stan equiv of BUGS model prior on tau
          phi_raw ~ icar_normal_lpdf(N, node1, node2);
        }
        generated quantities {
          real tau = sigma^-2;
          vector[N] P = inv_logit( x * betas + phi);
          
        }
        
        """
    return(univariate_bym)


def logisticHierarchicalCAR():
    """
    The multispecies logistic hierarchical model (No mixing effect) between sample
    and species.

    Assumptions: 
        Betas come from different distributions, (mus and sigmas)

    Example usage with data:
        ok = simulation.nodeslist_to_cannonical_indexlist(simulation,raw_nodes)
        idx_node1 = ok['idx_node1']
        idx_node2 = ok['idx_node2']
        levels = np.array(simulation.index.get_level_values(0)) + 1
        
        MaxLevels =2
        N_edges = len(idx_node1)
        N = X.shape[0]
        K = X.shape[1] 
        J = MaxLevels 
        ## number of levels to model
        
        data_multilevel = {'N' : N,
                'K' : K,
                'J': J,                  
                'N_edges' : N_edges,
                'node1': idx_node1,
                'node2': idx_node2,
                'N_areas' : X.shape[0]/J,   
                'level': levels, # Stan counts starting at 1
                 #'scaling_factor': 1.0,
                'y': simulation['logit_p_sim'].values,
                #'y': simulation['q'].values,              
                #'y': yy.simulated.values.flatten().astype('float'),
                'x': X,
               }
        
    """
    logistic_hierarchical_car = """
        functions {
          real icar_normal_lpdf(vector phi, int N, int[] node1, int[] node2) {
            return -0.5 * dot_self(phi[node1] - phi[node2])
              + normal_lpdf(sum(phi) | 0, 0.001 * N);
         }
        }
        
        data {
          int<lower=0> N; // num obs. 
          int<lower=0> J; // number of levels
          int<lower=1> K;                // num covariates
          // matrix[N, K] x;                // design matrix
          row_vector[K] x[N];  // supposedly much more efficient
          int<lower=1,upper=J> level[N];  // type of level (spec)
          vector[N] y;
        //  int<lower=0,upper=1> y[N];              // observations, in this case is binary.
        // data for the spatial structure
          int<lower=0> N_areas; // number of areas in the region.
          int<lower=0> N_edges;
          int<lower=1, upper=N> node1[N_edges];  // node1[i] adjacent to node2[i]
          int<lower=1, upper=N> node2[N_edges];  // and node1[i] < node2[i]
        
        }
        
        parameters {
        
          //real mu[K];                // This is assuming each covariate comes from different distribution
          //real mu;                  // This assumes all parameters come from same distribution
          vector[K] beta[J];         // Each level has an assigend beta of K dimension.
          //real<lower=0> sigma[J];    // This assumes that each level has the different distribution (variance)
          //real<lower=0> sigma;       // This assumes same variance for each level.  
        
          // Spatial thing..y
          //real<lower=0> sigma_phi;      // spatial (variance) 
          //real<lower=0> sigma_theta;    // Unstructured random effect (variance)
          real<lower=0> tau_theta;   // precision of heterogeneous effects
          real<lower=0> tau_phi;     // precision of spatial effects  
        
          
          vector[N_areas] phi_raw;      // spatial effects
          vector[N_areas] theta;        // Unstructured random effect (field)
          real<lower=0> nu_2;           // variance in the observations
          
        }
        
        
        transformed parameters { 
        
          real<lower=0> sigma_theta = inv(sqrt(tau_theta));  // convert precision to sigma
          real<lower=0> sigma_phi = inv(sqrt(tau_phi));      // convert precision to sigma  
          vector[N_areas] phi = (phi_raw * sigma_phi) + (theta * sigma_theta);  
          }
          
          
        model {
            // def variable
            vector[N] x_beta_ll;
            
            // prior
           //mu ~ normal(0,10000);
           //mu ~ uniform(0,500);
            
            
            for (j in 1:J){
                    //beta[j] ~ normal(mu, 100000);
                    //beta[j] ~ normal(mu, sigma[j]); // Different variance per level
                    //beta[j] ~ normal(0, sigma[j]); // Different variance per level
                    beta[j] ~ normal(0, 100000); // Same variance per level
            }
            for (i in 1:N)
                x_beta_ll[i] = x[i] * beta[level[i]] + phi[ (i % N_areas)+1 ]; // This because the N_areas is half N, this assures common component
                
            y ~ normal(x_beta_ll,nu_2);
            //y ~ bernoulli_logit(x_beta_ll);
            
            //target += -3*log(sigma_phi) - 1/(sigma_phi)^2;  // Stan equiv of BUGS model prior on tau
            phi_raw ~ icar_normal_lpdf(N_areas, node1, node2);
            theta ~ normal(0,1);
            nu_2 ~ inv_gamma(1, 0.01);  // Carlin WinBUGS priors
            //sigma ~ inv_gamma(1, 0.01);
            //sigma ~ cauchy(0,1);
            //sigma_theta ~ inv_gamma(1, 0.01);
        
            tau_theta ~ gamma(3.2761, 1.81);  // Carlin WinBUGS priors
            tau_phi ~ gamma(1, 1);            // Carlin WinBUGS priors   
        
        }
        
        generated quantities {
          real tau = sigma_phi^-2;
              vector[J] P[N_areas];
              for (j in 1:J) { 
                  for (i  in 1:N_areas)
                      //P[i][j] = inv_logit(x[i] * beta[j] + phi[i ]);
                      P[i][j] = x[i] * beta[j] + phi[i ];
        }
        
          
        }
        
        """
    return(logistic_hierarchical_car)


def multispeciesCARModelGAUSSIANOBSERVATIONS():
    """
    The multispecies presence-only model implemented in STAN.
    
    > note: This is the original code. Still underdevelopment but had worked good.
    It was used for fitting the simulation. 

    Assumptions:
        Betas come from different distributions, (mus and sigmas).
        The Spatial random effect is shared between all the processes.

    Example Usage:
        idx_node1 = ok['idx_node1']
        idx_node2 = ok['idx_node2']
        levels = np.array(simulation.index.get_level_values(0)) + 1
        
        MaxLevels = 4
        N_edges = len(idx_node1)
        N = X.shape[0]
        K = X.shape[1] 
        J = MaxLevels 
        ## number of levels to model
        
        data_multilevel = {'N' : N,
                'K' : K,
                'J': J,                  
                'N_edges' : N_edges,
                'node1': idx_node1,
                'node2': idx_node2,
                'N_areas' : X.shape[0]/J,   
                'level': levels, # Stan counts starting at 1
                 #'scaling_factor': 1.0,
                #'y': simulation['logit_p_sim'].values,
                'y': simulation['q'].values,              
                #'y': yy.simulated.values.flatten().astype('float'),
                'x': X,
               }
    """
    ## The next layer, a process q that mixes the two processes
    multispecies_model = """
functions {
  real icar_normal_lpdf(vector phi, int N, int[] node1, int[] node2) {
    return -0.5 * dot_self(phi[node1] - phi[node2])
      + normal_lpdf(sum(phi) | 0, 0.001 * N);
 }
}

data {
  int<lower=0> N; // num obs. 
  int<lower=0> J; // number of levels
  int<lower=1> K;                // num covariates
  row_vector[K] x[N];  // supposedly much more efficient
  int<lower=1,upper=J> level[N];  // type of level (spec)
  vector[N] y;
//  int<lower=0,upper=1> y[N];              // observations, in this case is binary.
// data for the spatial structure
  int<lower=0> N_areas; // number of areas in the region.
  int<lower=0> N_edges;
  int<lower=1, upper=N> node1[N_edges];  // node1[i] adjacent to node2[i]
  int<lower=1, upper=N> node2[N_edges];  // and node1[i] < node2[i]

}



parameters {

  real mu[K];                // This is assuming each covariate comes from different distribution
  real<lower=0> sigma_betas[J];    // This assumes that each level has the different distribution (variance)
  vector[K] beta[J];         // Each level has an assigend beta of K dimension.

  // Spatial thing..y
  real<lower=0> sigma_phi;      // spatial (variance) 
  real<lower=0> sigma_theta;    // Unstructured random effect (variance)
  vector[N_areas] phi_raw;      // spatial effects
  vector[N_areas] theta;        // Unstructured random effect (field)
  //vector[N] gamma;  

  real<lower=0> nu_2;           // variance in the observations
  //real<lower=0> sigma_q[J];           // variance in the observations
  
  // The alpha parameter, one per level.

  simplex[2] alpha_1[J - 1];


}


transformed parameters { 
    //The spatial effect
    vector[N_areas] phi = phi_raw + theta;
    simplex[2] alpha[J];
    for (j in 1:J - 1){
        alpha[j] = alpha_1[j];
    }
    alpha[J][1] = 0.0;
    alpha[J][2] = 1.0; 
  }
  


  
model {
    // def variable
  vector[N] S;
  vector[N] P;
  vector[N] Q;

    


    // Priors
        mu ~ normal(0,10000);
        //gamma ~ normal(0,10000);
        nu_2 ~ inv_gamma(1, 0.01);  // Carlin WinBUGS priors
        //nu_2_s ~ inv_gamma(1, 0.01); 
        //sigma_q ~ inv_gamma(1, 0.01);
        // The variance of the betas
        sigma_betas ~ cauchy(0,1);

       // Spatial prior
            phi_raw ~ icar_normal_lpdf(N_areas, node1, node2);
            theta ~ normal(0,sigma_theta);
            sigma_theta ~ inv_gamma(1, 0.01);
            


    // For the betas in the multilevel
    for (j in 1:J - 1){
            //beta[j] ~ normal(mu, 100000);
            beta[j] ~ normal(mu, sigma_betas[j]); // Different variance per level
            //alpha[j] ~ normal(0, 100000);
            alpha_1[j] ~ uniform(0,1);            
            //beta[j] ~ normal(0, sigma[j]); // Different variance per level
            //beta[j] ~ normal(0, 100000); // Same variance per level
    }
    // parameters for the sample
    beta[J] ~ normal(mu, sigma_betas[J]);


    // The multilevel   
    for (i in 1:N){
        // P and S with spatial random effect.
        P[i] = x[i] * beta[level[i]] + phi[ (i % N_areas)+1 ]; // This because the N_areas is half N, this assures common component      
        S[i] = x[(i % N_areas) + (N - N_areas)] * beta[J] + phi[ (i % N_areas)+1 ];
        // P and S without spatial random effect.
        P[i] = x[i] * beta[level[i]]; // This because the N_areas is half N, this assures common component      
        S[i] = x[(i % N_areas) + (N - N_areas)] * beta[J];        
        //Q[i] = (sigma_q[level[i]] * (alpha[level[i],1] * P[i] + alpha[level[i],2] * S[i])) + phi[ (i % N_areas)+1 ] ;
        Q[i] = alpha[level[i],1] * P[i] + alpha[level[i],2] * S[i] + phi[ (i % N_areas)+1 ] ;  
        }
        
    y ~ normal(Q,nu_2); 

    
    
    //y ~ bernoulli_logit(x_beta_ll);
    
    target += -3*log(sigma_phi) - 1/(sigma_phi)^2;  // Stan equiv of BUGS model prior on tau
      
}



generated quantities {
  real tau = sigma_phi^-2;
      vector[J] P[N_areas];
      for (j in 1:J) { 
          for (i  in 1:N_areas)
              //P[i][j] = inv_logit(x[i] * beta[j] + phi[i ]);
              P[i][j] = x[i] * beta[j] + phi[i ];
}

  
}
"""
    return(multispecies_model)

def multispeciesCARModel_stationaryGO():
    """
    A multispecies stationary (exact) CAR Model for Gaussian Observations.
    The multispecies presence-only model implemented in STAN.
    
    > note: This is the original code. Still underdevelopment but had worked good.
    It was used for fitting the simulation. 

    Assumptions:
        Betas come from different distributions, (mus and sigmas).
        The Spatial random effect is shared between all the processes.

    Example Usage:
	levels = np.array(simulation.index.get_level_values(0)) + 1
	MaxLevels = 2
	## seems that N_edges is defined differently here.
	N_edges = int(np.sum(NM)/2.0)
	
	N = X.shape[0]
	K = X.shape[1] 
	J = MaxLevels
	adjacency_matrix = NM  # Removed islands from M
	## number of levels to model
	
	data_multilevel_CAR = {'N' : N,
	        'K' : K,
	        'J': J,                  
	        'N_edges' : N_edges,
	        'N_areas' : X.shape[0]/J,   
	        'level': levels, # Stan counts starting at 1
	        'W' : adjacency_matrix,            
	        #'y': simulation['logit_p_sim'].values,
	        'y': simulation['q'].values,              
	        #'y': yy.simulated.values.flatten().astype('float'),
	        'x': X,
	       }
	"""

    multispecies_model_stationaryCAR_model = """
functions {
  /**
  * Return the log probability of a proper conditional autoregressive (CAR) prior 
  * with a sparse representation for the adjacency matrix
  *
  * @param phi Vector containing the parameters with a CAR prior
  * @param tau Precision parameter for the CAR prior (real)
  * @param alpha Dependence (usually spatial) parameter for the CAR prior (real)
  * @param W_sparse Sparse representation of adjacency matrix (int array)
  * @param n Length of phi (int)
  * @param W_n Number of adjacent pairs (int)
  * @param D_sparse Number of neighbors for each location (vector)
  * @param lambda Eigenvalues of D^{-1/2}*W*D^{-1/2} (vector)
  *
  * @return Log probability density of CAR prior up to additive constant
  */
  real sparse_car_lpdf(vector phi, real tau, real alpha, 
    int[,] W_sparse, vector D_sparse, vector lambda, int n, int W_n) {
      row_vector[n] phit_D; // phi' * D
      row_vector[n] phit_W; // phi' * W
      vector[n] ldet_terms;
    
      phit_D = (phi .* D_sparse)';
      phit_W = rep_row_vector(0, n);
      for (i in 1:W_n) {
        phit_W[W_sparse[i, 1]] = phit_W[W_sparse[i, 1]] + phi[W_sparse[i, 2]];
        phit_W[W_sparse[i, 2]] = phit_W[W_sparse[i, 2]] + phi[W_sparse[i, 1]];
      }
    
      for (i in 1:n) ldet_terms[i] = log1m(alpha * lambda[i]);
      return 0.5 * (n * log(tau) + sum(ldet_terms) - tau * (phit_D * phi - alpha * (phit_W * phi)));
  }
}

data {
  int<lower=0> N;       // num obs. 
  int<lower=0> J;      // number of levels
  int<lower=1> K;     // num covariates
  row_vector[K] x[N];// Size of design matrix
  int<lower=1,upper=J> level[N];  // type of level (spec)
  vector[N] y;
  
// data for the spatial structure
  int<lower=0> N_areas; // number of areas in the region.
  int<lower=0> N_edges;  // Number of pairs 
  matrix<lower = 0, upper = 1>[N_areas, N_areas] W; // adjacency matrix of lattice

}

transformed data {

// Sparse representation of W
  int<lower=0> W_n = N_edges; // just to make it compliant with the rest of the models
  int W_sparse[W_n, 2];   // adjacency pairs
  vector[N_areas] D_sparse;     // diagonal of D (number of neigbors for each site)
  vector[N_areas] lambda;       // eigenvalues of invsqrtD * W * invsqrtD  
  
  { // generate sparse representation for W
  int counter;
  counter = 1;
  // loop over upper triangular part of W to identify neighbor pairs
    for (i in 1:(N_areas - 1)) {
      for (j in (i + 1):N_areas) {
        if (W[i, j] == 1) {
          W_sparse[counter, 1] = i;
          W_sparse[counter, 2] = j;
          counter = counter + 1;
        }
      }
    }
  }  
  for (i in 1:N_areas) D_sparse[i] = sum(W[i]);
  {
    vector[N_areas] invsqrtD;  
    for (i in 1:N_areas) {
      invsqrtD[i] = 1 / sqrt(D_sparse[i]);
    }
    lambda = eigenvalues_sym(quad_form(W, diag_matrix(invsqrtD)));
  }
}


parameters {
  // Multilevel fixed effect
  //real mu[K];                // This is assuming each covariate comes from different distribution
  //real<lower=0> sigma_betas[J];    // This assumes that each level has the different distribution (variance)
  vector[K] beta[J];         // Each level has an assigend beta of K dimension.
  
  // Spatial effect
  vector[N_areas] G;         // spatial effects   
  real<lower = 0> tau;
  real<lower = 0, upper =1> alpha_car;
  real<lower=0> sigma_q;    // Unstructured random effect (variance)
  
  // Mixing effect for Q
  // The alpha parameter, one per level.
  simplex[2] alpha_1[J - 1];
}


transformed parameters { 
  simplex[2] alpha[J];

  // Define the last level (sampling effort) with no mixing effect
    for (j in 1:J - 1){
        alpha[j] = alpha_1[j];
    }
    alpha[J][1] = 0.0;
    alpha[J][2] = 1.0;
  }
  
model {
    // def variable
  vector[N] S;
  vector[N] P;
 vector[N] Q;

    // Priors for multilevel fixed effects
        // mean for betas
        //mu ~ normal(0,10000);
        // The variance of the betas
        //sigma_betas ~ inv_gamma(1,0.01); 
        
        
    // Priors for the stationary CAR
        tau ~ inv_gamma(1, 0.01);
        alpha_car ~ beta(5,5); 
        //alpha_car ~ uniform(0,1);
       
    // Spatial prior
            G ~ sparse_car(tau,alpha_car,W_sparse, D_sparse, lambda, N_areas, W_n);
            sigma_q ~ inv_gamma(1, 0.01);
            
    // Model for priors in the mixing Qs 
    // For the betas in the multilevel
    for (j in 1:J - 1){
            //beta[j] ~ normal(mu, sigma_betas[j]); // Different variance per level
            beta[j] ~ normal(0, 10000);
            //alpha_1[j] ~ uniform(0,1);
            alpha_1[j] ~ beta(5,5); 
    }
    // parameters for the sample (J is the last number of the level)
    //beta[J] ~ normal(mu, sigma_betas[J]);
    beta[J] ~ normal(0,100000);

    // The Qs   
    for (i in 0:N - 1){  //starts with 0 because we are using modulus
        // P and S with spatial random effect.
        P[i + 1] = x[i + 1] * beta[level[i + 1]] + G[ (i % N_areas)+1 ]; // This because the N_areas is half N, this assures common component      
        S[i + 1] = x[(i % N_areas + 1) + (N - N_areas)] * beta[J] + (G[ (i % N_areas)+1 ]);
        Q[i +1] = (alpha[level[i + 1],1] *  P[i + 1]) + (alpha[level[i + 1],2] * S[i + 1]);
        
        }
        

    target += normal_lpdf(y | Q, sigma_q);
    //y ~ normal(Q,sigma_q); 
   
   
}

"""
    return(multispecies_model_stationaryCAR_model)

def multispeciesCARModel_stationaryBernoulli(priors={'betas_sd':10000}):
    """
    A multispecies stationary (exact) CAR Model for Bernoulli Observations.
    The multispecies presence-only model implemented in STAN.
    
    Usage:
        The data requirements for running the model is the following:
        'N' : number of observations,
        'N_ecological_covariates' : number of covariates specified for the ecological
        suitability process. These are the first N columns of the design matrix.
        'N_sample_covariates' : number of covariates for the sampling effort. These
        are the last N_sample columns of the design matrix. 
        'J': number of levels,                  
        'N_edges' : Number of pairs in the adjacency matrix int(np.sum(W)/2.0)
        'N_areas' : Number of areas, corresponding to number of columns (row) of W.   
        'level': vector corresponding to the level (taxa) number  1
        'W' : adjacency_matrix,            
        'y': vector of observations in binary (0,1)               
        'x': design matrix of (N_ecological_covariates + N_sample_covariates) columns.

     Notes >> For fitting all covariates to both processes (Eco. suitability and
     Sample effort) assign N_sample_covariates to 0 and N_ecological_covariates = K
     where K is the total number of columns of the design matrix. This was the
     configuration used for fitting the simulation. 

    Assumptions:
        Betas come from different distributions, (mus and sigmas).
        The Spatial random effect is shared between all the processes.

    Example Usage:
	levels = np.array(simulation.index.get_level_values(0)) + 1
	MaxLevels = 2
	## seems that N_edges is defined differently here.
	N_edges = int(np.sum(NM)/2.0)
	
	N = X.shape[0]
	K = X.shape[1] 
	J = MaxLevels
	adjacency_matrix = NM  # Removed islands from M
	## number of levels to model
	
	data_multilevel_CAR = {'N' : N,
                'N_ecological_covariates' : n_eco_covs,
                'N_sample_covariates' : n_samp_covs,
	        'J': J,                  
	        'N_edges' : N_edges,
	        'N_areas' : X.shape[0]/J,   
	        'level': levels, # Stan counts starting at 1
	        'W' : adjacency_matrix,            
	        'y': data.Y.values.astype('int'), 
                'x': X,
	       }
	"""

    multispecies_model_stationaryCAR_model = """
functions {
  /**
  * Return the log probability of a proper conditional autoregressive (CAR) prior 
  * with a sparse representation for the adjacency matrix
  *
  * @param phi Vector containing the parameters with a CAR prior
  * @param tau Precision parameter for the CAR prior (real)
  * @param alpha Dependence (usually spatial) parameter for the CAR prior (real)
  * @param W_sparse Sparse representation of adjacency matrix (int array)
  * @param n Length of phi (int)
  * @param W_n Number of adjacent pairs (int)
  * @param D_sparse Number of neighbors for each location (vector)
  * @param lambda Eigenvalues of D^{-1/2}*W*D^{-1/2} (vector)
  *
  * @return Log probability density of CAR prior up to additive constant
  */
  real sparse_car_lpdf(vector phi, real tau, real alpha, 
    int[,] W_sparse, vector D_sparse, vector lambda, int n, int W_n) {
      row_vector[n] phit_D; // phi' * D
      row_vector[n] phit_W; // phi' * W
      vector[n] ldet_terms;
    
      phit_D = (phi .* D_sparse)';
      phit_W = rep_row_vector(0, n);
      for (i in 1:W_n) {
        phit_W[W_sparse[i, 1]] = phit_W[W_sparse[i, 1]] + phi[W_sparse[i, 2]];
        phit_W[W_sparse[i, 2]] = phit_W[W_sparse[i, 2]] + phi[W_sparse[i, 1]];
      }
    
      for (i in 1:n) ldet_terms[i] = log1m(alpha * lambda[i]);
      return 0.5 * (n * log(tau) + sum(ldet_terms) - tau * (phit_D * phi - alpha * (phit_W * phi)));
  }
}

data {
  int<lower=0> N;       // num obs. 
  int<lower=0> J;      // number of levels
  int<lower=0> N_ecological_covariates; // number of covariates for the eco. suit process.
  int<lower=0> N_sample_covariates; // number of covariates for the sample effort.
  row_vector[N_ecological_covariates + N_sample_covariates] x[N];// Size of design matrix
  int<lower=1,upper=J> level[N];  // type of level (spec)
  //vector[N] y;
  int<lower=0,upper=1> y[N];              // observations, in this case is binary.
// data for the spatial structure
  int<lower=0> N_areas; // number of areas in the region.
  int<lower=0> N_edges;  // Number of pairs 
  matrix<lower = 0, upper = 1>[N_areas, N_areas] W; // adjacency matrix of lattice

}

transformed data {
 // rename variables for better usage 
   int L = N_ecological_covariates;
   int M = N_sample_covariates;
   int K = L + M;
// Sparse representation of W
  int<lower=0> W_n = N_edges; // just to make it compliant with the rest of the models
  int W_sparse[W_n, 2];   // adjacency pairs
  vector[N_areas] D_sparse;     // diagonal of D (number of neigbors for each site)
  vector[N_areas] lambda;       // eigenvalues of invsqrtD * W * invsqrtD  
  
  { // generate sparse representation for W
  int counter;
  counter = 1;
  // loop over upper triangular part of W to identify neighbor pairs
    for (i in 1:(N_areas - 1)) {
      for (j in (i + 1):N_areas) {
        if (W[i, j] == 1) {
          W_sparse[counter, 1] = i;
          W_sparse[counter, 2] = j;
          counter = counter + 1;
        }
      }
    }
  }  
  for (i in 1:N_areas) D_sparse[i] = sum(W[i]);
  {
    vector[N_areas] invsqrtD;  
    for (i in 1:N_areas) {
      invsqrtD[i] = 1 / sqrt(D_sparse[i]);
    }
    lambda = eigenvalues_sym(quad_form(W, diag_matrix(invsqrtD)));
  }
}


parameters {
  // Multilevel fixed effect
  //real mu[J];                // This is assuming each covariate comes from different distribution
  //real<lower=0> sigma_betas[J];    // This assumes that each level has the different distribution (variance)

  // The splitted betas for the ecological processes
  vector[L + M] beta_eco[J];
  //vector[M] beta_samp;
  //vector[K] beta[J];         // Each level has an assigend beta of K dimension.
  
  // Spatial effect
  vector[N_areas] G;         // spatial effects   
  real<lower = 0> tau;
  real<lower = 0, upper =1> alpha_car;
  //real<lower=0> sigma_q;    // Unstructured random effect (variance)
  
  // Mixing effect for Q
  // The alpha parameter, one per level.
  simplex[2] alpha_1[J - 1];
}


transformed parameters { 
  simplex[2] alpha[J];
  vector[K] beta[J]; // Each level has an assigend beta of K dimension.
  
  // Define the last level (sampling effort) with no mixing effect
    for (j in 1:J - 1){
        alpha[j] = alpha_1[j];
    }
    alpha[J][1] = 0.0;
    alpha[J][2] = 1.0;
 
  for (j in 1:J -  1){ // Do this for the multispecies level
    for (i in 1:L){
        beta[j][i] = beta_eco[j][i];
  }
    for(i in L + 1: L + M){
            beta[j][i] = 0.0;
  }
  }
  // Assign values to covariates of the sample.
    for (i in 1:K){
         if (M == 0 && i <= L){
            beta[J][i] = beta_eco[J][ i ];
        }
        
        else if ( i > L ){
            //beta[J][i] = beta_samp[ i - L ];
            beta[J][i] = beta_eco[J][ i ];
            }

    // if number of covariates for sample effort is 0 then assume both process have
    // the same covariates        
        else {
           beta[J][i] = 0.0;
        }
  }
  }

model {
    // def variable
  vector[N] S;
  vector[N] P;
 vector[N] Q;

    // Priors for multilevel fixed effects
        // mean for betas
        //mu ~ normal(0,10000);
        // The variance of the betas
        //sigma_betas ~ inv_gamma(1,0.01); 
        
        
    // Priors for the stationary CAR
        tau ~ inv_gamma(1, 0.1);
        // a very informative one
        //tau ~ normal(2,1);
        //tau ~ cauchy(0,2);
        //alpha_car ~ beta(5,2); 
        alpha_car ~ beta(1,1); 
       
    // Spatial prior
            G ~ sparse_car(tau,alpha_car,W_sparse, D_sparse, lambda, N_areas, W_n);
            //sigma_q ~ inv_gamma(1, 0.01);
            
    // Model for priors in the mixing Qs 
    // For the betas in the multilevel
    for (j in 1:J - 1){
            //beta[j] ~ normal(mu[j], sigma_betas[j]); // Different variance per level
            //beta[j] ~ normal(0, 10000);
            // betas for ecological process
            beta_eco[j] ~ normal(0,10000);
            //beta[j] ~ normal(0, %'(betas_sd)'s);
            alpha_1[j] ~ beta(5,5); 
    }
    // parameters for the sample (J is the last number of the level)
    //beta[J] ~ normal(mu[J], sigma_betas[J]);
    //beta[J] ~ normal(0,10000);
    //beta_samp ~ normal(0,10000);
    beta_eco[J] ~ normal(0,10000);
    //beta[J] ~ normal(0, %'(betas_sd)'s);
    
    // The Qs   
    for (i in 0:N - 1){  //starts with 0 because we are using modulus
        // P and S with spatial random effect.
        P[i + 1] = x[i + 1] * beta[level[i + 1]] + G[ (i % N_areas)+1 ]; // This because the N_areas is half N, this assures common component      
        S[i + 1] = x[(i % N_areas + 1) + (N - N_areas)] * beta[J] + (G[ (i % N_areas)+1 ]);
        Q[i +1] = (alpha[level[i + 1],1] *  P[i + 1]) + (alpha[level[i + 1],2] * S[i + 1]);
        
        //Q[i + 1] = log_sum_exp(log(alpha[level[i + 1],1]) +  P[i + 1],  log(alpha[level[i + 1],2]) +  S[i + 1]);
        }
        

    y ~ bernoulli_logit(Q);
    //target += normal_lpdf(y | Q, sigma_q);
    //y ~ normal(Q,sigma_q); 
   
}


generated quantities {
 vector[N_areas] S;
 vector[N] P;
 vector[N] Q;

for (i in 0:N_areas - 1 ){
        S[i + 1] = x[(N - N_areas) + (i + 1) ] * beta[J] + (G[ i + 1 ]);
        //S[i + 1] = x[i + 1 ] * beta[J]; // + (G[ i +1 ]);
}

// The Qs   
    for (i in 0:N - 1){

        //S[i + 1] = x[(i % N_areas + 1) + (N - N_areas)] * beta[J] + (G[ (i % N_areas)+1 ]);
        P[i + 1] = x[i + 1] * beta[level[i + 1]] + G[ (i % N_areas)+1 ]; 
        Q[i +1] = (alpha[level[i + 1],1] *  P[i + 1]) + (alpha[level[i + 1],2] * S[(i % N_areas)+1]);
 }


}






"""
    return(multispecies_model_stationaryCAR_model)

def multispeciesCARModel_stationaryBernoulliMissingData(priors={'betas_sd':10000}):
    """
    A multispecies stationary (exact) CAR Model for Bernoulli Observations.
    This model supports missing data.
    The multispecies presence-only model implemented in STAN.
    
    Usage:
        The data requirements for running the model is the following:
        'N' : number of observations,
        'N_ecological_covariates' : number of covariates specified for the ecological
        suitability process. These are the first N columns of the design matrix.
        'N_sample_covariates' : number of covariates for the sampling effort. These
        are the last N_sample columns of the design matrix. 
        'J': number of levels,                  
        'N_edges' : Number of pairs in the adjacency matrix int(np.sum(W)/2.0)
        'N_areas' : Number of areas, corresponding to number of columns (row) of W.   
        'level': vector corresponding to the level (taxa) number  1
        'W' : adjacency_matrix,            
        'y': vector of observations in binary (0,1)               
        'x': design matrix of (N_ecological_covariates + N_sample_covariates) columns.
        'N_miss': Number of missing operations.

     Notes >> For fitting all covariates to both processes (Eco. suitability and
     Sample effort) assign N_sample_covariates to 0 and N_ecological_covariates = K
     where K is the total number of columns of the design matrix. This was the
     configuration used for fitting the simulation. 

    Assumptions:
        Betas come from different distributions, (mus and sigmas).
        The Spatial random effect is shared between all the processes.

    Example Usage:
	levels = np.array(simulation.index.get_level_values(0)) + 1
	MaxLevels = 2
	## seems that N_edges is defined differently here.
	N_edges = int(np.sum(NM)/2.0)
	
	N = X.shape[0]
	K = X.shape[1] 
	J = MaxLevels
	adjacency_matrix = NM  # Removed islands from M
	## number of levels to model
	
	data_multilevel_CAR = {'N' : N,
                'N_ecological_covariates' : n_eco_covs,
                'N_sample_covariates' : n_samp_covs,
	        'J': J,                  
	        'N_edges' : N_edges,
	        'N_areas' : X.shape[0]/J,   
	        'level': levels, # Stan counts starting at 1
	        'W' : adjacency_matrix,            
	        'y': data.Y.values.astype('int'), 
                'x': X,
                'N_miss' : N_miss,
	       }
	"""

    multispecies_model_stationaryCAR_model = """
        functions {
          /**
          * Return the log probability of a proper conditional autoregressive (CAR) prior 
          * with a sparse representation for the adjacency matrix
          *
          * @param phi Vector containing the parameters with a CAR prior
          * @param tau Precision parameter for the CAR prior (real)
          * @param alpha Dependence (usually spatial) parameter for the CAR prior (real)
          * @param W_sparse Sparse representation of adjacency matrix (int array)
          * @param n Length of phi (int)
          * @param W_n Number of adjacent pairs (int)
          * @param D_sparse Number of neighbors for each location (vector)
          * @param lambda Eigenvalues of D^{-1/2}*W*D^{-1/2} (vector)
          *
          * @return Log probability density of CAR prior up to additive constant
          */
          real sparse_car_lpdf(vector phi, real tau, real alpha, 
            int[,] W_sparse, vector D_sparse, vector lambda, int n, int W_n) {
              row_vector[n] phit_D; // phi' * D
              row_vector[n] phit_W; // phi' * W
              vector[n] ldet_terms;
            
              phit_D = (phi .* D_sparse)';
              phit_W = rep_row_vector(0, n);
              for (i in 1:W_n) {
                phit_W[W_sparse[i, 1]] = phit_W[W_sparse[i, 1]] + phi[W_sparse[i, 2]];
                phit_W[W_sparse[i, 2]] = phit_W[W_sparse[i, 2]] + phi[W_sparse[i, 1]];
              }
            
              for (i in 1:n) ldet_terms[i] = log1m(alpha * lambda[i]);
              return 0.5 * (n * log(tau) + sum(ldet_terms) - tau * (phit_D * phi - alpha * (phit_W * phi)));
          }
        }
        
        data {
          int<lower=0> N;       // num obs. 
          int<lower=0> J;      // number of levels
          int<lower=0> N_ecological_covariates; // number of covariates for the eco. suit process.
          int<lower=0> N_sample_covariates; // number of covariates for the sample effort.
          row_vector[N_ecological_covariates + N_sample_covariates] x[N];// Size of design matrix
          int<lower=1,upper=J> level[N];  // type of level (spec)
          int<lower=0,upper=2> y[N];              // observations, in this case is binary.
        // data for the spatial structure
          int<lower=0> N_areas; // number of areas in the region.
          int<lower=0> N_edges;  // Number of pairs 
          matrix<lower = 0, upper = 1>[N_areas, N_areas] W; // adjacency matrix of lattice
          
          int<lower=0> N_miss; // Number of missing information
          //int<lower=0> Y_miss_array[N_miss]; // array of indexed missing observations
        
        }
        
        transformed data {
         // rename variables for better usage 
           int L = N_ecological_covariates;
           int M = N_sample_covariates;
           int K = L + M;
        // Sparse representation of W
          int<lower=0> W_n = N_edges; // just to make it compliant with the rest of the models
          int W_sparse[W_n, 2];   // adjacency pairs
          vector[N_areas] D_sparse;     // diagonal of D (number of neigbors for each site)
          vector[N_areas] lambda;       // eigenvalues of invsqrtD * W * invsqrtD  
          
          { // generate sparse representation for W
          int counter;
          counter = 1;
          // loop over upper triangular part of W to identify neighbor pairs
            for (i in 1:(N_areas - 1)) {
              for (j in (i + 1):N_areas) {
                if (W[i, j] == 1) {
                  W_sparse[counter, 1] = i;
                  W_sparse[counter, 2] = j;
                  counter = counter + 1;
                }
              }
            }
          }  
          for (i in 1:N_areas) D_sparse[i] = sum(W[i]);
          {
            vector[N_areas] invsqrtD;  
            for (i in 1:N_areas) {
              invsqrtD[i] = 1 / sqrt(D_sparse[i]);
            }
            lambda = eigenvalues_sym(quad_form(W, diag_matrix(invsqrtD)));
          }
        
        
        }
        
        
        parameters {
          // Multilevel fixed effect
         
          // The splitted betas for the ecological processes
          vector[L + M] beta_eco[J];
        
          // Spatial effect
          vector[N_areas] G;         // spatial effects   
          real<lower = 0> tau;
          real<lower = 0, upper =1> alpha_car;
        
          // Mixing effect for Q
          // The alpha parameter, one per level.
          simplex[2] alpha_1[J - 1];
          
        
        }
        
        
        transformed parameters { 
          simplex[2] alpha[J];
          vector[K] beta[J]; // Each level has an assigend beta of K dimension.
          
        
          
          // Define the last level (sampling effort) with no mixing effect
            for (j in 1:J - 1){
                alpha[j] = alpha_1[j];
            }
            alpha[J][1] = 0.0;
            alpha[J][2] = 1.0;
         
          for (j in 1:J -  1){ // Do this for the multispecies level
            for (i in 1:L){
                beta[j][i] = beta_eco[j][i];
          }
            for(i in L + 1: L + M){
                    beta[j][i] = 0.0;
          }
          }
          // Assign values to covariates of the sample.
            for (i in 1:K){
                 if (M == 0 && i <= L){
                    beta[J][i] = beta_eco[J][ i ];
                }
                
                else if ( i > L ){
                    //beta[J][i] = beta_samp[ i - L ];
                    beta[J][i] = beta_eco[J][ i ];
                    }
        
            // if number of covariates for sample effort is 0 then assume both process have
            // the same covariates        
                else {
                   beta[J][i] = 0.0;
                }
          }
          
        
          
          }
        
        model {
            // def variable
          vector[N] S;
          vector[N] P;
          vector[N] Q;
        
            // Priors for multilevel fixed effects
                
            // Priors for the stationary CAR
                tau ~ inv_gamma(1, 0.1);
                // a very informative one
                alpha_car ~ beta(1,1); 
               
            // Spatial prior
                    G ~ sparse_car(tau,alpha_car,W_sparse, D_sparse, lambda, N_areas, W_n);
                    
            // Model for priors in the mixing Qs 
            // For the betas in the multilevel
            for (j in 1:J - 1){
                    // betas for ecological process
                    beta_eco[j] ~ normal(0,2);
                    alpha_1[j] ~ beta(5,5); 
            }
            // parameters for the sample (J is the last number of the level)
            beta_eco[J] ~ normal(0,2);
            
            // The Qs   
            for (i in 0:N - 1){  //starts with 0 because we are using modulus
                // P and S with spatial random effect.
                P[i + 1] = x[i + 1] * beta[level[i + 1]] + G[ (i % N_areas)+1 ]; // This because the N_areas is half N, this assures common component      
                S[i + 1] = x[(i % N_areas + 1) + (N - N_areas)] * beta[J] + (G[ (i % N_areas)+1 ]);
                Q[i +1] = (alpha[level[i + 1],1] *  P[i + 1]) + (alpha[level[i + 1],2] * S[i + 1]);
                }
                
        
          // The parsing element to support missing data. (prototype: missing-data = 2)
          for (i in 1:N){
              if (y[i] > 1) {
              
              // The marginal of y
              target += log_mix(inv_logit(Q[i]),bernoulli_logit_lpmf(1 | Q[i]),
                                                bernoulli_logit_lpmf(0 | Q[i]));
              }
              else {
              target += bernoulli_logit_lpmf(y[i] | Q[i]);
               //Y[i] = y[i];
              }
          }
        
        
            // y ~ bernoulli_logit(Q);
            // target += normal_lpdf(y | Q, sigma_q);
           
        }
        
        
        generated quantities {
         vector[N_areas] S;
         vector[N] P;
         vector[N] Q;
         int y_imp[N_miss];
         int k = 1;
        
        for (i in 0:N_areas - 1 ){
                S[i + 1] = x[(N - N_areas) + (i + 1) ] * beta[J] + (G[ i + 1 ]);
        }
        
        // The Qs   
            for (i in 0:N - 1){
                P[i + 1] = x[i + 1] * beta[level[i + 1]] + G[ (i % N_areas)+1 ]; 
                Q[i +1] = (alpha[level[i + 1],1] *  P[i + 1]) + (alpha[level[i + 1],2] * S[(i % N_areas)+1]);
         }
        
        // The missing values
            for (i in 0:N - 1){
             if (y[ i + 1 ] > 1) {
                 y_imp[k] = bernoulli_logit_rng(Q[i + 1]);
                 k += 1;
                 
             }
            
                
         }
        
        }
"""
    return(multispecies_model_stationaryCAR_model)




def multispeciesCARModel_stationaryBinomial(priors={'betas_sd':'10000'}):
    """
    A multispecies stationary (exact) CAR Model for Gaussian Observations.
    The multispecies presence-only model implemented in STAN.
    
    It was used for fitting the simulation. 

    Assumptions:
        Betas come from different distributions, (mus and sigmas).
        The Spatial random effect is shared between all the processes.

    Example Usage:
	levels = np.array(simulation.index.get_level_values(0)) + 1
	MaxLevels = 2
	## seems that N_edges is defined differently here.
	N_edges = int(np.sum(NM)/2.0)
	
	N = X.shape[0]
	K = X.shape[1] 
	J = MaxLevels
	adjacency_matrix = NM  # Removed islands from M
	## number of levels to model
	
	data_multilevel_CAR = {'N' : N,
	        'K' : K,
	        'J': J,                  
	        'N_edges' : N_edges,
	        'N_areas' : X.shape[0]/J,   
	        'level': levels, # Stan counts starting at 1
	        'W' : adjacency_matrix,            
	        #'y': simulation['logit_p_sim'].values,
	        'y': simulation['q'].values,              
	        #'y': yy.simulated.values.flatten().astype('float'),
	        'x': X,
	       }
	"""

    multispecies_model_stationaryCAR_model = """
functions {
  /**
  * Return the log probability of a proper conditional autoregressive (CAR) prior 
  * with a sparse representation for the adjacency matrix
  *
  * @param phi Vector containing the parameters with a CAR prior
  * @param tau Precision parameter for the CAR prior (real)
  * @param alpha Dependence (usually spatial) parameter for the CAR prior (real)
  * @param W_sparse Sparse representation of adjacency matrix (int array)
  * @param n Length of phi (int)
  * @param W_n Number of adjacent pairs (int)
  * @param D_sparse Number of neighbors for each location (vector)
  * @param lambda Eigenvalues of D^{-1/2}*W*D^{-1/2} (vector)
  *
  * @return Log probability density of CAR prior up to additive constant
  */
  real sparse_car_lpdf(vector phi, real tau, real alpha, 
    int[,] W_sparse, vector D_sparse, vector lambda, int n, int W_n) {
      row_vector[n] phit_D; // phi' * D
      row_vector[n] phit_W; // phi' * W
      vector[n] ldet_terms;
    
      phit_D = (phi .* D_sparse)';
      phit_W = rep_row_vector(0, n);
      for (i in 1:W_n) {
        phit_W[W_sparse[i, 1]] = phit_W[W_sparse[i, 1]] + phi[W_sparse[i, 2]];
        phit_W[W_sparse[i, 2]] = phit_W[W_sparse[i, 2]] + phi[W_sparse[i, 1]];
      }
    
      for (i in 1:n) ldet_terms[i] = log1m(alpha * lambda[i]);
      return 0.5 * (n * log(tau) + sum(ldet_terms) - tau * (phit_D * phi - alpha * (phit_W * phi)));
  }
}

data {
  int<lower=0> N;       // num obs. 
  int<lower=0> J;      // number of levels
  int<lower=1> K;     // num covariates
  row_vector[K] x[N];// Size of design matrix
  int N_trials;     // Number of trials (binomial case)
  int<lower=1,upper=J> level[N];  // type of level (spec)
  //vector[N] y;
  int<lower=0,upper=N_trials> y[N];              // observations, in this case is binary.
// data for the spatial structure
  int<lower=0> N_areas; // number of areas in the region.
  int<lower=0> N_edges;  // Number of pairs 
  matrix<lower = 0, upper = 1>[N_areas, N_areas] W; // adjacency matrix of lattice

}

transformed data {

// Sparse representation of W
  int<lower=0> W_n = N_edges; // just to make it compliant with the rest of the models
  int W_sparse[W_n, 2];   // adjacency pairs
  vector[N_areas] D_sparse;     // diagonal of D (number of neigbors for each site)
  vector[N_areas] lambda;       // eigenvalues of invsqrtD * W * invsqrtD  
  
  { // generate sparse representation for W
  int counter;
  counter = 1;
  // loop over upper triangular part of W to identify neighbor pairs
    for (i in 1:(N_areas - 1)) {
      for (j in (i + 1):N_areas) {
        if (W[i, j] == 1) {
          W_sparse[counter, 1] = i;
          W_sparse[counter, 2] = j;
          counter = counter + 1;
        }
      }
    }
  }  
  for (i in 1:N_areas) D_sparse[i] = sum(W[i]);
  {
    vector[N_areas] invsqrtD;  
    for (i in 1:N_areas) {
      invsqrtD[i] = 1 / sqrt(D_sparse[i]);
    }
    lambda = eigenvalues_sym(quad_form(W, diag_matrix(invsqrtD)));
  }
}


parameters {
  // Multilevel fixed effect
  //real mu[K];                // This is assuming each covariate comes from different distribution
  //real<lower=0> sigma_betas[J];    // This assumes that each level has the different distribution (variance)
  vector[K] beta[J];         // Each level has an assigend beta of K dimension.
  
  // Spatial effect
  vector[N_areas] G;         // spatial effects   
  real<lower = 0> tau;
  real<lower = 0, upper =1> alpha_car;
  //real<lower=0> sigma_q;    // Unstructured random effect (variance)
  
  // Mixing effect for Q
  // The alpha parameter, one per level.
  simplex[2] alpha_1[J - 1];
}


transformed parameters { 
  simplex[2] alpha[J];

  // Define the last level (sampling effort) with no mixing effect
    for (j in 1:J - 1){
        alpha[j] = alpha_1[j];
    }
    alpha[J][1] = 0.0;
    alpha[J][2] = 1.0;
  }
  
model {
    // def variable
  vector[N] S;
  vector[N] P;
 vector[N] Q;

    // Priors for multilevel fixed effects
        // mean for betas
        //mu ~ normal(0,10000);
        // The variance of the betas
        //sigma_betas ~ inv_gamma(1,0.01); 
        
        
    // Priors for the stationary CAR
        tau ~ inv_gamma(1, 0.01);
        alpha_car ~ beta(5,5); 
        //alpha_car ~ uniform(0,1);
       
    // Spatial prior
            G ~ sparse_car(tau,alpha_car,W_sparse, D_sparse, lambda, N_areas, W_n);
            //sigma_q ~ inv_gamma(1, 0.01);
            
    // Model for priors in the mixing Qs 
    // For the betas in the multilevel
    for (j in 1:J - 1){
            //beta[j] ~ normal(mu, sigma_betas[j]); // Different variance per level
            beta[j] ~ normal(0, 10000);
            beta[j] ~ normal(0, 10000);
            //alpha_1[j] ~ uniform(0,1);
            alpha_1[j] ~ beta(5,5); 
    }
    // parameters for the sample (J is the last number of the level)
    //beta[J] ~ normal(mu, sigma_betas[J]);
    beta[J] ~ normal(0,100000);

    // The Qs   
    for (i in 0:N - 1){  //starts with 0 because we are using modulus
        // P and S with spatial random effect.
        P[i + 1] = x[i + 1] * beta[level[i + 1]] + G[ (i % N_areas)+1 ]; // This because the N_areas is half N, this assures common component      
        S[i + 1] = x[(i % N_areas + 1) + (N - N_areas)] * beta[J] + (G[ (i % N_areas)+1 ]);
        Q[i +1] = (alpha[level[i + 1],1] *  P[i + 1]) + (alpha[level[i + 1],2] * S[i + 1]);
        
        //Q[i +1] = log_sum_exp(log(alpha[level[i + 1],1]) +  P[i + 1],  log(alpha[level[i + 1],2]) +  S[i + 1]);
        }
        

    //y ~ bernoulli_logit(Q);
    y ~ binomial_logit(N_trials,Q);
    //target += normal_lpdf(y | Q, sigma_q);
    //y ~ normal(Q,sigma_q); 
   
   
}

"""
    return(multispecies_model_stationaryCAR_model)
