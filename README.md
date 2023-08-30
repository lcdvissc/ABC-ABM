# ABC-ABM

Code that was used for the illustrative examples in "A critical review of common pitfalls and guidelines to effectively infer parameters of agent-based models using Approximate Bayesian Computation".

## Installation
This code base is using the Julia Language and [DrWatson](https://juliadynamics.github.io/DrWatson.jl/stable/)
to make a reproducible scientific project named
> ABC-ABM

To (locally) reproduce this project, do the following:

0. Download this code base. Notice that raw data are typically not included in the
   git-history and may need to be downloaded independently.
1. Open a Julia console and do:
   ```
   julia> using Pkg
   julia> Pkg.add("DrWatson") # install globally, for using `quickactivate`
   julia> Pkg.activate("path/to/this/project")
   julia> Pkg.instantiate()
   ```

This will install all necessary packages for you to be able to run the scripts and
everything should work out of the box, including correctly finding local paths.

## Overview
The code in this project is divided in source code and scripts.
The source code, located in the `src` directory, contains custom functions used in the scripts.
This includes:
- `plot_functions.jl` : all functions used for generating plots
- `predator_prey.jl` : ABM from the `Agents.jl` library
- `rejection_ABC.jl` : all functions used for performing and evaluating ABC

Two scripts are purely used for perfomring simulations with the ABM:
- `wolf_sheep_observations.jl` : script for generating the "observed" data
- `wolf_sheep_prior_observations.jl` : script for drawing a batch of prior simulations

The remaining follow the ABC analysis as described in the paper as follows:
1. `prior_elicitation.jl` : determines the duration of the transient phase 
2. `train_summary_statistics.jl` : constructs the summaries using PLS
3. `summary_statistics.jl`: compares the posteriors of different summary statistics using cross-validation
4. `posterior_validation.jl`: performs the final rejection ABC iteration and do a 

Between steps 1-2 and 3-4 the priors are refined based on the extenction events (using script `prior_elicitation.jl` and `wolf_sheep_prior_observations.jl`).