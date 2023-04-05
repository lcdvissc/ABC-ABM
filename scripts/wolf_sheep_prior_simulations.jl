using DrWatson
@quickactivate "ABC_ABM"

using Distributions
using DataFrames
using StatsBase
using FileIO
using CairoMakie
using Pipe

using Distributed
addprocs(4) # add cores for parallel computation

@everywhere using DrWatson
@everywhere @quickactivate "ABC_ABM"

# ## Load functions from src
@everywhere include(srcdir("predator_prey.jl")) 
# ## Define datacollection
@everywhere adata = [(sheep, count), (wolf, count)]


# load observational data
obs_data = datadir("sims","observations", "observed_data.csv") |> load |> DataFrame
steps_obs = maximum(obs_data.step)


# ## Prior Distributions
# Fixed model/simulation parameters
priors_id = 2

steps_transient = 500 # buffer for transient phase (see: summary_statistics.jl)
steps = steps_obs + steps_transient

fixed_params = (;
    dims = (50, 50),
    Δenergy_sheep = 5, 
    Δenergy_wolf = 30,
)


# prior estimation (see: summary_statistics.jl)
priors = (;
wolf_reproduce  = Beta(1.59861, 20.1555),
sheep_reproduce = Beta(1.81811, 3.28002)
)



# ## Prior Samples
n_batches = 10              # simulate in multiple batches
n_prior_samples = 10000     # samples/batch
n_repetitions = 1           # repetitions per sample
for batch in 10:n_batches

    println(batch)
    Δensemble = (batch-1)*n_prior_samples*n_repetitions
    n_sheep = zeros(Int64,n_prior_samples)
    n_wolves = zeros(Int64,n_prior_samples)
    for i in 1:n_prior_samples
        i_init = rand(1:steps_obs) 
        n_sheep[i], n_wolves[i] = obs_data[i_init,:]
    end

    prior_samples_df = DataFrame([θ => repeat(rand(π₀, n_prior_samples), inner=n_repetitions) for (θ, π₀) in pairs(priors)])
    prior_samples_df[!,:n_sheep] = repeat(n_sheep,inner=n_repetitions)
    prior_samples_df[!,:n_wolves] = repeat(n_wolves, inner=n_repetitions)
    prior_samples_df[!,:seed] = rand(UInt16, n_prior_samples*n_repetitions)


    # ## Prior Simulations
    # run ensemble simulations
    models = @time [initialize_model(;fixed_params..., prior_samples_df[i,:]...) for i in 1:n_prior_samples*n_repetitions]
    sim_res_df, _ = @time ensemblerun!(models, sheepwolf_step!, grass_step!, (model,s) -> stop_if_extinct(model,s,s_max=steps);
                                        adata, parallel = true, showprogress = true)
    
    file_config = @dict(samples=n_prior_samples,rep=n_repetitions,steps=steps,batch)

    # save prior samples
    prior_samples_df[!,:id] = repeat((batch-1)*n_prior_samples+1:batch*n_prior_samples,inner=n_repetitions)  
    prior_samples_df[!,:ensemble] = Δensemble+1:Δensemble+n_prior_samples
    @pipe savename("params", file_config ,"csv") |>
                datadir("sims", "priors_$(priors_id)", _ ) |>
                safesave(_, prior_samples_df)                                                                    
                                    

    # save prior simulations
    sim_res_df[!,:id] = prior_samples_df.id[sim_res_df.ensemble]
    sim_res_df[!,:ensemble] .+= Δensemble
    @pipe savename("sims", file_config ,"csv") |>
                datadir("sims", "priors_$(priors_id)", _ ) |>
                safesave(_, sim_res_df)
end

#= log
39.338177 seconds (280.18 M allocations: 12.051 GiB, 27.10% gc time, 0.59% compilation time: 15% of which was recompilation)
424.060248 seconds (68.53 M allocations: 3.273 GiB, 15.72% gc time, 0.45% compilation time)
=#

# merge multiple batches

load_batch(data::String, file_info::Dict, priors_id::Int) = @pipe savename( data , file_info ,"csv") |> datadir("sims","uninformed_start","priors_$(priors_id)", _) |> load |> DataFrame 
 
θ_prior_samples = @time vcat([load_batch("params",@dict(samples=n_prior_samples,rep=n_repetitions,steps=steps,batch), priors_id) for batch in 1:n_batches]...)
prior_simulations = @time vcat([load_batch("sims",@dict(samples=n_prior_samples,rep=n_repetitions,steps=steps,batch), priors_id) for batch in 1:n_batches]...)

merged_file_info = @dict(samples=n_prior_samples*n_batches,rep=n_repetitions,steps=steps)
@pipe savename("sims" , merged_file_info ,"csv") |> 
      datadir("sims","priors_$(priors_id)", _) |>
      safesave(_, prior_simulations)

@pipe savename("params" , merged_file_info ,"csv") |> 
      datadir("sims","priors_$(priors_id)", _) |>
      safesave(_, θ_prior_samples)
                                                              
                                    