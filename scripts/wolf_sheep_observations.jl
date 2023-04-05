#=============================================================================================#
# # Create simulated observations
#=============================================================================================#

using DrWatson
@quickactivate "ABC_ABM"

using Distributions
using DataFrames
using FileIO
using JLD2
using Pipe
using Agents


# ### Load functions from src
#
include(srcdir("predator_prey.jl"))  # ABM functions
agent_data = [(sheep, count), (wolf, count)] # ABM data to collect

include(srcdir("plot_functions.jl")) # plot functions

# ### Save model parameters 
#
parameters="stable" # name tag for model parameters

# The following result in a landscape where sheep, wolves and grass coexist.
let model_parameters = @strdict(
    n_sheep = 200,
    n_wolves = 50,
    dims = (50, 50),
    Δenergy_sheep = 5,
    sheep_reproduce = 0.31,
    Δenergy_wolf = 30,
    wolf_reproduce = 0.06)
    safesave(datadir("sims","observations","parameters_$(parameters).jld2"), model_parameters)
end

# ## Repeated simulations
#----------------------------------------------------------------------------------------------------
Nstep = 5000  # number of steps
Nsim = 100    # number of replicates
simulation_config = @dict Nstep Nsim parameters

simulations = produce_or_load(datadir("sims","observations"), 
                                    simulation_config,
                                    suffix="csv") do simulation_config
                                        @unpack Nstep, Nsim, parameters = simulation_config
                                        model_parameters = load(datadir("sims","observations","parameters_$(parameters).jld2")) |> tosymboldict
                                        model_parameters[:seed] = Vector{UInt16}(420:(420+Nsim-1)) #make & save seed for every sim
                                        agents_sim, _ = paramscan(model_parameters, initialize_model; 
                                                                    agent_step! = sheepwolf_step!, 
                                                                    model_step! = grass_step!, 
                                                                    adata = agent_data, 
                                                                    n=Nstep, 
                                                                    showprogress=true)
                                        agents_sim
                                    end |> first |> DataFrame


# ### plot variance
#
agents_var = @pipe simulations |> groupby(_,:step) |> combine(_, [:count_wolf, :count_sheep] .=> var) 

fig_agents_var = let save_plot = false
    fig, ax, _ = lines(agents_var.count_sheep_var, label="sheep_count", axis=(;xlabel="step", ylabel="variance"))
    lines!(ax, agents_var.count_wolf_var, label="wolf_count")
    axislegend()
    if save_plot
        @pipe  savename("variance", simulation_config, "png") |> plotsdir("observations",_) |> safesave(_, fig)
    end
    fig
end


# ## Select and save observations
#----------------------------------------------------------------------------------------------------
# constraints: 
# - we know it is a "stable system" (-> in "equilibrium")
# - we only have a snapshot of observations; there has been co-existence for a while

seed_obs = 429   # select one from the repeated simulations (from the previous section)
Nstep_obs = 1000 # length of "observed" data + 1 (starts at step 0)

observations =  simulations[simulations.seed .== seed_obs,2:3][end-Nstep_obs:end,:] # select last steps from the simulation
observations[!, :step] = 0:Nstep_obs # reset observed step count

fig_obs = plot_population_timeseries(observations)

plot_population_timeseries(simulations_selection[1:1000,:]) # plot transient
plot_population_timeseries(simulations_selection[simulations_selection.step .>= (Nstep_sim - Nstep_obs),:]) # plot equilibrium

@pipe (["svg","png"] .|> savename("observed_data",
            (;nstep=Nstep_obs, seed = seed_obs), _ ) .|> 
            plotsdir("observations", _ ) .|> 
            safesave(_ ,fig_obs));

