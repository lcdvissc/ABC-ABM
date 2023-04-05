#=============================================================================================#
# # Construct summary statistics using PLS
#=============================================================================================#

using DrWatson
@quickactivate "ABC_ABM"

using DataFrames, FileIO, Pipe
using Statistics, StatsBase, Random
using CairoMakie
using Distances, LinearAlgebra
using SingularSpectrumAnalysis, FFTW

import MLJ: MLJ, Standardizer, Pipeline #, TunedModel, UnivariateBoxCoxTransformer
import PartialLeastSquaresRegressor: PLSRegressor, PLS2Model


include(srcdir("rejection_ABC.jl")); 
include(srcdir("plot_functions.jl")); 

# ## Load prior simulations
#---------------------------------------------------------------------------------------------#

prior_id = 1             # prior settings identifier 
n_prior_samples = 1000   # simulation batch size
n_repetitions = 1        # number of repetitions
steps_transient = 500    # transient steps (to be cut off from prior simulations)
n_steps_obs = 1000
n_steps_max = n_steps_obs + steps_transient

prior_info = @dict(samples = n_prior_samples, rep = n_repetitions, steps = n_steps_max)
prior_simulations, θ_prior_samples = @pipe ("sims", "params") .|> 
                                    savename( _ , prior_info ,"csv") .|> 
                                    datadir("sims","priors_$(prior_id)", _) .|>
                                    load .|> DataFrame
θ_prior_samples[!,:ensemble] = prior_simulations.ensemble |> unique



# ## Pre-process prior simulations
#---------------------------------------------------------------------------------------------#

# filter out extinctions
go_extinct(x::AbstractVector) = last(x) == 0
coexist(s::AbstractVector, w::AbstractVector) = ~go_extinct(s) & ~go_extinct(w)
nonconstant_TS(s::AbstractVector, w::AbstractVector) = ~any( ([s,w] .|> unique .|> length) .== 1)

prior_sim_grouped = groupby(prior_simulations,:ensemble)
summaries_extinctions = combine(prior_sim_grouped, 
                        nrow => :steps , 
                        :count_sheep => go_extinct => :sheep_die, 
                        :count_wolf => go_extinct => :wolves_die,
                        [:count_sheep,:count_wolf] => coexist => :coexist,
                        [:count_sheep,:count_wolf] => nonconstant_TS => :nonconstant_TS)
                        
println("total sheep extinction probability: ", sum(summaries_extinctions.sheep_die)/(n_prior_samples*n_repetitions))
println("total wolf  extinction probability: ", sum(summaries_extinctions.wolves_die)/(n_prior_samples*n_repetitions))
println("total coexistence      probability: ", sum(summaries_extinctions.coexist)/(n_prior_samples*n_repetitions))

filter_simulations = summaries_extinctions.coexist .& summaries_extinctions.nonconstant_TS
prior_simulations_filtered = @pipe DataFrame(prior_sim_grouped[filter_simulations]) |> subset(_, :step => s -> s.> steps_transient)

# plot some simulations
let s = :count_sheep,  plot_ensembles =  θ_prior_samples[summaries_extinctions.coexist, :ensemble][1:10]
        plot_n_agent_timeseries(groupby(prior_simulations_filtered, :ensemble),s, plot_ensembles )
end


# ## Compute summary statsitics
#---------------------------------------------------------------------------------------------#
summaries_counts_symb = [:count_sheep,:count_wolf]

lag_max = n_steps_obs ÷ 2
L = 30


summaries_config = @dict(lag = lag_max, L, stepstrans=steps_transient)

path = datadir("sims","priors_$(prior_id)","summaries", "machines")
summaries_TS = produce_or_load(path, summaries_config , filename = savename("time_series_stats", merge(summaries_config,prior_info) ,"csv"), tag=false) do machine_config
        time_series_summaries(groupby(prior_simulations_filtered,:ensemble), summaries_config)
        end |> first |> DataFrame
                                                                                

# ## Train PLS on a set of simulations
#---------------------------------------------------------------------------------------------#                                                                                
X_df = (:ensemble ∈ propertynames(summaries_TS)) ? select!(summaries_TS, Not(:ensemble)) : summaries_TS
y_df = θ_prior_samples[filter_simulations, [:sheep_reproduce, :wolf_reproduce]]

partitionseed = 420
trainfraction = 0.9

train, test = MLJ.partition(1:sum(filter_simulations), trainfraction, shuffle=true, rng=Random.MersenneTwister(partitionseed));

maxfactors = 100
pls_model = PLSRegressor(n_factors=maxfactors)
PLS_machine = MLJ.machine(pls_model, X_df, y_df)
@time MLJ.fit!(PLS_machine, rows=train, verbosity=1)

m = PLS_machine.fitresult
partitions = @dict train test

#= log
7.130878 seconds (14.88 M allocations: 1.585 GiB, 4.36% gc time, 91.08% compilation time: 4% of which was recompilation)
 =#

machine_config = merge(summaries_config, prior_info,
                @dict(seed = partitionseed, trainfraction, maxfactors=maxfactors, summaries="autocor_fft_SSA_diststat"))

let part = :test, n_latent_vars = maxfactors, savefig = true
        i_plot = partitions[part]
        PLS_cor = latent_variable_cor(m, X_df[i_plot,:],y_df[i_plot,:],n_latent_vars)
        fig = Figure()
        θ = propertynames(y_df)
        fig[1,1] = Axis(fig, ylabel="ρ $(θ[1])")
        fig[2,1] = Axis(fig, xlabel="latent variables",ylabel="ρ  $(θ[2])")
        series!(fig[1,1],[PLS_cor[:,1]], markersize=5, solid_color=:black)
        series!(fig[2,1],[PLS_cor[:,2]], markersize=5, solid_color=:black)

        if savefig
                @pipe (["svg","png"] .|> savename("PLS_performance_$(part)", machine_config, _ ) .|> 
                plotsdir("priors_$(prior_id)",
                        "raw", _) .|> 
                safesave(_ ,fig));
        end
        fig
end
nfactors = 38
PLS_machine.model.n_factors = PLS_machine.fitresult.nfactors = nfactors # set PLS factors to chosen value
merge!(machine_config, @dict(nfactors))
y_hat = @pipe MLJ.predict(PLS_machine, X_df) |> DataFrame |> rename!(_, propertynames(y_df))

# plot PLS predictions
let part = :train, savefig = true
        i_plot = partitions[part]
        fig = Figure(resolution=(600,1200))
        fig[1,1] = Axis(fig, xlabel="true wolf_reproduce", ylabel="PLS estimate wolf_reproduce")
        fig[1,2] = Axis(fig, xlabel="true sheep_reproduce", ylabel="PLS estimate sheep_reproduce")   
        fig[2,1] = Axis(fig, ylabel="true wolf_reproduce", xlabel="true sheep_reproduce", title="PLS estimate wolf_reproduce") 
        fig[2,2] = Axis(fig, ylabel="true wolf_reproduce", xlabel="true sheep_reproduce", title="PLS estimate sheep_reproduce")   
        fig[3:4,1:2] = Axis(fig, ylabel="true wolf_reproduce", xlabel="true sheep_reproduce", title="PLS estimate (normalized sum)")
        scatter!(fig[1,1],y_df[i_plot,:wolf_reproduce], y_hat[i_plot,:wolf_reproduce])
        scatter!(fig[1,2],y_df[i_plot,:sheep_reproduce], y_hat[i_plot,:sheep_reproduce])
        scatter!(fig[2,1],y_df[i_plot,:sheep_reproduce], y_df[i_plot,:wolf_reproduce], color = y_hat[i_plot,:wolf_reproduce])
        scatter!(fig[2,2],y_df[i_plot,:sheep_reproduce], y_df[i_plot,:wolf_reproduce], color = y_hat[i_plot,:sheep_reproduce])
        scatter!(fig[3:4,1:2],y_df[i_plot,:sheep_reproduce], y_df[i_plot,:wolf_reproduce], color = normalize(y_hat[i_plot,:sheep_reproduce]) .+ normalize(y_hat[i_plot,:wolf_reproduce]), markersize=10 )
        
        if savefig
        @pipe (["svg","png"] .|> savename("PLS_predictions_$(part)", machine_config, _ ) .|> 
                plotsdir("priors_$(prior_id)",
                        "raw", _) .|> 
                safesave(_ ,fig));
        end
        fig 
end


let n_factors = nfactors, sum_stat_i = 1:size(X_df,2) #@pipe names(X_df) .|> occursin("F_abs_sheep",_) |> length
        fig = Figure()
        θ = propertynames(y_df)
        fig[1,1] = Axis(fig, ylabel="loadings $(θ[1])")
        fig[2,1] = Axis(fig, ylabel="loadings $(θ[2])")
        lns = []
        
        for i in 1:n_factors
                loadings = m.W[:,i]*m.Q[:,i]'
                lines!(fig[1,1],loadings[sum_stat_i,1])
                l = lines!(fig[2,1],loadings[sum_stat_i,2])
                push!(lns,l)
        end
        Legend(fig[1:2,2],lns, ["fac $(i)" for i in 1:n_factors], nbanks= 3)
        fig
end

# save best machine
machine_path = datadir("sims","priors_$(prior_id)","machines")
@pipe machine_path |> joinpath(_, "PLS_machine_config.jld2") |>
        tagsave(_, Dict(machine_config...), safe=true)  # remark: symbol keys converted to strings
@pipe savename("PLS_machine",machine_config,"jls") |> joinpath(machine_path, _) |> 
        MLJ.save(_,PLS_machine)












