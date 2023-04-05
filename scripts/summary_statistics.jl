#=============================================================================================#
# # Test discrepancy measures & summary statistics
#=============================================================================================#

using DrWatson
@quickactivate "ABC_ABM"

using DataFrames, FileIO, Pipe
using CairoMakie, Colors
using Distributions
using StatsBase, Distances, DynamicAxisWarping
using Random


include(srcdir("rejection_ABC.jl")); 
include(srcdir("plot_functions.jl")); 

# ## Load prior simulations
#---------------------------------------------------------------------------------------------#

prior_id = 2             # prior settings identifier 
n_prior_samples = 10000  # simulation batch size
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
θ_symbols = [:sheep_reproduce, :wolf_reproduce]


# ## Pre-process prior simulations
#---------------------------------------------------------------------------------------------#

# filter out extinctions
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
let s = :count_sheep,  plot_ensembles =  θ_prior_samples[summaries_extinctions.coexist, :ensemble][1:20]
        plot_n_agent_timeseries(groupby(prior_simulations_filtered, :ensemble),s, plot_ensembles )
end



# re-fit prior
p_ABC = Dict(
    :wolf_reproduce  => fit_mle(Beta, θ_prior_samples.wolf_reproduce[filter_simulations]),
    :sheep_reproduce => fit_mle(Beta, θ_prior_samples.sheep_reproduce[filter_simulations])
    )

let θ = :sheep_reproduce, save_plot = false
    fig = hist(θ_prior_samples[filter_simulations, θ], normalization =:pdf, label="accepted", axis=(;xlabel="$(θ)", ylabel="density"))
    @pipe LinRange(0, 1. ,1000) |> collect |> lines!( fig.axis, _ , pdf.(p_ABC[θ], _ ), label="fitted β distribution")
    axislegend()


    if save_plot
        @pipe (["svg","png"] .|> savename("posterior_coexistence",
        (;par=θ,
        samples=n_prior_samples,
        rep=n_repetitions,
        steps=n_steps_max), _ ) .|> 
        plotsdir("priors_$(prior_id)", _) .|> 
        safesave(_ ,fig));
    end
    fig
end


# ## Compute summary statsitics
#---------------------------------------------------------------------------------------------#
summaries_counts_symb = [:count_sheep,:count_wolf]

moments_to_compute = [s => f for (s,f) in Iterators.product(summaries_counts_symb, [mean, std, skewness, kurtosis]) |> collect |> vec]
summaries_moments = @time @pipe prior_simulations_filtered |> groupby(_, :ensemble) |> 
                                combine(_ , moments_to_compute...)

# load last PLS machine configuration
PLS_prior_id = 1
machine_path = datadir("sims","priors_$(PLS_prior_id)","machines") 
machine_config = @pipe machine_path |> joinpath(_, "PLS_machine_config.jld2") |> load 
println("loading PLS machine, trained in git commit: \n",pop!(machine_config, "gitcommit"))
PLS_machine = @pipe savename("PLS_machine",machine_config,"jls") |> joinpath(machine_path, _) |> MLJ.machine

@assert steps_transient == machine_config["stepstrans"]

lag_max = machine_config["lag"]
L = machine_config["L"]

summaries_config = @dict(lag = lag_max, L, stepstrans=steps_transient)
path = datadir("sims","priors_$(prior_id)", "summaries")
summaries_TS = produce_or_load(path, summaries_config , 
                                filename = savename(machine_config["summaries"], 
                                merge(summaries_config,prior_info) ,"csv"), tag=false) do config
        time_series_summaries(groupby(prior_simulations_filtered,:ensemble), config)
        end |> first |> DataFrame

# remove :ensemble from summary data (if necessary)
X_df = (:ensemble ∈ propertynames(summaries_TS)) ? select!(summaries_TS, Not(:ensemble)) : summaries_TS
summaries_PLS = @pipe MLJ.predict(PLS_machine, X_df) |> DataFrame |> rename!(_, θ_symbols)

summaries_PLS[!, :ensemble] = summaries_extinctions.ensemble[filter_simulations]

# ## Define discrepancy measures
#---------------------------------------------------------------------------------------------#
σ_π₀(x::AbstractVector) = 1/std(x)

eucl_weights_counts = combine(groupby(prior_simulations_filtered, :step), summaries_counts_symb .=> σ_π₀ .=> summaries_counts_symb);
summaries_counts_D = @dict(
        WSE = [WeightedEuclidean(eucl_weights_counts[!,s]) for s in summaries_counts_symb],
        DTW5 = DTW(dist=SqEuclidean(),radius=5),
        DTW100 = DTW(dist=SqEuclidean(),radius=100),
        DTW250 = DTW(dist=SqEuclidean(),radius=250))

eucl_weights_moments = @pipe select(summaries_moments, Not(:ensemble)) |> combine(_, propertynames(_) .=> σ_π₀ .=> propertynames(_))
summaries_moments_D = @dict( WSE = [values(eucl_weights_moments[1,:])...] |> WeightedEuclidean)


# ## compute count distances in LOOCV procedure
#---------------------------------------------------------------------------------------------#
n_LOOCV = 100 # number of LOOCV samples
i_obs_seed = 42    # seed for picking pseudo-observations
D = :WSE
summaries = :moments
config = merge(prior_info, @dict(D, seed = i_obs_seed, nLOOCV=n_LOOCV))

# dirty code, needs rework
path = datadir("sims","priors_$(prior_id)","distances")
distances_df = produce_or_load(path, config , filename = savename("$(summaries)", config,"csv"), tag=false) do config
        i_obs_vec = sample(Random.MersenneTwister(config[:seed]), θ_prior_samples[summaries_extinctions.coexist, :ensemble], config[:nLOOCV], replace = false)
        
        if summaries == :counts
                @time alldistances_LOOCV(groupby(prior_simulations_filtered, :ensemble),
                        (sim,obs) -> total_discrepancy(summaries_counts_D[config[:D]], sim, obs; summaries = summaries_counts_symb, keepkeys=true), 
                        config[:nLOOCV],
                        i_obs_vec)
        elseif summaries == :moments
                @time alldistances_LOOCV(summaries_moments, summaries_moments_D[config[:D]], 
                        config[:nLOOCV],
                        i_obs_vec)
        elseif summaries == :PLS
                @time alldistances_LOOCV(summaries_PLS, SqEuclidean(), 
                        config[:nLOOCV],
                        i_obs_vec)
        else
                error("use 'counts','moments' or 'PLS' as summary stats ")
        end
        end |> first |> DataFrame

distances_df = innerjoin(distances_df, θ_prior_samples, on=:ensemble) 
i_obs_vec = distances_df.i_obs |> unique

#= logs
WSE     nLOOCV=100_rep=1_samples=1000_seed=42_steps=1500    :    3.168440 seconds (2.45 M allocations: 725.477 MiB, 4.07% compilation time)
DTW100  nLOOCV=100_rep=1_samples=1000_seed=42_steps=1500    :    259.637030 seconds (2.83 M allocations: 1.212 GiB, 0.05% compilation time)
=#


# visualize distances
#= 
let gdf = groupby(count_distances_df, :i_obs), θ = :sheep_reproduce, n_accepted = 100
        with_theme(Theme(markersize=5)) do

                fig = Figure()
                fig[1,1] = Axis(fig, title="$(θ)", ylabel="dev from true") #, limits = (0, nothing, -0.5,0.5))
                sorted_i = @pipe subset(θ_prior_samples, :ensemble => ByRow(x -> x ∈ i_obs_vec )) |> sort(_, θ).ensemble
                for i_obs in sorted_i[1:10]
                        scatter!(fig[1,1], gdf[(i_obs,)][1:n_accepted,:dist],(gdf[(i_obs,)][1:n_accepted,θ] .- θ_prior_samples[i_obs,θ]), label = "$(round(θ_prior_samples[i_obs,θ]; digits = 2))", alpha=0.3)
                end
                axislegend()
                fig
        end
end
 =#



# ## rejection sampling with selected distances and summ_stat
#---------------------------------------------------------------------------------------------#
q_ϵ = 0.1
n_prior_LOOCV = sum(filter_simulations) - 1 # number of effective prior samples, in LOOCV
n_post_samples_LOOCV = q_ϵ * n_prior_LOOCV |> floor |> Int


rejection_distances = combine(groupby(distances_df, :i_obs)) do df
        df = df[1:n_post_samples_LOOCV,:]
        df[!,:ϵ_log10] .= log10(maximum(df.dist)) # used in violinplot
        df
end

let savefig = false, θ = :wolf_reproduce, col=:ϵ_log10
        axlimits_θ = Dict(:wolf_reproduce => (0,0.3), :sheep_reproduce => (0,1))
        violinwidth_θ = Dict(:wolf_reproduce => 0.005, :sheep_reproduce => 0.015)
        rejection_info = merge(config, @dict(Naccepted=n_post_samples_LOOCV, θ, col, summaries))

        fig = plot_LOOCV_violin(θ, rejection_distances,θ_prior_samples,col;
                        axlimits = axlimits_θ[θ], violinkwargs=(;width=violinwidth_θ[θ]))
        if savefig
                @pipe (["svg","png"] .|> savename("violin_LOOCV", rejection_info, _ ) .|> 
                        plotsdir("priors_$(prior_id)",
                                "raw", _) .|> 
                        safesave(_ ,fig));
        end
        fig
end
        

