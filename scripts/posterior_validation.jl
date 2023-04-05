#=============================================================================================#
# # Infer posterior from observations
#=============================================================================================#

using DrWatson
@quickactivate "ABC_ABM"

using DataFrames, DataFramesMeta
using FileIO, CSV, Pipe, Chain
using CairoMakie, Colors
using Distributions
using StatsBase, Distances
using Random
import LinearAlgebra 
using Random

import MLJ
import PartialLeastSquaresRegressor: PLSRegressor, PLS2Model
import HypothesisTests: ExactOneSampleKSTest, ApproximateOneSampleKSTest, pvalue

include(srcdir("rejection_ABC.jl")) 
include(srcdir("plot_functions.jl")) 


# ## Load observations
#---------------------------------------------------------------------------------------------#
# load observational data
obs_data = datadir("sims","observations", "observed_data.csv") |> load |> DataFrame
n_steps_obs = maximum(obs_data.step)

obs_data[!,:ensemble] .= -1 # give observations a distinctive ID


# ## Load prior simulations
#---------------------------------------------------------------------------------------------#

prior_id = 2             # prior settings identifier 
n_prior_samples = 100_000  # simulation batch size
n_repetitions = 1        # number of repetitions
steps_transient = 500    # transient steps (to be cut off from prior simulations)
n_steps_max = n_steps_obs + steps_transient
prior_info = @dict(samples = n_prior_samples, rep = n_repetitions, steps = n_steps_max)

prior_simulations = @pipe savename( "sims" , prior_info ,"csv") |> 
                                    datadir("sims","priors_$(prior_id)", _) |>
                                    CSV.read(_, DataFrame)
select!(prior_simulations, Not(:id))

θ_prior_samples = @pipe savename("params" , prior_info ,"csv") |> 
                                    datadir("sims","priors_$(prior_id)", _) |>
                                    CSV.read(_, DataFrame)

(:ensemble ∉ propertynames(θ_prior_samples)) && (θ_prior_samples[!,:ensemble] = prior_simulations.ensemble |> unique) # extra safety for files generated with old code

θ_symbols = [:sheep_reproduce, :wolf_reproduce]
select!(θ_prior_samples, [:ensemble, θ_symbols...])


# ## Pre-process prior simulations
#---------------------------------------------------------------------------------------------#

# filter out extinctions
summaries_extinctions = @time combine(groupby(prior_simulations,:ensemble), 
                        nrow => :steps , 
                        :count_sheep => go_extinct => :sheep_die, 
                        :count_wolf => go_extinct => :wolves_die,
                        [:count_sheep,:count_wolf] => coexist => :coexist,
                        [:count_sheep,:count_wolf] => nonconstant_TS => :nonconstant_TS)

filter_simulations = summaries_extinctions.coexist .& summaries_extinctions.nonconstant_TS
prior_simulations = @time @pipe groupby(prior_simulations,:ensemble) |> 
                        _[filter_simulations] |> DataFrame |>
                        subset(_, :step => s -> s.> steps_transient) 


# ## Define discrepancy measures & summaries
#---------------------------------------------------------------------------------------------#
summaries_counts_symb = [:count_sheep,:count_wolf]
σ_π₀(x::AbstractVector) = 1/std(x) # prior standard deviation, to be used as weights for Euclid.

# load last PLS machine configuration
PLS_prior_id = 1
machine_path = datadir("sims","priors_$(PLS_prior_id)","machines") 
machine_config = @pipe machine_path |> joinpath(_, "PLS_machine_config.jld2") |> load 
println("loading PLS machine, trained in git commit: \n",pop!(machine_config, "gitcommit"))
PLS_machine = @pipe savename("PLS_machine",machine_config,"jls") |> 
                    joinpath(machine_path, _) |>
                    MLJ.machine

@assert steps_transient == machine_config["stepstrans"]

lag_max = machine_config["lag"]
L = machine_config["L"]

# compute TS summaries
summaries_config = @dict(lag = lag_max, L, stepstrans=steps_transient)
path = datadir("sims","priors_$(prior_id)", "summaries")
summaries_TS_filename = @time produce_or_load(path, 
        summaries_config, 
        filename = savename(machine_config["summaries"], merge(summaries_config,prior_info) ,"csv"), 
        tag=false,
        loadfile=false) do config
            time_series_summaries(groupby(prior_simulations,:ensemble), config)
        end |> last
println("read as DataFrame: $(summaries_TS_filename)")
summaries_TS = @time CSV.read(summaries_TS_filename, DataFrame)
(:ensemble ∉ propertynames(summaries_TS)) && (summaries_TS[!, :ensemble] = summaries_extinctions.ensemble[filter_simulations]) # extra safety for files generated with old code

#= log (load)
4.975617 seconds (14.32 M allocations: 833.254 MiB, 6.46% gc time, 99.59% compilation time)

priors_2\summaries\autocor_fft_SSA_diststat_L=30_lag=500_rep=1_samples=100000_steps=1500_stepstrans=500.csv
36.650198 seconds (483.65 k allocations: 853.736 MiB, 1.89% gc time)
=#

summaries_TS_obs = time_series_summaries(groupby(obs_data,:ensemble), summaries_config)
summaries_moments, summaries_moments_obs = @pipe [summaries_TS, summaries_TS_obs] .|> select(_, :ensemble, r"^count") # extract columns starting with count
plot_summaries(
    innerjoin(θ_prior_samples, summaries_moments, on=:ensemble), 
    θ_symbols,
    scatter_kwargs=(;markersize=5),
    savefig=true,
    filenames= @pipe (["svg","png"] .|> 
                savename("scatter_moments", merge(prior_info), _ ) .|> 
                plotsdir("priors_$(prior_id)", "raw", "validation", _)))


# compute PLS summaries
summaries_PLS_obs, summaries_PLS = @time  @pipe [summaries_TS_obs, summaries_TS] .|> 
                                select(_, Not(:ensemble)) .|>   # remove :ensemble from summary data
                                MLJ.predict(PLS_machine, _) .|>  # use trained PLS to transform summaries
                                DataFrame .|> 
                                rename!(_, θ_symbols)

#= log
samples
100k  : 108.644318 seconds (13.87 M allocations: 58.675 GiB, 52.19% gc time, 16.47% compilation time) 
=#

# add simulation ID to PLS DF
summaries_PLS = hcat( DataFrame(:ensemble => summaries_TS.ensemble), summaries_PLS)
summaries_PLS_obs = hcat( DataFrame(:ensemble => [-1]), summaries_PLS_obs)


plot_summaries(
    innerjoin(θ_prior_samples, rename(summaries_PLS, [θ => "PLS_$(θ)" for θ in θ_symbols]), on=:ensemble), 
    θ_symbols,
    scatter_kwargs=(;markersize=5),
    savefig=true,
    filenames= @pipe (["svg","png"] .|> 
                savename("scatter_PLS", merge(summaries_config,prior_info), _ ) .|> 
                plotsdir("priors_$(prior_id)", "raw", "validation", _)))


# plot PLS summaries
let savefig = true, n_plot=1000, markersize=5
    θ_true = innerjoin(θ_prior_samples, DataFrame(:ensemble => summaries_TS.ensemble), on=:ensemble) # filter out ensembles
    @assert(all(θ_true.ensemble .== summaries_PLS.ensemble))
    fig = Figure(resolution=(800,800))
    fig[1,1] = Axis(fig, xlabel="true wolf_reproduce", ylabel="PLS estimate wolf_reproduce")
    fig[1,2] = Axis(fig, xlabel="true sheep_reproduce", ylabel="PLS estimate wolf_reproduce")   
    fig[2,1] = Axis(fig, xlabel="true wolf_reproduce",  ylabel="PLS estimate sheep_reproduce") 
    fig[2,2] = Axis(fig, xlabel="true sheep_reproduce", ylabel="PLS estimate sheep_reproduce")   
    scatter!(fig[1,1],θ_true[1:n_plot, :wolf_reproduce], summaries_PLS[1:n_plot, :wolf_reproduce], markersize=markersize)
    scatter!(fig[1,2],θ_true[1:n_plot, :sheep_reproduce], summaries_PLS[1:n_plot, :wolf_reproduce], markersize=markersize)
    scatter!(fig[2,1],θ_true[1:n_plot, :wolf_reproduce], summaries_PLS[1:n_plot, :sheep_reproduce], markersize=markersize)
    scatter!(fig[2,2],θ_true[1:n_plot, :sheep_reproduce], summaries_PLS[1:n_plot, :sheep_reproduce], markersize=markersize)
    if savefig
        if n_plot ≤ 1000
            #save both svg as png
            @pipe (["svg","png"] .|> savename("PLS_estimates", merge(summaries_config,prior_info, @dict(n_plot)), _ ) .|> 
                    plotsdir("priors_$(prior_id)",
                            "raw", "validation",_) .|> 
                    safesave(_ ,fig));
        else
            # only save png
            @pipe (["png"] .|> savename("PLS_estimates", merge(summaries_config,prior_info, @dict(n_plot) ), _ ) .|> 
                    plotsdir("priors_$(prior_id)",
                            "raw", "validation",_) .|> 
                    safesave(_ ,fig));
        end
    end
    fig 
end






#= log
100k
110.680176 seconds (12.96 M allocations: 57.887 GiB, 35.48% gc time, 8.17% compilation time)
=#

# calculate weights for euclidean dist
D_PLS, D_moments = @time @pipe [summaries_PLS,summaries_moments] .|> 
                            select(_, Not(:ensemble)) .|>
                            combine(_, propertynames(_) .=> σ_π₀) .|> # compute sd for each summary statistic
                            WeightedSqEuclidean([values(_[1,:])...])  # use the sd in WeightedEuclidean
#= log
10k samples
2.214760 seconds (1.03 M allocations: 53.774 MiB, 95.92% compilation time)

100k samples
29.159544 seconds (3.05 M allocations: 169.592 MiB, 97.95% compilation time: 0% of which was recompilation)
=#


# ## LOOCV
#---------------------------------------------------------------------------------------------#

n_LOOCV = 100 # number of LOOCV samples
i_obs_seed = 42    # seed for picking pseudo-observations
summaries = :PLS
LOOCV_config = merge(prior_info, summaries_config, @dict(summaries, seed = i_obs_seed, nLOOCV=n_LOOCV))

path = datadir("sims","priors_$(prior_id)","distances")
distances_LOOCV_filename = @time produce_or_load(path, LOOCV_config , filename = savename("$(summaries)", LOOCV_config,"csv"), tag=false, loadfile=false) do config
        i_obs_vec = sample(Random.MersenneTwister(config[:seed]), summaries_TS.ensemble , config[:nLOOCV], replace = false)
        if summaries == :moments
                @time alldistances_LOOCV(summaries_moments, D_moments, 
                        config[:nLOOCV],
                        i_obs_vec)
        elseif summaries == :PLS
                @time alldistances_LOOCV(summaries_PLS, D_PLS, 
                        config[:nLOOCV],
                        i_obs_vec)
        else
                error("use 'moments' or 'PLS' as summary stats ")
        end
        end |> last

distances_LOOCV = @pipe CSV.read(distances_LOOCV_filename, DataFrame) |> innerjoin(_, θ_prior_samples, on=:ensemble) 
i_obs_vec = distances_LOOCV.i_obs |> unique


let savefig = true, θ = :sheep_reproduce, col=:ϵ_log10, accepted = 500
    rejection_LOOCV = @time combine(groupby(distances_LOOCV, :i_obs)) do df
        df = df[1:accepted,:]
        df[!,:ϵ_log10] .= log10(maximum(df.dist)) # used in violinplot
        df
    end
    axlimits_θ = Dict(:wolf_reproduce => (0,0.3), :sheep_reproduce => (0,1))
    violinwidth_θ = Dict(:wolf_reproduce => 0.005, :sheep_reproduce => 0.015)
    rejection_info = merge(LOOCV_config, @dict(accepted, θ, col))
    fig = plot_LOOCV_violin(θ, rejection_LOOCV,θ_prior_samples,col;
                    axlimits = axlimits_θ[θ], violinkwargs=(;width=violinwidth_θ[θ]))
    if savefig
            @pipe (["svg","png"] .|> savename("violin_LOOCV", rejection_info, _ ) .|> 
                    plotsdir("priors_$(prior_id)",
                            "raw","validation", _) .|> 
                    safesave(_ ,fig));
    end
    fig
end





# ## Compute distances to observed summaries
#---------------------------------------------------------------------------------------------#

dist_PLS = @pipe vcat(summaries_PLS, summaries_PLS_obs) |> 
                    alldistances_LOOCV(_, D_PLS, 1, [-1])

dist_moments = @pipe vcat(summaries_moments, summaries_moments_obs) |> 
                    alldistances_LOOCV(_, D_moments, 1, [-1])

#= 

# plot PLS summaries with distance to df
let savefig = false, n_plot=88066
    θ_true = innerjoin(θ_prior_samples, dist_PLS, on=:ensemble)
    @assert(all(θ_true.ensemble .== summaries_PLS.ensemble))
    fig = Figure(resolution=(600,1200))
    fig[1,1] = Axis(fig, xlabel="true wolf_reproduce", ylabel="PLS estimate wolf_reproduce")
    fig[1,2] = Axis(fig, xlabel="true sheep_reproduce", ylabel="PLS estimate sheep_reproduce")   
    fig[2,1] = Axis(fig, ylabel="true wolf_reproduce", xlabel="true sheep_reproduce", title="PLS estimate wolf_reproduce") 
    fig[2,2] = Axis(fig, ylabel="true wolf_reproduce", xlabel="true sheep_reproduce", title="PLS estimate sheep_reproduce")   
    fig[3:4,1:2] = Axis(fig, ylabel="true wolf_reproduce", xlabel="true sheep_reproduce", title="distance to observed")
    scatter!(fig[1,1],θ_true[1:n_plot, :wolf_reproduce], summaries_PLS[1:n_plot, :wolf_reproduce], markersize=1)
    scatter!(fig[1,2],θ_true[1:n_plot, :sheep_reproduce], summaries_PLS[1:n_plot, :sheep_reproduce], markersize=1)
    scatter!(fig[2,1],θ_true[1:n_plot, :sheep_reproduce], θ_true[1:n_plot, :wolf_reproduce], color = summaries_PLS[1:n_plot, :wolf_reproduce], markersize=1)
    scatter!(fig[2,2],θ_true[1:n_plot, :sheep_reproduce], θ_true[1:n_plot, :wolf_reproduce], color = summaries_PLS[1:n_plot, :sheep_reproduce], markersize=1)
    scatter!(fig[3:4,1:2],θ_true[1:n_plot, :sheep_reproduce], θ_true[1:n_plot, :wolf_reproduce], color = θ_true[1:n_plot,:dist], markersize=3)
    
    if savefig
    @pipe (["svg","png"] .|> savename("PLS_predictions_$(part)", machine_config, _ ) .|> 
            plotsdir("priors_$(prior_id)",
                    "raw", _) .|> 
            safesave(_ ,fig));
    end
    fig 
end =#





# ## Coverage
#---------------------------------------------------------------------------------------------#
#= # ### sanity check: coverage on prior

let K_π₀ = 200, nⱼ_vec = [10,15,20,30,40,50,75,100,200,300,500,1000], θ_prior_samples = θ_prior_samples[filter_simulations,:]
    i_shuffled = shuffle(1:size(θ_prior_samples,1))
    p_0_π₀ = combine(groupby(θ_prior_samples, :ensemble)[i_shuffled[1:K_π₀]]) do df
        p_0_df = DataFrame(:nⱼ => nⱼ_vec,  
                        :wolf_reproduce => zeros(length(nⱼ_vec)), 
                        :sheep_reproduce => zeros(length(nⱼ_vec)))
        for i in eachindex(nⱼ_vec)
            θ_accepted = θ_prior_samples[sample(i_shuffled,nⱼ_vec[i], replace=false),:]
            p_0_df[i,2:end] = [estimate_p₀(θ_accepted.wolf_reproduce, values(df.wolf_reproduce)..., nⱼ_vec[i]), 
                                estimate_p₀(θ_accepted.sheep_reproduce, values(df.sheep_reproduce)..., nⱼ_vec[i])]
        end
        p_0_df
    end
    coverage_test_statistics = @combine(groupby(p_0_π₀,:nⱼ), 
        :wolf_reproduce = pvalue(ApproximateOneSampleKSTest(:wolf_reproduce, Uniform())),
        :sheep_reproduce = pvalue(ApproximateOneSampleKSTest(:sheep_reproduce, Uniform())))
    

    fig = Figure()
    ax = fig[1:2,1:3] = Axis(fig, title="K=$(K_π₀)",xlabel = "nⱼ", ylabel="pₖₛ",yscale=log10, limits=(nothing,nothing,1e-6,1), xscale=log10, xticks = coverage_test_statistics.nⱼ)
    series!(ax, coverage_test_statistics.nⱼ, Matrix(coverage_test_statistics[!, [:wolf_reproduce, :sheep_reproduce]])', marker='+', markersize=20, labels=["wolf_reproduce", "sheep_reproduce"])
    axislegend(position=:rb)

    hist_col = categorical_colors(:lighttest,2)
    for (i,nⱼ) in enumerate(nⱼ_vec[[1, 8, end]])
        hist(fig[3,i], @subset(p_0_π₀, :nⱼ .== nⱼ).wolf_reproduce, color = hist_col[1],
            axis=(;title="nⱼ = $(nⱼ)", limits=(0,1,0,nothing)))
        hist(fig[4,i], @subset(p_0_π₀, :nⱼ .== nⱼ).sheep_reproduce, color = hist_col[2],
            axis=(;xlabel = "p₀", limits=(0,1,0,nothing)))
    end
    fig

end =#



# ### perform coarse inference
s = :moments
K_max = 200
posterior_LOOCV = @time @pipe alldistances_LOOCV(summaries_dict[s], D_dict[s], K_max, dist_dict[s].ensemble[1:K_max]) |> 
    innerjoin(_, θ_prior_samples, on=:ensemble)
ϵ_vec =  @pipe ([10,20,30,50,100,200,300,500,1000,2000,3000,5000] ./ sum(filter_simulations)) |> 
                quantile(posterior_LOOCV.dist, _ ) # generate ϵ based on number of accepted samples

# ### assess coverage, with varying K
K = 200
coverage_samples = groupby(posterior_LOOCV,:i_obs)[1:K]
@time begin
    p_0 = combine(coverage_samples) do df
                θ_true = θ_prior_samples[θ_prior_samples.ensemble .== df.i_obs[1],:]
                p_0_df = DataFrame(:ϵ => ϵ_vec,  
                                :nⱼ => zeros(Int64,length(ϵ_vec)),
                                :wolf_reproduce => zeros(length(ϵ_vec)), 
                                :sheep_reproduce => zeros(length(ϵ_vec)))
                for i in eachindex(ϵ_vec)
                    i_accepted = df.dist .≤ ϵ_vec[i]
                    W_j = df[i_accepted , :] 
                    nⱼ = sum(i_accepted)
                    p_0_df[i,2] = nⱼ
                    p_0_df[i,3:end] = [estimate_p₀(W_j.wolf_reproduce, θ_true[1,:wolf_reproduce], nⱼ), estimate_p₀(W_j.sheep_reproduce, θ_true[1,:sheep_reproduce], nⱼ)]
                end
                p_0_df
        end 
    coverage_test_statistics = @combine(groupby(p_0,:ϵ), 
        :nⱼ_mean = mean(:nⱼ),
        :nⱼ_sd = std(:nⱼ),
        :wolf_reproduce = pvalue(ApproximateOneSampleKSTest(:wolf_reproduce, Uniform())),
        :sheep_reproduce = pvalue(ApproximateOneSampleKSTest(:sheep_reproduce, Uniform())))
    
    coverage_config = @dict summaries=s K ϵ_vec
    fig_coverage_p = let savefig=true
        fig = Figure(resolution=(900,900))
        ax_series = fig[1:2,1:3] = Axis(fig, xlabel = "ϵ", ylabel="pₖₛ",yscale=log10, limits=(nothing,nothing,1e-6,1), xscale=log10, xticks = round.(coverage_test_statistics.ϵ, digits=3))
        series!(ax_series, coverage_test_statistics.ϵ, Matrix(coverage_test_statistics[!, [:wolf_reproduce, :sheep_reproduce]])', marker='+', markersize=20, labels=["wolf_reproduce", "sheep_reproduce"])
        axislegend(position=:rb)
        hist_col = categorical_colors(:lighttest,2)
        for (i,ϵ) in enumerate(ϵ_vec[[1, 5, end]])
            hist(fig[3,i], @subset(p_0, :ϵ .== ϵ).wolf_reproduce, color = hist_col[1],
                axis=(;title="ϵ = $(round(ϵ, digits=2)), nⱼ = $(round(Int,coverage_test_statistics[ϵ_vec.==ϵ,:nⱼ_mean]...)) ($(round(Int,coverage_test_statistics[ϵ_vec.==ϵ,:nⱼ_sd]...)))", limits=(0,1,0,nothing)))
            hist(fig[4,i], @subset(p_0, :ϵ .== ϵ).sheep_reproduce, color = hist_col[2],
                axis=(;xlabel = "p₀", limits=(0,1,0,nothing)))
        end

        if savefig
            @pipe (["svg","png"] .|> 
            savename("coverage_pKS", coverage_config, _ ) .|> 
            plotsdir("priors_$(prior_id)",
                    "raw",savename(@dict(nsamples=n_prior_samples)), _ ) .|> 
            safesave(_ ,fig));
        end
        fig
    end
end





# violin(p_0.ϵ, p_0.nⱼ, axis=(;xlabel="ϵ", ylabel="n accepted"), width=10)

let ϵ = ϵ_vec[end], savefig=false
    p_0_hist = filter(:ϵ => ==(ϵ), p_0)
    fig = Figure(;resolution=(800,400))
    hist(fig[1,1], p_0_hist.wolf_reproduce, 
        axis=(;xlabel = "p₀",title="wolf_reproduce", limits=(0,1,0,nothing)))
    hist(fig[1,2],p_0_hist.sheep_reproduce, 
        axis=(;xlabel = "p₀",title="sheep_reproduce", limits=(0,1,0,nothing)))
    
    if savefig
        @pipe (["svg","png"] .|> 
        savename("coverage_hist", merge(coverage_config, @dict(ϵ)), _ ) .|> 
        plotsdir("priors_$(prior_id)",
                "raw", savename(machine_config), _ ) .|> 
        safesave(_ ,hist_coverage));
    end
    fig
end

ϵ = 9
nLOOCV = 50
@assert(nLOOCV <= K_max)
rejection_distances = @chain coverage_samples[1:floor(Int,K/nLOOCV):K] begin
    DataFrame
    @subset(:dist .<= ϵ ) 
    groupby(:i_obs)
    @transform(:naccept = length(:ensemble))
end 

let savefig = true, θ = :wolf_reproduce, col=:naccept
    axlimits_θ = Dict(:wolf_reproduce => (0.02,0.12), :sheep_reproduce => (0.15,0.45))
    violinwidth_θ = Dict(:wolf_reproduce => 0.002, :sheep_reproduce => 0.005)
    rejection_info = merge(coverage_config, @dict(nLOOCV, θ, col, ϵ))

    fig = plot_LOOCV_violin(θ, rejection_distances,θ_prior_samples,col;
                                axlimits = axlimits_θ[θ], violinkwargs=(;width=violinwidth_θ[θ],datalimits=extrema))
    if savefig
            @pipe (["svg","png"] .|> savename("violin_LOOCV_coverage", rejection_info, _ ) .|> 
                    plotsdir("priors_$(prior_id)",
                            "raw",savename(@dict(nsamples=n_prior_samples)), _) .|> 
                    safesave(_ ,fig));
    end
    fig
end


# ## Resulting posterior
#---------------------------------------------------------------------------------------------#

summaries_dict = @dict PLS=summaries_PLS moments=summaries_moments
D_dict = @dict PLS=D_PLS moments=D_moments
dist_dict = @dict PLS=dist_PLS moments=dist_moments

priors_dict = Dict(2 => (; wolf_reproduce  = Beta(1.59861, 20.1555), sheep_reproduce = Beta(1.81811, 3.28002))
)

let n_accepted = 500, s=:PLS, wolf_rep_lim = (0,0.5), sheep_rep_lim = (0,0.5), savefig = true
    posterior_df = @pipe dist_dict[s] |> innerjoin(_[1:n_accepted,2:3], θ_prior_samples, on=:ensemble)
    fig = Figure()
    ax_scat = fig[2:3,1:2] = Axis(fig, xlabel="wolf_reproduce", ylabel="sheep_reproduce", limits=(wolf_rep_lim...,sheep_rep_lim...))
    ax_wolf = fig[1,1:2] = Axis(fig, xlabel="wolf_reproduce", ylabel="density", limits=(wolf_rep_lim...,0,nothing))
    contour()
    ax_sheep = fig[2:3,3] = Axis(fig, xlabel="density", ylabel="sheep_reproduce", limits=(0,nothing,sheep_rep_lim...))
    scatter!(ax_scat, posterior_df.wolf_reproduce, posterior_df.sheep_reproduce, color=posterior_df.dist, markersize=5, colormap=:plasma)
    @pipe LinRange(sheep_rep_lim... ,1000) |> collect |> lines!( ax_sheep, pdf.(priors_dict[prior_id].sheep_reproduce, _ ), _ , label="prior", color=:darkgrey)
    @pipe LinRange(wolf_rep_lim... ,1000) |> collect |> lines!( ax_wolf, _ , pdf.(priors_dict[prior_id].wolf_reproduce, _ ), label="prior", color=:darkgrey)
    density!(ax_sheep, posterior_df.sheep_reproduce,direction=:y, color=(:grey, 0.7))
    density!(ax_wolf, posterior_df.wolf_reproduce, color=(:grey, 0.7))
    if savefig
        @pipe ["svg","png"] .|> 
            savename("posterior", merge(machine_config, @dict(s, n_accepted)), _ ) .|> 
            plotsdir("priors_$(prior_id)",
                    "raw",savename(@dict(nsamples=n_prior_samples)), _ ) .|> 
            safesave(_ ,fig);
    end
    fig
end