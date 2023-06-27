#=============================================================================================#
# # Prior elicitation
#=============================================================================================#

using DrWatson
@quickactivate "ABC_ABM"

using CairoMakie
using Distributions, Random, StatsBase
using DataFrames
using Pipe
using FileIO


# ## Inspect prior samples
#---------------------------------------------------------------------------------------------#
prior_id = 0             # prior settings identifier 
n_prior_samples = 1000   # simulation batch size
n_repetitions = 1        # number of repetitions
n_steps_max = 1000

prior_simulations, θ_prior_samples = @pipe ("sims", "params") .|> 
                                    savename( _ , (;samples=n_prior_samples,
                                                    rep=n_repetitions,
                                                    steps=n_steps_max) ,"csv") .|> 
                                    datadir("sims","priors_$(prior_id)", _) .|>
                                    load .|> DataFrame


# ## Check for extinctions
#---------------------------------------------------------------------------------------------#
sim_grouped = groupby(prior_simulations,:ensemble)
go_extinct(x) = last(x) == 0

sim_extinctions = combine(sim_grouped, nrow => :steps , :count_sheep => go_extinct => :sheep_die, :count_wolf => go_extinct => :wolves_die)
println("total sheep extinction probability: ", sum(sim_extinctions.sheep_die)/(n_prior_samples*n_repetitions))
println("total wolf  extinction probability: ", sum(sim_extinctions.wolves_die)/(n_prior_samples*n_repetitions))

# ## Estimate new prior distributions
#---------------------------------------------------------------------------------------------#
ensembles_coexistence = @pipe sim_extinctions |> _.ensemble[.~(_.sheep_die .| _.wolves_die)] 

p_ABC = Dict(
    :wolf_reproduce  => fit_mle(Beta, θ_prior_samples.wolf_reproduce[ensembles_coexistence]),
    :sheep_reproduce => fit_mle(Beta, θ_prior_samples.sheep_reproduce[ensembles_coexistence])
    )

let θ = :wolf_reproduce, save_plot = false
    fig = hist(θ_prior_samples[ensembles_coexistence, θ], normalization =:pdf, label="accepted", axis=(;xlabel="$(θ)", ylabel="density"))
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


# ## Estimate length of transient phase
#---------------------------------------------------------------------------------------------#
# calculate stepwise variance of all simulations with co-existence
total_var_coexistence = @pipe sim_grouped[ensembles_coexistence] |> DataFrame |> groupby(_,:step) |> combine(_, [:count_wolf, :count_sheep] .=> var) 

# plot step-wise variance
let save_plot = true
    fig, ax, l = lines(total_var_coexistence.count_sheep_var, label="sheep_count", axis=(;xlabel="step", ylabel="variance"))
    lines!(ax, total_var_coexistence.count_wolf_var, label="wolf_count")
    axislegend()

    if save_plot
        @pipe (["svg","png"] .|> savename("coexistence_variance",
            (;samples=n_prior_samples,
            rep=n_repetitions,
            steps=n_steps_max), _ ) .|> 
            plotsdir("priors_$(prior_id)", _) .|> 
            safesave(_ ,fig));

    end
    fig
end

# from fig: 
# 500 steps suffies to reach a "stable" state


# ## Parameter sets without extinctions
#---------------------------------------------------------------------------------------------#

# plot parameter sets without extinctions
@pipe extinction_probs[id_coexist,:] |> scatter(_.sheep_reproduce, _.wolf_reproduce)


# ### plot simulations with parameter sets without extinction
let alpha=0.3,n_rep=10, colors = [:red, :blue, :orange, :purple, :green, :cyan]
    fig = Figure()
    fig[1,1] = Axis(fig, xlabel="step", ylabel="count_sheep")
    fig[2,1] = Axis(fig, xlabel="step", ylabel="count_wolf")
    for i in eachindex(colors)
        println(colors[i])
        combine(groupby(coexist_gdf[i], :ensemble)[1:n_rep]) do df
            lines!(fig[1,1], df.count_sheep, color=(colors[i], alpha))
            lines!(fig[2,1], df.count_wolf, color=(colors[i], alpha))
        end

    end
    fig
end 

# ### estimate stepwise variance of simulations with parameter sets without extinction
count_var = @time @pipe coexist_gdf |> DataFrame |> groupby(_, [:id,:step]) |> combine(_, [:count_wolf, :count_sheep] .=> var)
let alpha=0.8, colors = [:red, :blue, :orange, :purple, :green, :cyan]
    fig = Figure()
    fig[1,1] = Axis(fig, xlabel="step", ylabel="count_sheep_var")
    fig[2,1] = Axis(fig, xlabel="step", ylabel="count_wolf_var")
    for i in eachindex(colors)
        println(colors[i])
        df = groupby(count_var, :id)[i]
        lines!(fig[1,1],df.step, df.count_sheep_var, color=(colors[i], alpha))
        lines!(fig[2,1],df.step, df.count_wolf_var, color=(colors[i], alpha))
    end
    fig
end



# ## Estimate length of transient phase
#---------------------------------------------------------------------------------------------#

id_coexist = subset(extinction_probs, :t_ext_av => ByRow(isnan)).id # parameter sets with no extinction
coexist_gdf= groupby(prior_simulations, [:id])[id_coexist]
  
# all simulations with co-extinction
ensembles_coexistence = @pipe sim_extinctions |> _.ensemble[.~(_.sheep_die .| _.wolves_die)]
total_var_coexistence = @pipe sim_grouped[ensembles_coexistence] |> DataFrame |> groupby(_,:step) |> combine(_, [:count_wolf, :count_sheep] .=> var) 

fig_var = let 
    fig, ax, l = lines(total_var_coexistence.count_sheep_var, label="sheep_count", axis=(;xlabel="step", ylabel="variance"))
    lines!(ax, total_var_coexistence.count_wolf_var, label="wolf_count")
    axislegend()
    fig
end

@pipe (["svg","png"] .|> savename("coexistence_variance",
            (;samples=n_prior_samples,
            rep=n_repetitions,
            steps=n_steps_max), _ ) .|> 
            plotsdir("priors_$(prior_id)", _) .|> 
            safesave(_ ,fig_var));


# ## Estimate new prior distributions
#---------------------------------------------------------------------------------------------#
p_ABC = Dict(
    :wolf_reproduce  => fit_mle(Beta, θ_prior_samples.wolf_reproduce[ensembles_coexistence]),
    :sheep_reproduce => fit_mle(Beta, θ_prior_samples.sheep_reproduce[ensembles_coexistence])
    )

let θ = :sheep_reproduce
    fig = hist(θ_prior_samples[ensembles_coexistence, θ], normalization =:pdf)
    @pipe LinRange(0, 1. ,1000) |> collect |> lines!( fig.axis, _ , pdf.(p_ABC[θ], _ ))
    fig
end


# plot new prior distribution
prior_id_new = prior_id + 1
let savefig = true
    wolf_reproduce = LinRange(0, 1. ,1000)
    sheep_reproduce = LinRange(0, 1. ,1000)
    p_wolf = pdf.(p_ABC[:wolf_reproduce], collect(wolf_reproduce) )
    p_sheep = pdf.(p_ABC[:sheep_reproduce], collect(sheep_reproduce) )
    fig = Figure(resolution=(800,800))
    ax_contour = fig[2:4,1:3] = Axis(fig, xlabel="sheep_reproduce", ylabel="wolf_reproduce")
    ax_sheep_density = fig[1,1:3] = Axis(fig)
    ax_wolf_density = fig[2:4,4] = Axis(fig)
    for ax in (ax_sheep_density,ax_wolf_density)
        hidedecorations!(ax)  # hides ticks, grid and lables
        hidespines!(ax)  # hide the frame
    end
    contour!(ax_contour, sheep_reproduce, wolf_reproduce, p_wolf' .* p_sheep)
    lines!(ax_sheep_density, sheep_reproduce, p_sheep )
    lines!(ax_wolf_density, p_wolf, wolf_reproduce )
    
    if savefig
        @info "saving figures"
        @pipe (["svg","png"] .|> savename("prior_pdf",
                (;), _ ) .|> 
                plotsdir("priors_$(prior_id_new)", _) .|> 
                safesave(_ ,fig))
    end
    fig
end
