using CairoMakie
using DataFrames
using ColorSchemes
using Colors
using LinearAlgebra

function plot_population_timeseries(adf::DataFrame, mdf::DataFrame)
    figure = Figure(resolution = (600, 400))
    ax = figure[1, 1] = Axis(figure; xlabel = "Step", ylabel = "Population")
    sheepl = lines!(ax, adf.step, adf.count_sheep, color = :cornsilk4)
    wolfl = lines!(ax, adf.step, adf.count_wolf, color = RGBAf(0.2, 0.2, 0.3))
    grassl = lines!(ax, mdf.step, mdf.count_grass, color = :green)
    figure[1, 2] = Legend(figure, [sheepl, wolfl, grassl], ["Sheep", "Wolves", "Grass"])
    figure
end

function plot_population_timeseries(adf::AbstractDataFrame)
    figure = Figure(resolution = (600, 400))
    ax = figure[1, 1] = Axis(figure; xlabel = "Step", ylabel = "Population")
    sheepl = lines!(ax, adf.step, adf.count_sheep, color = :cornsilk4)
    wolfl = lines!(ax, adf.step, adf.count_wolf, color = RGBAf(0.2, 0.2, 0.3))
    figure[1, 2] = Legend(figure, [sheepl, wolfl], ["Sheep", "Wolves"])
    figure
end


"""
plot_n_agent_timeseries

gdf: simulations, grouped per simulation_ID
agent_data: symbol of agent data to plot
simulation_ids: ids of simulations to plot
"""
function plot_n_agent_timeseries!(fig::Union{Figure,GridPosition, GridSubposition},gdf::GroupedDataFrame, agent_data::Symbol, simulation_ids::AbstractArray, colorvalues::AbstractArray;
    start = 1,
    axis::NamedTuple=(;),
    line::NamedTuple=(;),
    colorscheme::Symbol=:diverging_gwr_55_95_c38_n256
    )

    cs = colorschemes[colorscheme]
    colormapping = Dict(simulation_ids .=> get(cs,colorvalues,:extrema))
    agent_name = split(String(agent_data), '_')[2]
    ax = fig[1, 1] = Axis(fig; xlabel = "Step", ylabel = "$(agent_name) population", axis...)
    for id in simulation_ids
        adf = gdf[(id,)]
        lines!(ax, adf.step[start:end], adf[start:end,agent_data], color = colormapping[id], transparency = true, line...)
    end
    fig
end


function plot_n_agent_timeseries!(fig::Union{Figure,GridPosition, GridSubposition},gdf::GroupedDataFrame, agent_data::Symbol, simulation_ids::AbstractArray;
    start = 1,
    axis::NamedTuple=(;),
    line::NamedTuple=(;)
    )


    agent_name = split(String(agent_data), '_')[2]
    ax = fig[1, 1] = Axis(fig; xlabel = "Step", ylabel = "$(agent_name) population", axis...)
    for id in simulation_ids
        adf = gdf[(id,)]
        lines!(ax, adf.step[start:end], adf[start:end,agent_data], transparency = true, line...)
    end
    fig
end




"""
plot_n_agent_timeseries

gdf: simulations, grouped per simulation_ID
agent_data: symbol of agent data to plot
simulation_ids: ids of simulations to plot
"""
function plot_n_agent_timeseries(gdf::GroupedDataFrame, agent_data::Symbol, simulation_ids::AbstractArray;
    start = 1,
    figure::NamedTuple=(; resolution = (800, 400)), 
    axis::NamedTuple=(;))
    """
    gdf: simulations, grouped per simulation_ID
    agent_data: symbol of agent data to plot
    simulation_ids: ids of simulations to plot
    """
    fig = Figure(;figure...)
    plot_n_agent_timeseries!(fig, gdf, agent_data, simulation_ids, start = start, axis=axis)
end

function plot_sample_vs_subsample!(fig::Union{Figure,GridPosition, GridSubposition},θ::Symbol, θ_sample::AbstractDataFrame, subsample_id::AbstractArray; 
    samplename = "full sample", 
    subsamplename = "subsample", 
    legend::Bool = true,
    legendpos::Union{GridPosition,GridSubposition} = fig[1,2],
    axis::NamedTuple=(;),
    density_sample::NamedTuple=(;),
    density_subsample::Union{Nothing,NamedTuple}=nothing)
    density_subsample = (density_subsample === nothing) ? density_sample : (;density_sample..., density_subsample...) # use density_sample as default
    axis = (;xlabel = "$(θ)", ylabel = "density",axis...)
    ax = Axis(fig[1, 1] ; axis...)
    sample_density = density!(ax, θ_sample[!,θ];density_sample...)
    subsample_density = density!(ax, θ_sample[subsample_id,θ]; density_subsample...)
    if legend
        Legend(legendpos, [sample_density, subsample_density], [samplename, subsamplename])
    end
    fig
end


function plot_sample_vs_subsample(θ::Symbol, θ_sample::AbstractDataFrame, subsample_id::AbstractArray; 
    samplename = "full sample", 
    subsamplename = "subsample", 
    legend::Bool = true,
    legendpos::Union{Nothing, GridPosition, GridSubposition} = nothing,
    figure::NamedTuple=(; resolution = (800, 400)), 
    axis::NamedTuple=(; xlabel = "$(θ)", ylabel = "density"),
    density_sample::NamedTuple=(;),
    density_subsample::Union{Nothing,NamedTuple}=nothing)
    fig = Figure(; figure...)

    plot_sample_vs_subsample!(fig, θ, θ_sample, subsample_id; 
    samplename = samplename, 
    subsamplename = subsamplename, 
    legend = legend,
    legendpos = (legendpos === nothing) ? fig[1,2] : legendpos,
    axis=axis,
    density_sample=density_sample,
    density_subsample=density_subsample)
end


"""
Make violin plot with LOOCV results of rejection_ABC.

θ: parameter name (as appearing in the DataFrame colums)
θ_posterior_samples: dataframe with colums θ, containing posterior samples for all pseudo-data (i_obs)
θ_prior_samples: dataframe with colums θ and :i_obs
"""
function plot_LOOCV_violin(θ::Symbol, 
        θ_posterior_samples::AbstractDataFrame, 
        θ_prior_samples::AbstractDataFrame;
        axlimits::Tuple = (0,0.5),
        violinkwargs::NamedTuple = (;))
    figure = Figure(resolution = (800, 800))
    θ_accepted = θ_posterior_samples[!,θ]
    θ_true = θ_prior_samples[θ_posterior_samples.i_obs, θ]
    figure[1, 1] = Axis(figure; xlabel = "true $(θ)", ylabel = "approximated posterior", aspect = 1, limits = (axlimits...,axlimits...))
    lines!(collect(Float64,axlimits),collect(Float64,axlimits),color="slategray3")
    violin!(θ_true,θ_accepted,color;width=0.01,show_median=true, datalimits = (x -> (quantile(x,0.05), quantile(x,0.95))),violinkwargs...)
    figure
end

"""
Make colored violin plot with LOOCV results of rejection_ABC.

θ: parameter name (as appearing in the DataFrame colums)
θ_posterior_samples: dataframe with colums θ, containing posterior samples for all pseudo-data (i_obs)
θ_prior_samples: dataframe with colums θ and :i_obs
"""
function plot_LOOCV_violin(θ::Symbol, 
                            θ_posterior_samples::AbstractDataFrame, 
                            θ_prior_samples::AbstractDataFrame,
                            tolerances::Symbol;
                            cs::Symbol=:plasma,
                            cs_lim::Union{Nothing,Tuple}=nothing,
                            alpha::Float64=0.8,
                            axlimits::Tuple = (0,0.5),
                            violinkwargs::NamedTuple = (;))

    cs_lim = (cs_lim === nothing) ? extrema(θ_posterior_samples[!,tolerances]) : cs_lim
    colors = RGBA.(get(colorschemes[cs], θ_posterior_samples[!,tolerances], cs_lim), alpha)
    figure = Figure(resolution = (1000, 800))
    θ_accepted = θ_posterior_samples[!,θ]
    θ_true = θ_prior_samples[θ_posterior_samples.i_obs, θ]
    figure[1, 1] = Axis(figure; xlabel = "true $(θ)", ylabel = "approximated posterior", aspect = 1, limits = (axlimits...,axlimits...))
    lines!(collect(Float64,axlimits),collect(Float64,axlimits),color="slategray3")
    violin!(θ_true,θ_accepted, color=colors, width=0.01, show_median=true, datalimits = (x -> (quantile(x,0.05), quantile(x,0.95)));  violinkwargs...)
    Colorbar(figure[1,2], label = "$(tolerances)", colormap = cs, limits = cs_lim)
    figure
end


function plot_LOOCV_scatter(θ_symbols::Tuple{Symbol, Symbol, Symbol}, 
    θ_posterior::AbstractDataFrame, 
    θ_prior_samples::AbstractDataFrame;
    axlimits::Tuple = (0,0.5),
    scatterkwargs::NamedTuple = (;markersize = 3, color = :black),
    errorkwargs::NamedTuple = (;color = :red))
    θ, θ_hat, θ_error = θ_symbols

    figure = Figure(resolution = (800, 800))
    θ_true = θ_prior_samples[θ_posterior.i_obs, θ]
    figure[1, 1] = Axis(figure; xlabel = "true $(θ)", ylabel = "point estimate ($(θ_hat))", aspect = 1, limits = (axlimits...,axlimits...))
    errorbars!(θ_posterior[!,θ_hat], θ_true,θ_posterior[!,θ_error];errorkwargs...)
    scatter!(θ_posterior[!,θ_hat], θ_true; scatterkwargs...)
    figure
end

function violin_df(df::AbstractDataFrame;
    axlimits::Tuple = (0,1),
    violinkwargs::NamedTuple = (;width = 0.5,show_median=true, datalimits = (x -> (quantile(x,0.05), quantile(x,0.95)))))
    fig = Figure(resolution = (200, 800))
    df_cols = propertynames(df)
    x_values = collect(1:length(df_cols))
    transform_dict = Dict(String.(df_cols) .=> x_values)
    ax = Axis(
        fig[1, 1], 
        xticks = (x_values, String.(df_cols))
        )
    ylims!(axlimits...)
    df_stack = stack(df,df_cols,[],variable_name=:parameter,value_name=:value)
    df_stack[!,"parameter"] = map(x -> transform_dict[x], Vector(df_stack.parameter))
    violin!(ax, df_stack.parameter,df_stack.value; violinkwargs...)
    fig
end



function video_LOOCV_scatter(θ_posterior::AbstractDataFrame, 
    θ_prior_samples::AbstractDataFrame, 
    i_obs_vec::AbstractArray,
    filename::AbstractString, 
    θ::Union{Nothing,Vector{Symbol}} = nothing; 
    framerate::Int = 1,
    limits::Tuple = (0,0.5,0,0.5))
    θ = (θ === nothing) ? propertynames(θ_posterior)[1:2] : θ
    # define Observable
    i_obs = Observable(i_obs_vec[1])
    θ_x_post = @lift θ_posterior[θ_posterior.i_obs .== $i_obs, θ[1]] 
    θ_y_post = @lift θ_posterior[θ_posterior.i_obs .== $i_obs, θ[2]] 
    θ_x_true = @lift θ_prior_samples[$i_obs,θ[1]]
    θ_y_true = @lift θ_prior_samples[$i_obs,θ[2]]

    fig = Figure()
    fig[1,1] = Axis(fig;xlabel="$(θ[1])",ylabel="$(θ[2])",limits=limits)
    scatter!(θ_x_post, θ_y_post)
    scatter!(θ_x_true,θ_y_true, color=:red)

    record(fig, filename, i_obs_vec; framerate = framerate) do i
        i_obs[] = i
    end
end


function plot_summaries(θ_summaries_df::AbstractDataFrame, θ_symbols::AbstractVector{Symbol};
    savefig::Bool = false, 
    n_plot::Int=1000, 
    simulation_ID::Symbol=:ensemble, 
    subplot_resolution::Int=400, 
    scatter_kwargs::NamedTuple=(;),
    filenames::Union{Nothing,AbstractVector{String}}=nothing)
    summaries_df = select(θ_summaries_df, Not(vcat(simulation_ID, θ_symbols...)))
    summary_symbols = propertynames(summaries_df)

    fig = @pipe (θ_symbols, summary_symbols) .|> length(_) * subplot_resolution |>  Figure(resolution=_) # base fig resolution on number of subplots
    for (i,summary_statistic) in enumerate(summary_symbols)
        for (j,θ) in enumerate(θ_symbols)
            fig[i,j] = Axis(fig, xlabel="$(θ)", ylabel="$(summary_statistic)")
            scatter!(fig[i,j], θ_summaries_df[1:n_plot, θ], summaries_df[1:n_plot, summary_statistic]; scatter_kwargs...)
        end
    end


    if savefig
        isnothing(filenames) && (filenames= @pipe (["svg","png"] .|> savename("scatterplots",@dict(summaries=(summary_symbols .|> String |> join)),_) .|> plotsdir(_)))
        for filename in filenames
            @info "saving $(filename)"
            safesave(filename ,fig)
        end
    end
    fig 
end



# not used anymore
function plot_distances!(fig::Figure, gdf::GroupedDataFrame, θ_prior_samples::AbstractDataFrame,  i::Int64 = 1)
    scatter!(fig[1,1], log10.(gdf[i][!,:dist]), (gdf[i][!,:sheep_reproduce].-θ_prior_samples[gdf[i].i_obs,:sheep_reproduce]),label = "$(θ_prior_samples[i_obs_vec[i],:sheep_reproduce])", alpha =0.1)
    scatter!(fig[2,1], log10.(gdf[i][!,:dist]), (gdf[i][!,:wolf_reproduce].-θ_prior_samples[gdf[i].i_obs,:wolf_reproduce]), label = "$(θ_prior_samples[i_obs_vec[i],:wolf_reproduce])", alpha =0.1)
    fig
end

function plot_distances(gdf::GroupedDataFrame, θ_prior_samples::AbstractDataFrame, i::Int64 = 1)
    fig = Figure()
    fig[1,1] = Axis(fig, title="sheep_reproduce ", ylabel="dev from true", limits = (nothing, nothing, -0.5,0.5))
    fig[2,1] = Axis(fig, title="wolf_reproduce  ", xlabel="log distance", ylabel="dev from true", limits = (nothing, nothing, -0.25,0.25))
    scatter!(fig[1,1], log10.(gdf[i][!,:dist]), (gdf[i][!,:sheep_reproduce].-θ_prior_samples[gdf[i].i_obs,:sheep_reproduce]), label = "$(θ_prior_samples[i_obs_vec[i],:sheep_reproduce])", alpha=0.1)
    scatter!(fig[2,1], log10.(gdf[i][!,:dist]), (gdf[i][!,:wolf_reproduce].-θ_prior_samples[gdf[i].i_obs,:wolf_reproduce]), label = "$(θ_prior_samples[i_obs_vec[i],:wolf_reproduce])", alpha=0.1)
    fig
end