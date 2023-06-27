using StatsBase
using DataFrames
using Distances

using SingularSpectrumAnalysis, FFTW
using Distances, DynamicAxisWarping
import MLJ
import PartialLeastSquaresRegressor: PLSRegressor, PLS2Model


# # Summary statistics (see: summary statistics.jl)
#######################################################################################################################################################

go_extinct(x::AbstractVector) = last(x) == 0
coexist(s::AbstractVector, w::AbstractVector) = ~go_extinct(s) & ~go_extinct(w)
nonconstant_TS(s::AbstractVector, w::AbstractVector) = ~any( ([s,w] .|> unique .|> length) .== 1)

function multiple_innerjoin(dfs::Vector{D}; on::Symbol=:ensemble) where {D<:AbstractDataFrame}
    joined_df = dfs[1]
    for i in 2:length(dfs)
            joined_df = innerjoin(joined_df, dfs[i] ; on = on) 
    end
    joined_df
end


function unstack_and_join(df::AbstractDataFrame, cols::Vector{Symbol}, rowkey::Symbol, colkey::Symbol)
    @pipe cols .|> (unstack(df, rowkey, colkey, _ , renamecols=(x->"$(_)_$(x)")) |> disallowmissing) |> 
            multiple_innerjoin(_, on=rowkey)
end

function multiple_unstack(df::AbstractDataFrame, cols::Vector{Symbol}, colkey::Symbol)
    @pipe cols .|> unstack(df[!, [colkey, _]], colkey, _ , renamecols=(x->"$(_)_$(x)")) .|> disallowmissing 
end


"""
norm_absfft_λSSA(x::AbstractArray, L::Int64)

Function to compute summary statistics with lag L of a one-dimesnional signal x.
x is first normalized, after which a discrete fourier transformation and singular spectrum analysis are performed.
The summaries returend are the first L values of the DFT and the eigenvalues of the 1-L SSA eigenvalues.

# Arguments:
- `x`: Signal
- `L`: time lag (with L < length(x))
"""
function norm_absfft_λSSA(x::AbstractArray, L::Int64)
    signal = normalize(x .- mean(x)) 
    abs.(rfft(signal)[2:L+1]), hsvd(signal, L).S
end


"""
time_series_summaries(s_π₀::GroupedDataFrame, config::Dict; simulationID::Symbol=:ensemble)

Computes the following summary statistics of the wolf/sheep count time series 
- 1-`lag_max` lagged autocorrelations -> :ρ_...
- the 1-`L`` absolute values of fourier coefficients -> :F_abs...
- the 1-`L` SSA eigenvalues -> :λ...
- the first 4 moments of each TS' marginal distribution -> :count...

`s_π₀` is the dataframe containing the summaries for every simulation, that is grouped per simulation

`lag_max` and `L` are passed in the `config` dict

`simulationID` is the column on which the dataframe containing the summaries is grouped 
"""
function time_series_summaries(s_π₀::GroupedDataFrame, config::Dict; 
    simulationID::Symbol=:ensemble)

    lag_max = config[:lag]
    lags=collect(1:lag_max)
    println("Compute autocorrelations...")
    autocor_df = @time combine(s_π₀) do df
        (lags=lags, 
        ρ_sheep=autocor(df.count_sheep, lags), 
        ρ_wolf=autocor(df.count_wolf, lags))
    end

    L = config[:L]
    println("Compute fft and SSA...")
    println()
    fft_SSA_df = @time combine(s_π₀) do df
            print("$(df[1,simulationID]),")
            F_abs_sheep, λ_sheep = norm_absfft_λSSA(df.count_sheep,L)
            F_abs_wolf, λ_wolf= norm_absfft_λSSA(df.count_wolf,L)
    
            (L=collect(1:L), 
            F_abs_sheep=F_abs_sheep,
            F_abs_wolf=F_abs_wolf,
            λ_sheep=λ_sheep,
            λ_wolf=λ_wolf)
    end

    println("Compute distribution stats...")
    distr_stats_df = @time combine(s_π₀, [s => f for (s,f) in Iterators.product([:count_sheep,:count_wolf],[mean, std, skewness, kurtosis]) |> collect |> vec]...)

    X_observed_df = let
        X_autocorr_df = unstack_and_join(autocor_df, [:ρ_sheep, :ρ_wolf], simulationID, :lags) 
        X_fft_df = unstack_and_join(fft_SSA_df, [:F_abs_sheep, :F_abs_wolf, :λ_sheep, :λ_wolf], simulationID, :L) 
        multiple_innerjoin([X_autocorr_df, X_fft_df, distr_stats_df], on=simulationID)
        end
    X_observed_df
end



function latent_variable_SSE(
    model::PLS2Model,
    X_df::AbstractDataFrame,
    Y_df::AbstractDataFrame,
    nfactors::Int64)
    W,Q,P    = model.W,model.Q,model.P
    X = Matrix(X_df)
    Y_obs = Matrix(Y_df)
    ΔY = zeros(nfactors,size(Y_df,2))
    Y = zeros(size(Y_df))
    #println("nfactors: ",nfactors)
    for i = 1:nfactors
    R       = X*W[:,i]
    X       = X - R * P[:,i]'
    Y       = Y + R * Q[:,i]'

    ΔY[i,:] = colwise(SqEuclidean(),Y,Y_obs)
    end

return ΔY
end

function latent_variable_cor(
    model::PLS2Model,
    X_df::AbstractDataFrame,
    Y_df::AbstractDataFrame,
    nfactors::Int64)
    W,Q,P    = model.W,model.Q,model.P
    X = Matrix(X_df)
    Y_obs = Matrix(Y_df)
    ΔY = zeros(nfactors,size(Y_df,2))
    Y = zeros(size(Y_df))
    #println("nfactors: ",nfactors)
    for i = 1:nfactors
    R       = X*W[:,i]
    X       = X - R * P[:,i]'
    Y       = Y + R * Q[:,i]'

    ΔY[i,:] = cor(Y,Y_obs) |> diag
    end

return ΔY
end




# # Distance functions
#######################################################################################################################################################

"""
Computes the summed discrepancy measures between the summary statsitics of the prior simulations and those of the observed data.

`s_π₀`  : grouped dataframe with simulated summary statistics 
`s_obs` : dataframe with summary statsitics of the observations
`d`     : array with discrepancy measure for every summary statistic
"""
function total_discrepancy(d::AbstractArray{D}, s_π₀::GroupedDataFrame, s_obs::AbstractDataFrame; summaries::Vector{Symbol}, keepkeys::Bool=false) where {D<:Union{Function, PreMetric}}

    combine(s_π₀, summaries => 
            ((cols...) -> sum([d[i](cols[i], s_obs[!,s]) for (i,s) in enumerate(summaries)])) => 
            :dist, keepkeys=keepkeys)
end
function total_discrepancy(d::Union{Function, PreMetric}, s_π₀::GroupedDataFrame, s_obs::AbstractDataFrame; summaries::Vector{Symbol}, keepkeys::Bool=false) 
    combine(s_π₀, summaries => 
            ((cols...) -> sum([d(cols[i], s_obs[!,s]) for (i,s) in enumerate(summaries)])) => 
            :dist, keepkeys=keepkeys)
end




# # Validation tools for simple rejection ABC 
#######################################################################################################################################################

"""
Calculates the distances for rejection ABC as a leave-one-out coss validation procedure, where samples of the prior simulations are iteratively used as pseudo-observations.

summaries_π₀ : grouped dataframe with simulated summary statistics (from prior)

returns:
- posteriors: dataframe of accepted posteriors with columns:
    - i_obs:    index of pseudo-obs
    - ensemble: index of acccepted posterior sample
    - dist:     distance of accepted posterior sample
- tolerances: dataframe of with ϵ values
"""
function alldistances_LOOCV(summaries_π₀::GroupedDataFrame, 
    discrepancy_measure::Function, 
    n_LOOCV::Int, 
    i_obs_vec::Union{Nothing,AbstractArray} = nothing)


    gdf_indices = eachindex(summaries_π₀) .|> first # get all simulation IDs
    i_obs_vec = (i_obs_vec === nothing) ? sample(gdf_indices, n_LOOCV, replace = false) : i_obs_vec

    # pre-allocate dataframes
    n_prior_samples = length(gdf_indices)
    n_post_samples = n_prior_samples-1
    distances = DataFrame(i_obs = zeros(Int64, n_LOOCV*n_post_samples), 
                           ensemble = zeros(Int64, n_LOOCV*n_post_samples), 
                           dist = zeros(Float64, n_LOOCV*n_post_samples))
    for i in 1:n_LOOCV
        i_obs = i_obs_vec[i]
        dist = @pipe discrepancy_measure(summaries_π₀[gdf_indices .!= i_obs], summaries_π₀[(i_obs,)]) |> sort( _ , :dist)
        distances[1+(i-1)*n_post_samples:i*n_post_samples,:i_obs] .= i_obs
        distances[1+(i-1)*n_post_samples:i*n_post_samples, 2:3] = dist[1:n_post_samples,:]
    end
    distances
end

"""
Calculates the distances for rejection ABC as a leave-one-out coss validation procedure, where samples of the prior simulations are iteratively used as pseudo-samples.

summaries_π₀ : grouped dataframe with simulated summary statistics (from prior)

returns:
- posteriors: dataframe of accepted posteriors with columns:
    - i_obs:    index of pseudo-obs
    - ensemble: index of acccepted posterior sample
    - dist:     distance of accepted posterior sample
- tolerances: dataframe of with ϵ values
"""
function alldistances_LOOCV(summaries_π₀::AbstractDataFrame, 
    discrepancy_measure::Union{Function, PreMetric}, 
    n_LOOCV::Int, 
    i_obs_vec::Union{Nothing,AbstractArray} = nothing;
    sim_id::Symbol = :ensemble)


    sim_indices = summaries_π₀[!,sim_id]
    i_obs_vec = (i_obs_vec === nothing) ? sample(sim_indices, n_LOOCV, replace = false) : i_obs_vec

    # pre-allocate dataframes
    n_prior_samples = length(sim_indices)
    n_post_samples = n_prior_samples-1
    distances = DataFrame(:i_obs => zeros(Int64, n_LOOCV*n_post_samples), 
                            sim_id => zeros(Int64, n_LOOCV*n_post_samples),
                           :dist => zeros(Float64, n_LOOCV*n_post_samples))
    for i in 1:n_LOOCV
        i_obs = i_obs_vec[i]
        summ_obs = @pipe summaries_π₀[sim_indices .== i_obs, Not(sim_id)] |> eachrow |> first |> [values(_)...]
        d(v...) = discrepancy_measure([v...], summ_obs)
        dist = @pipe select(summaries_π₀[sim_indices .!= i_obs, :] , sim_id, Not(sim_id) => ByRow(d) => :dist) |> 
                      sort( _ , :dist)
        distances[1+(i-1)*n_post_samples:i*n_post_samples,:i_obs] .= i_obs
        distances[1+(i-1)*n_post_samples:i*n_post_samples, sim_id] = dist[1:n_post_samples, sim_id]
        distances[1+(i-1)*n_post_samples:i*n_post_samples, :dist] = dist[1:n_post_samples,:dist]
    end
    distances
end






"""
Performs rejection ABC as a leave-one-out coss validation procedure, where samples of the prior simulations are iteratively used as pseudo-samples.

summaries_π₀ : grouped dataframe with simulated summary statistics (from prior)

returns:
- posteriors: dataframe of accepted posteriors with columns:
    - i_obs:    index of pseudo-obs
    - ensemble: index of acccepted posterior sample
    - dist:     distance of accepted posterior sample
- tolerances: dataframe of with ϵ values
"""
function rejection_LOOCV(summaries_π₀::GroupedDataFrame, 
    discrepancy_measure::Function, 
    n_LOOCV::Int, 
    i_obs_vec::Union{Nothing,AbstractArray} = nothing; 
    q_ϵ::Float64=0.01)


    gdf_indices = eachindex(summaries_π₀) .|> first
    i_obs_vec = (i_obs_vec === nothing) ? sample(gdf_indices, n_LOOCV, replace = false) : i_obs_vec

    # pre-allocate dataframes
    n_prior_samples = length(gdf_indices) - 1
    n_post_samples = q_ϵ * n_prior_samples |> floor |> Int

    ϵ = zeros(n_LOOCV)
    posteriors = DataFrame(i_obs = zeros(Int64, n_LOOCV*n_post_samples), 
                           ensemble = zeros(Int64, n_LOOCV*n_post_samples), 
                           dist = zeros(Float64, n_LOOCV*n_post_samples))
    for i in 1:n_LOOCV
        i_obs = i_obs_vec[i]
        dist = @pipe discrepancy_measure(summaries_π₀[gdf_indices .!= i_obs], summaries_π₀[(i_obs,)]) |> sort( _ , :dist)
        ϵ[i] = dist.dist[n_post_samples]
        posteriors[1+(i-1)*n_post_samples:i*n_post_samples,:i_obs] .= i_obs
        posteriors[1+(i-1)*n_post_samples:i*n_post_samples, 2:3] = dist[1:n_post_samples,:]
    end
    posteriors, DataFrame(i_obs = i_obs_vec , ϵ = ϵ)
end



"""
estimate of p₀ 
Prangle et al (2014): this estimation is equivalent to the posterior mean for the binomial probability that θ⁽ˡ⁾ < θ₀  under a uniform prior
"""
estimate_p₀(θ_accepted::Vector{Float64}, θ₀::Float64, nⱼ::Int64 = length(θ_accepted)) =  (1 + sum(θ_accepted[1:nⱼ] .≤ θ₀)) /(2 + nⱼ)


"""
summaries_π₀ : dataframe with simulated summary statistics (from prior)
θ_π₀ : dataframe with prior parameters samples (col names: to be inferred parameters)
"""
function rejection_LOOCV_fixed_tol(summaries_π₀::GroupedDataFrame,  
    θ_π₀::AbstractDataFrame, 
    discrepancy_measure::Function, 
    n_LOOCV::Int, 
    ϵ::Number,
    i_obs_vec::Union{Nothing,AbstractArray} = nothing; 
    θ::Union{Nothing,AbstractArray{Symbol}}=nothing)

    i_obs_vec = (i_obs_vec === nothing) ? sample(1:length(summaries_π₀), n_LOOCV, replace = false) : i_obs_vec
    θ = (θ === nothing) ? propertynames(θ_π₀) : θ
    n_accepted = zeros(Int, n_LOOCV)
    p₀ = zeros(n_LOOCV,length(θ))
    for i in 1:n_LOOCV
        i_obs = i_obs_vec[i]
        dist = discrepancy_measure(summaries_π₀[1:end .!= i_obs], summaries_π₀[i_obs])
        θ_accepted = θ_π₀[1:end .!= i_obs, θ][dist .<= ϵ, :]
        p₀[i,:], n_accepted[i]  = estimate_p₀(θ_accepted, θ_π₀[i_obs,θ])
    end
    p₀_df = DataFrame(p₀, θ)
    p₀_df[!,:i_obs] = i_obs_vec
    p₀_df[!,:n_accepted] = n_accepted
    p₀_df
end



