# # Predator-prey dynamics
# source: Agents.jl v5.5.0.
# The predator-prey model emulates the population dynamics of predator and prey animals who
# live in a common ecosystem and compete over limited resources. This model is an
# agent-based analog to the classic
# [Lotka-Volterra](https://en.wikipedia.org/wiki/Lotka%E2%80%93Volterra_equations)
# differential equation model.

# This example illustrates how to develop models with
# heterogeneous agents (sometimes referred to as a *mixed agent based model*),
# incorporation of a spatial property in the dynamics (represented by a stanadard
# array, not an agent, as is done in most other ABM frameworks),
# and usage of [`GridSpace`](@ref), which allows multiple agents per grid coordinate.

# ## Model specification
# The environment is a two dimensional grid containing sheep, wolves and grass. In the
# model, wolves eat sheep and sheep eat grass. Their populations will oscillate over time
# if the correct balance of resources is achieved. Without this balance however, a
# population may become extinct. For example, if wolf population becomes too large,
# they will deplete the sheep and subsequently die of starvation.

# We will begin by loading the required packages and defining two subtypes of
# `AbstractAgent`: `Sheep`, `Wolf`. Grass will be a spatial property in the model.  All three agent types have `id` and `pos`
# properties, which is a requirement for all subtypes of `AbstractAgent` when they exist
# upon a `GridSpace`. Sheep and wolves have identical properties, but different behaviors
# as explained below. The property `energy` represents an animals current energy level.
# If the level drops below zero, the agent will die. Sheep and wolves reproduce asexually
# in this model, with a probability given by `reproduction_prob`. The property `Δenergy`
# controls how much energy is acquired after consuming a food source.

# Grass is a replenishing resource that occupies every position in the grid space. Grass can be
# consumed only if it is `fully_grown`. Once the grass has been consumed, it replenishes
# after a delay specified by the property `regrowth_time`. The property `countdown` tracks
# the delay between being consumed and the regrowth time.

# It is also available from the `Models` module as [`Models.predator_prey`](@ref).

# ## Making the model
# First we define the agent types
# (here you can see that it isn't really that much
# of an advantage to have two different agent types. Like in the [Rabbit, Fox, Wolf](@ref)
# example, we could have only one type and one additional filed to separate them.
# Nevertheless, for the sake of example, we will use two different types.)
using Agents, Random

@agent Sheep GridAgent{2} begin
    energy::Float64
    reproduction_prob::Float64
    Δenergy::Float64
end

@agent Wolf GridAgent{2} begin
    energy::Float64
    reproduction_prob::Float64
    Δenergy::Float64
end

# The function `initialize_model` returns a new model containing sheep, wolves, and grass
# using a set of pre-defined values (which can be overwritten). The environment is a two
# dimensional grid space, which enables animals to walk in all
# directions.

function initialize_model(;
        n_sheep = 100,
        n_wolves = 50,
        dims = (20, 20),
        regrowth_time = 30,
        Δenergy_sheep = 4,
        Δenergy_wolf = 20,
        sheep_reproduce = 0.04,
        wolf_reproduce = 0.05,
        seed = 23182,
    )

    rng = MersenneTwister(seed)
    space = GridSpace(dims, periodic = true)
    ## Model properties contain the grass as two arrays: whether it is fully grown
    ## and the time to regrow. Also have static parameter `regrowth_time`.
    ## Notice how the properties are a `NamedTuple` to ensure type stability.
    properties = (
        fully_grown = falses(dims),
        countdown = zeros(Int, dims),
        regrowth_time = regrowth_time,
    )
    model = ABM(Union{Sheep, Wolf}, space;
        properties, rng, scheduler = Schedulers.randomly, warn = false
    )
    ## Add agents
    for _ in 1:n_sheep
        energy = rand(model.rng, 1:(Δenergy_sheep*2)) - 1
        add_agent!(Sheep, model, energy, sheep_reproduce, Δenergy_sheep)
    end
    for _ in 1:n_wolves
        energy = rand(model.rng, 1:(Δenergy_wolf*2)) - 1
        add_agent!(Wolf, model, energy, wolf_reproduce, Δenergy_wolf)
    end
    ## Add grass with random initial growth
    for p in positions(model)
        fully_grown = rand(model.rng, Bool)
        countdown = fully_grown ? regrowth_time : rand(model.rng, 1:regrowth_time) - 1
        model.countdown[p...] = countdown
        model.fully_grown[p...] = fully_grown
    end
    return model
end

# ## Defining the stepping functions
# Sheep and wolves behave similarly:
# both lose 1 energy unit by moving to an adjacent position and both consume
# a food source if available. If their energy level is below zero, they die.
# Otherwise, they live and reproduce with some probability.
# They move to a random adjacent position with the [`walk!`](@ref) function.

# Notice how the function `sheepwolf_step!`, which is our `agent_step!`,
# is dispatched to the appropriate agent type via Julia's Multiple Dispatch system.
function sheepwolf_step!(sheep::Sheep, model)
    walk!(sheep, rand, model)
    sheep.energy -= 1
    if sheep.energy < 0
        kill_agent!(sheep, model)
        return
    end
    eat!(sheep, model)
    if rand(model.rng) ≤ sheep.reproduction_prob
        reproduce!(sheep, model)
    end
end

function sheepwolf_step!(wolf::Wolf, model)
    walk!(wolf, rand, model;ifempty=false)
    wolf.energy -= 1
    if wolf.energy < 0
        kill_agent!(wolf, model)
        return
    end
    ## If there is any sheep on this grid cell, it's dinner time!
    dinner = first_sheep_in_position(wolf.pos, model)
    !isnothing(dinner) && eat!(wolf, dinner, model)
    if rand(model.rng) ≤ wolf.reproduction_prob
        reproduce!(wolf, model)
    end
end

function first_sheep_in_position(pos, model)
    ids = ids_in_position(pos, model)
    j = findfirst(id -> model[id] isa Sheep, ids)
    isnothing(j) ? nothing : model[ids[j]]::Sheep
end

# Sheep and wolves have separate `eat!` functions. If a sheep eats grass, it will acquire
# additional energy and the grass will not be available for consumption until regrowth time
# has elapsed. If a wolf eats a sheep, the sheep dies and the wolf acquires more energy.
function eat!(sheep::Sheep, model)
    if model.fully_grown[sheep.pos...]
        sheep.energy += sheep.Δenergy
        model.fully_grown[sheep.pos...] = false
    end
    return
end

function eat!(wolf::Wolf, sheep::Sheep, model)
    kill_agent!(sheep, model)
    wolf.energy += wolf.Δenergy
    return
end

# Sheep and wolves share a common reproduction method. Reproduction has a cost of 1/2 the
# current energy level of the parent. The offspring is an exact copy of the parent, with
# exception of `id`.
function reproduce!(agent::A, model) where {A}
    agent.energy /= 2
    id = nextid(model)
    offspring = A(id, agent.pos, agent.energy, agent.reproduction_prob, agent.Δenergy)
    add_agent_pos!(offspring, model)
    return
end

# The behavior of grass function differently. If it is fully grown, it is consumable.
# Otherwise, it cannot be consumed until it regrows after a delay specified by
# `regrowth_time`. The dynamics of the grass is our `model_step!` function.
function grass_step!(model)
    @inbounds for p in positions(model) # we don't have to enable bound checking
        if !(model.fully_grown[p...])
            if model.countdown[p...] ≤ 0
                model.fully_grown[p...] = true
                model.countdown[p...] = model.regrowth_time
            else
                model.countdown[p...] -= 1
            end
        end
    end
end

# Datacollection functions
sheep(a) = a isa Sheep
wolf(a) = a isa Wolf
count_grass(model) = count(model.fully_grown)

function stop_if_extinct(model::ABM, s::Int; s_max::Int)
    if s ≥ s_max
        return true
    end

    n_sheep = [sheep(a) for a in allagents(model)] |> sum
    if (n_sheep == 0) | (n_sheep == length(allagents(model)))
        return true
    else
        return false
    end
end
