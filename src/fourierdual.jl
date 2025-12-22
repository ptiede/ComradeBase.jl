"""
    $(TYPEDEF)

This defines an abstract cache that can be used to
hold or precompute some computations.
"""
abstract type AbstractFourierDualDomain <: AbstractDomain end

abstract type FourierTransform end


forward_plan(g::AbstractFourierDualDomain) = getfield(g, :plan_forward)
reverse_plan(g::AbstractFourierDualDomain) = getfield(g, :plan_reverse)
imgdomain(g::AbstractFourierDualDomain) = getfield(g, :imgdomain)
visdomain(g::AbstractFourierDualDomain) = getfield(g, :visdomain)
algorithm(g::AbstractFourierDualDomain) = getfield(g, :algorithm)

EnzymeRules.inactive(::typeof(forward_plan), args...) = nothing
EnzymeRules.inactive(::typeof(reverse_plan), args...) = nothing

abstract type AbstractPlan end
getplan(p::AbstractPlan) = getfield(p, :plan)
getphases(p::AbstractPlan) = getfield(p, :phases)
EnzymeRules.inactive(::typeof(getplan), args...) = nothing
EnzymeRules.inactive(::typeof(getphases), args...) = nothing

function create_plans(algorithm, imgdomain, visdomain)
    plan_forward = create_forward_plan(algorithm, imgdomain, visdomain)
    plan_reverse = inverse_plan(plan_forward)
    return plan_forward, plan_reverse
end

function create_vismap(arr::AbstractArray, g::AbstractFourierDualDomain)
    return ComradeBase.create_map(arr, visdomain(g))
end

function create_imgmap(arr::AbstractArray, g::AbstractFourierDualDomain)
    return ComradeBase.create_map(arr, imgdomain(g))
end

function visibilitymap_analytic(m::AbstractModel, grid::AbstractFourierDualDomain)
    return visibilitymap_analytic(m, visdomain(grid))
end

function visibilitymap_numeric(m::AbstractModel, grid::AbstractFourierDualDomain)
    img = intensitymap_analytic(m, imgdomain(grid))
    vis = applyft(forward_plan(grid), img)
    return vis
end

function intensitymap_analytic(m::AbstractModel, grid::AbstractFourierDualDomain)
    return intensitymap_analytic(m, imgdomain(grid))
end

function intensitymap_numeric(m::AbstractModel, grid::AbstractFourierDualDomain)
    # This is because I want to make a grid that is the same size as the image
    # so we revert to the standard method and not what ever was cached
    img = intensitymap_numeric(m, imgdomain(grid))
    return img
end
