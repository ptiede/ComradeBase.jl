const AMeta = DimensionalData.Dimensions.Lookups.AbstractMetadata

abstract type AbstractHeader{T, X} <: AMeta{T, X} end

"""
    MinimalHeader{T}

A minimal header type for ancillary image information.

# Fields
$(FIELDS)
"""
struct MinimalHeader{T} <: AbstractHeader{T, NamedTuple{(), Tuple{}}}
    """
    Common source name
    """
    source::String
    """
    Right ascension of the image in degrees (J2000)
    """
    ra::T
    """
    Declination of the image in degrees (J2000)
    """
    dec::T
    """
    Modified Julian Date in days
    """
    mjd::T
    """
    Frequency of the image in Hz
    """
    frequency::T
end

function MinimalHeader(source, ra, dec, mjd, freq)
    raT, decT, mjdT, freqT = promote(ra, dec, mjd, freq)
    return MinimalHeader(source, raT, decT, mjdT, freqT)
end

function DimensionalData.val(m::AbstractHeader)
    n = propertynames(m)
    pm = Base.Fix1(getproperty, m)
    return NamedTuple{n}(map(pm, n))
end

"""
    NoHeader


"""
const NoHeader = DimensionalData.NoMetadata
