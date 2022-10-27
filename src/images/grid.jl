"""
    AbstractGrid

An abstract grid that specifies the location of uv points, or image locations. This
grid can be irregular and even non grid like.
"""
abstract type AbstractGrid{F,N} end

Base.IteratorSize(::AbstractGrid{F,N}) where {F,N} = Base.HasShape{N}()
Base.IteratorEltype(::AbstractGrid{F,N}) where {F,N} = Base.HasEltype()

Base.eltype(::AbstractGrid{F,N}) where {F,N} = NTuple{N,F}


struct RectangularSpatialGrid{F,2} <: AbstractGrid{F,2}
    xitr::I
    yitr::I
end

function RectangularSpatialGrid(fovx::Real, fovy::Real, nx::Int, ny::Int, x0::Real = 0.0, y0::Real = 0.0)
    psizex=fovx/max(nx-1,1)
    psizey=fovy/max(ny-1,1)

    xitr = LinRange(-fovx/2 + psizex/2 - x0, fovx/2 - psizex/2, nx)
    yitr = LinRange(-fovy/2 + psizey/2 - y0, fovy/2 - psizey/2, ny)
    return new{F}(
            xitr, yitr
            )
end

function RectangularSpatialGrid(fov::Real, npix::Int, x0 = 0.0, y0 = 0.0)
    return RectangularSpatialGrid(fov, fov, npix, npix, x0, y0)
end

@inline function imagepixels(grid::RectangularSpatialGrid)
    return (ra = grid.xitr, dec = grid.yitr)
end

@inline function pixelsize(grid::RectangularSpatialGrid)
    return (ra = step(grid.xitr), dec = step(grid.yitr))
end

@inline function fov(grid::RectangularSpatialGrid)
    (;xitr, yitr) = grid
    return (ra = last(xtir) - first(xitr), dec = last(yitr) - first(xitr))
end

Base.length(grid::RectangularSpatialGrid) = grid.nx*grid.ny
Base.size(grid::RectangularSpatialGrid) = (grid.ny, grid.nx)
Base.size(grid::RectangularSpatialGrid) = (grid.ny, grid.nx)

function center(grid::RectangularSpatialGrid)
    (;xitr, yitr) = grid
    return (ra = last(xtir) + first(xitr), dec = last(yitr) + first(xitr))

end
