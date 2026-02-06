export Serial, ThreadsEx

"""
    Serial()

Uses serial execution when computing the intensitymap or visibilitymap
"""
struct Serial end

"""
    ThreadsEx(;scheduler::Symbol = :dynamic)

Uses Julia's Threads @threads macro when computing the intensitymap or visibilitymap.
You can choose from Julia's various schedulers by passing the scheduler as a parameter.
The default is :dynamic, but it isn't considered part of the stable API and may change
at any moment.
"""
struct ThreadsEx{S} end
ThreadsEx() = ThreadsEx(:dynamic)
ThreadsEx(s) = ThreadsEx{s}()

"""
    ReactantEx
Uses Reactant.jl for execution when computing the intensitymap or visibilitymap. Note that specifying
this should be unnecessary as ComradeBase will automatically switch to this backend when
it detects that it is inside a Reactant tracing context.
"""
struct ReactantEx end


#TODO can this be made nicer?
@static if VERSION ≥ v"1.11"
    const schedulers = (:(:dynamic), :(:static), :(:greedy))
else
    const schedulers = (:(:dynamic), :(:static))
end

"""
    @threaded executor expr

Threads the for-loop expression `expr` using the specified `executor`. The executor must be one of
`ThreadsEx` or `Serial`. Note that if the `Threads.nthreads() == 1` we automatically default to 
a regular for-loop to prevent overhead.
"""
macro threaded(executor, expr)
    return esc(
        quote
            if Threads.nthreads() > 1 && $(executor) != Serial()
                if $(executor) == ThreadsEx{:static}()
                    Threads.@threads :static $(expr)
                elseif $(executor) == ThreadsEx{:dynamic}()
                    Threads.@threads :dynamic $(expr)
                end
            else
                $(expr)
            end
        end
    )
end

macro threaded(expr)
    return :(@threaded(ThreadsEx(), $(expr)))
end
