module ComradeBaseOhMyThreadsExt

using ComradeBase
using OhMyThreads

function ComradeBase._pointmap!(img, f, d, executor::OhMyThreads.Scheduler)
    dest = baseimage(img)
    g = domainpoints(d)
    cis = ComradeBase._pointindices(dest, g)
    @tasks for i in eachindex(IndexLinear(), cis)
        @set scheduler = executor
        I = cis[i]
        ComradeBase._setpoint!(dest, I, f(g[I]))
    end
    return nothing
end

end
