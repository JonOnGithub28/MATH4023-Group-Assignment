#include("Takt.jl"), using .Takt
using JuMP
using HiGHS

P = Takt.Problem(1)
nE, nA = number_of_events(P), number_of_activities(P)

m = Model(HiGHS.Optimizer)
set_optimizer_attribute(m, "time_limit", 3600.0)   # seconds, tune to taste
set_optimizer_attribute(m, "mip_rel_gap", 0.0)   # stop once within 1% — matches gap()
set_optimizer_attribute(m, "presolve", "on")
set_optimizer_attribute(m, "parallel", "on")
set_optimizer_attribute(m, "threads", Sys.CPU_THREADS)

@variable(m, 0 <= pi[e=1:nE] <= period(P, e) - 1, Int)
@variable(m, k[a=1:nA], Int)
@variable(m, d[a=1:nA])          # duration of each activity

for a in 1:nA
    f, t  = events_of(P, a)
    mod_a = modulus(P, a)
    @constraint(m, d[a] == pi[t] - pi[f] - mod_a * k[a])
    @constraint(m, lower(P, a) <= d[a] <= upper(P, a))
end

@objective(m, Min, sum(weight(P, a) * (d[a] - lower(P, a)) for a in 1:nA))

  t0 = times(published(P))
  set_start_value.(pi, t0)

optimize!(m)

t = round.(Int, value.(pi))
S = Solution(P, t)
is_feasible(S), score(S)