using JuMP, HiGHS
using Takt

P = Problem(3)

nE = number_of_events(P)
nA = number_of_activities(P)

lo_k = Vector{Int}(undef, nA)
hi_k = Vector{Int}(undef, nA)

# Basic bounds
for a in 1:nA
    f, t  = events_of(P, a)
    mod_a = modulus(P, a)
    lo_k[a] = ceil(Int, (lower(P,a) + 1 - period(P,f)) / mod_a)
    hi_k[a] = floor(Int,  (upper(P,a) - 1 + period(P,t)) / mod_a)
end

m = Model(HiGHS.Optimizer)
set_optimizer_attribute(m, "time_limit", 600.0)
set_optimizer_attribute(m, "mip_rel_gap", 0.0)
set_optimizer_attribute(m, "presolve", "on")
set_optimizer_attribute(m, "parallel", "on")
set_optimizer_attribute(m, "threads", Sys.CPU_THREADS)
set_optimizer_attribute(m, "user_objective_scale", -2)

@variable(m, 0 <= pi_[e=1:nE] <= period(P, e) - 1)   # `pi` shadows Base.pi, so `pi_`
@variable(m, lo_k[a] <= k[a=1:nA] <= hi_k[a], Int)

d_expr = Vector{AffExpr}(undef, nA)
for a in 1:nA
    f, t = events_of(P, a)
    d_expr[a] = pi_[t] - pi_[f] + modulus(P, a) * k[a]
    @constraint(m, lower(P, a) <= d_expr[a] <= upper(P, a))
end

@objective(m, Min, sum(weight(P, a) * (d_expr[a] - lower(P, a)) for a in 1:nA))

t0 = times(published(P))
set_start_value.(pi_, t0)

optimize!(m)

T = round.(Int, value.(pi_))
S = Solution(P, T)

@show is_feasible(S)
@show score(S)
@show gap(S, objective_bound(m))

#savesolution(S)