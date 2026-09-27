#include("Takt.jl"), using .Takt
using JuMP, HiGHS

# ---------------------------------------------------------------------
# 1. Pick the problem
# ---------------------------------------------------------------------
P = Problem(2)   # change to whichever problem number you're solving

nE = number_of_events(P)
nA = number_of_activities(P)

# ---------------------------------------------------------------------
# 2. Derive genuine (provably valid) bounds on k[a] from the problem
#    data itself, rather than guessing a blanket range.
#
#    k[a] = (pi[to] - pi[from] - d[a]) / modulus(a)
#    with pi bounded by each event's period, and d bounded by
#    [lower(a), upper(a)].
# ---------------------------------------------------------------------
lo_k = Vector{Int}(undef, nA)
hi_k = Vector{Int}(undef, nA)

for a in 1:nA
    f, t  = events_of(P, a)
    mod_a = modulus(P, a)
    lo_k[a] = floor(Int, ( -(period(P, f) - 1) - upper(P, a) ) / mod_a)
    hi_k[a] = ceil(Int,  (  (period(P, t) - 1) - lower(P, a) ) / mod_a)
end

# Sanity check: bounds should never be inverted. If they are, something
# is wrong upstream (e.g. a mismatched from/to/period), not with the
# solve itself.
bad = findfirst(a -> lo_k[a] > hi_k[a], 1:nA)
bad === nothing || error("infeasible k bound at activity $bad")

# ---------------------------------------------------------------------
# 3. Build the model
# ---------------------------------------------------------------------
m = Model(HiGHS.Optimizer)
set_optimizer_attribute(m, "time_limit", 3600.0)   # tune to taste
set_optimizer_attribute(m, "mip_rel_gap", 0.0)    # aim for ~0% gap
#Bunch of settings that should make it run faster
set_optimizer_attribute(m, "presolve", "on")
set_optimizer_attribute(m, "parallel", "on")
set_optimizer_attribute(m, "threads", Sys.CPU_THREADS)
set_optimizer_attribute(m, "user_objective_scale", -2)

@variable(m, 0 <= pi_[e=1:nE] <= period(P, e) - 1, Int)   # `pi` shadows Base.pi, so `pi_`
@variable(m, lo_k[a] <= k[a=1:nA] <= hi_k[a], Int)

# d[a] is eliminated algebraically: instead of a variable tied down by
# an equality constraint, it's just an affine expression built from
# pi_ and k. HiGHS's presolve would likely fold the variable form away
# anyway, but this keeps the model itself smaller and more explicit.
d_expr = Vector{AffExpr}(undef, nA)
for a in 1:nA
    f, t = events_of(P, a)
    d_expr[a] = pi_[t] - pi_[f] - modulus(P, a) * k[a]
    @constraint(m, lower(P, a) <= d_expr[a] <= upper(P, a))
end

@objective(m, Min, sum(weight(P, a) * (d_expr[a] - lower(P, a)) for a in 1:nA))

# ---------------------------------------------------------------------
# 4. Warm start from the published (known-feasible) solution
# ---------------------------------------------------------------------
t0 = times(published(P))
set_start_value.(pi_, t0)

# Sanity-check the derived k bounds actually contain the published
# solution's implied k values. If this assertion ever fails, it means
# the k bound derivation has a bug -- re-check the from/to/period
# indexing above before trusting the solve.
for a in 1:nA
    f, t  = events_of(P, a)
    mod_a = modulus(P, a)
    d0 = duration(published(P), a)
    k0 = (t0[t] - t0[f] - d0) / mod_a
    @assert lo_k[a] <= k0 <= hi_k[a] "bound too tight at activity $a"
end

# ---------------------------------------------------------------------
# 5. Solve
# ---------------------------------------------------------------------
optimize!(m)

println("relative gap = ", relative_gap(m))
println("objective    = ", objective_value(m))
println("bound        = ", objective_bound(m))

# ---------------------------------------------------------------------
# 6. Convert back to a Takt Solution and verify
# ---------------------------------------------------------------------
t = round.(Int, value.(pi_))
S = Solution(P, t)

@show is_feasible(S)
@show score(S)
@show gap(S, objective_bound(m))