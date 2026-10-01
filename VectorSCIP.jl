#Run this at the start of any session
#=
include("Takt.jl"), using .Takt
=#
using JuMP, HiGHS, SCIP

# ---------------------------------------------------------------------
# 1. Pick the problem
# ---------------------------------------------------------------------
P = Problem(3)   # change to whichever problem number you're solving

nE = number_of_events(P)
nA = number_of_activities(P)

# ---------------------------------------------------------------------
# 2. Derive genuine (provably valid) bounds on k[a] from the problem
#    data itself, rather than guessing a blanket range.
#
#    k[a] = (d[a] - pi[to] + pi[from]) / modulus(a)
#    with pi bounded by each event's period, and d bounded by
#    [lower(a), upper(a)].
# ---------------------------------------------------------------------
lo_k = Vector{Int}(undef, nA)
hi_k = Vector{Int}(undef, nA)

for a in 1:nA
    f, t  = events_of(P, a)
    mod_a = modulus(P, a)
    lo_k[a] = ceil(Int, (lower(P,a) - period(P,t) - 1) / mod_a)
    hi_k[a] = floor(Int, (upper(P,a) + period(P,f) + 1) / mod_a)
end

# Sanity check: bounds should never be inverted. If they are, something
# is wrong upstream (e.g. a mismatched from/to/period), not with the
# solve itself.
bad = findfirst(a -> lo_k[a] > hi_k[a], 1:nA)
bad === nothing || error("infeasible k bound at activity $bad")

# ---------------------------------------------------------------------
# 3. Build the model
# ---------------------------------------------------------------------
m = Model(SCIP.Optimizer)
set_optimizer_attribute(m, "limits/time", 600.0)   # tune to taste
set_optimizer_attribute(m, "limits/gap", 0.0)    # aim for ~0% gap
#Bunch of settings that should make it run faster
# 1. Core structural limits (Always supported)    
set_optimizer_attribute(m, "parallel/mode", 1) 

# 2. Symmetry handling
set_optimizer_attribute(m, "misc/usesymmetry", 7)

# 3. Direct heuristic activation (Forces aggressive behavior manually)
# setting the frequency to 1 tells SCIP to run these at every single branch node
set_optimizer_attribute(m, "heuristics/feaspump/freq", 1)  # Feasibility Pump
set_optimizer_attribute(m, "heuristics/alns/freq", 1)      # Adaptive Large Neighborhood Search
set_optimizer_attribute(m, "heuristics/rins/freq", 1)      # Relaxation Induced Neighborhood Search

@variable(m, 0 <= pi_[e=1:nE] <= period(P, e) - 1, Int)   # `pi` shadows Base.pi, so `pi_`
@variable(m, lo_k[a] <= k[a=1:nA] <= hi_k[a], Int)
t0 = times(published(P))
fix(pi_[1], t0[1]; force=true) #Break translational symmetry

# --- SUGGESTION B: CACHE LOOKUPS OUTSIDE THE CONSTRAINT GENERATOR ---
from_events = [events_of(P, a)[1] for a in 1:nA]  # Gets the 'f' event
to_events   = [events_of(P, a)[2] for a in 1:nA]  # Gets the 't' event
mods        = [modulus(P, a) for a in 1:nA]
lowers      = [lower(P, a) for a in 1:nA]
uppers      = [upper(P, a) for a in 1:nA]

# --- SUGGESTION A + B: VECTORISED CONSTRAINT WITH ZERO FOR-LOOPS ---
@constraint(m, [a=1:nA], 
    lowers[a] <= pi_[to_events[a]] - pi_[from_events[a]] + mods[a] * k[a] <= uppers[a]
)

# --- RE-DEFINED OBJECTIVE FUNCTION USING PRE-CACHED VALUES ---
@objective(m, Min, 
    sum(weight(P, a) * (pi_[to_events[a]] - pi_[from_events[a]] + mods[a] * k[a] - lowers[a]) for a in 1:nA)
)
# ---------------------------------------------------------------------
# 4. Warm start from the published (known-feasible) solution
# ---------------------------------------------------------------------
#t0 = times(published(P)) - moved higher
set_start_value.(pi_, t0)

for a in 1:nA
    f, t = from_events[a], to_events[a]
    d0 = duration(published(P), a)
    
    # Rearranged based on: d0 = pi_[t] - pi_[f] + mods[a] * k[a]
    k0 = (d0 - t0[t] + t0[f]) / mods[a]
    
    # Pass the exact integer to the solver
    set_start_value(k[a], round(Int, k0))
end

# Sanity-check the derived k bounds actually contain the published
# solution's implied k values. If this assertion ever fails, it means
# the k bound derivation has a bug -- re-check the from/to/period
# indexing above before trusting the solve.
for a in 1:nA
    f, t  = events_of(P, a)
    mod_a = modulus(P, a)
    d0 = duration(published(P), a)
    k0 = (d0 - t0[t] + t0[f]) / mod_a
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