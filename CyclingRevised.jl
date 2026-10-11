# =====================================================================
# MATH4023: Mathematical Optimisation - Swiss Taktfahrplan Solver
# Corrected Symmetry Break and Validated Cycle Space Setup
# =====================================================================

#=
include("Takt.jl")
using .Takt
=#

using JuMP, HiGHS, Graphs

# ---------------------------------------------------------------------
# 1. Problem Selection
# ---------------------------------------------------------------------
P = Problem(4)   # Tested and ready to scale up to Problems 4-12

nE = number_of_events(P)
nA = number_of_activities(P)

# --- CACHE LOOKUPS FOR INSTANT VECTORIZED OPERATIONS ---
from_events = [events_of(P, a)[1] for a in 1:nA]  
to_events   = [events_of(P, a)[2] for a in 1:nA]  
mods        = [modulus(P, a) for a in 1:nA]
lowers      = [lower(P, a) for a in 1:nA]
uppers      = [upper(P, a) for a in 1:nA]
weights     = [weight(P, a) for a in 1:nA]

# ---------------------------------------------------------------------
# 2. Genuine Rigorous Bounds on k[a]
# ---------------------------------------------------------------------
lo_k = Vector{Int}(undef, nA)
hi_k = Vector{Int}(undef, nA)

for a in 1:nA
    f, t  = from_events[a], to_events[a]
    mod_a = mods[a]
    lo_k[a] = ceil(Int, (lowers[a] - Takt.period(P, t) - 1) / mod_a)
    hi_k[a] = floor(Int, (uppers[a] + Takt.period(P, f) + 1) / mod_a)
end

# ---------------------------------------------------------------------
# 3. Model Construction & Solver Tuning
# ---------------------------------------------------------------------
m = Model(HiGHS.Optimizer)

set_optimizer_attribute(m, "time_limit", 360.0)   
set_optimizer_attribute(m, "mip_rel_gap", 0.0)    
set_optimizer_attribute(m, "mip_abs_gap", 0.0)
set_optimizer_attribute(m, "presolve", "on")
set_optimizer_attribute(m, "parallel", "on")
set_optimizer_attribute(m, "threads", Sys.CPU_THREADS)
set_optimizer_attribute(m, "mip_heuristic_effort", 0.5) 

@variable(m, 0 <= pi_[e=1:nE] <= Takt.period(P, e) - 1, Int)   
@variable(m, lo_k[a] <= k[a=1:nA] <= hi_k[a], Int)

# --- FIXED: TRUE SYMMETRY BREAK ---
# We ONLY fix the very first event node to anchor the network. 
# This breaks translation symmetry without freezing your variables.
t0 = times(published(P))
fix(pi_[1], t0[1]; force=true) 

# Main PESP Edge Constraints
@constraint(m, link[a=1:nA], 
    lowers[a] <= pi_[to_events[a]] - pi_[from_events[a]] + mods[a] * k[a] <= uppers[a]
)

# ---------------------------------------------------------------------
# 4. STRUCTURAL CYCLE CUTS (CONSTRAINING INTEGER BOUNDS DIRECTLY)
# ---------------------------------------------------------------------
g = SimpleDiGraph(nE)
afromto = Dict{Tuple{Int,Int}, Int}()

for a in 1:nA
    f, t = from_events[a], to_events[a]
    add_edge!(g, f, t) # Connect events by activities
    afromto[(f, t)] = a # Allow looking up activities by from and to event
end

cycles = simplecycles_limited_length(g, 20, 500) #Returns lists of events comprising cycles
for cycle in cycles
    sum_lower = 0.0
    sum_upper = 0.0
    int_combination = AffExpr(0.0) #Track sum mod[a]*k[a]
    push!(cycle, cycle[1])
    for j in 1:(length(cycle)-1)
        a = afromto[(cycle[j], cycle[j+1])]
        sum_lower += lowers[a]
        sum_upper += uppers[a]
        add_to_expression!(int_combination, Float64(mods[a]), k[a])
    end
    @constraint(m, sum_lower <= int_combination <= sum_upper) #New constraint for every cycle
end


# ---------------------------------------------------------------------
# 5. Objective Function & Warm Start Initialization
# ---------------------------------------------------------------------
@objective(m, Min, 
    sum(weights[a] * (pi_[to_events[a]] - pi_[from_events[a]] + mods[a] * k[a] - lowers[a]) for a in 1:nA)
)

# Seed the solver using the entire published timetable as a warm start
set_start_value.(pi_, t0)
for a in 1:nA
    f, t = from_events[a], to_events[a]
    d0 = duration(published(P), a)
    k0 = (d0 - t0[t] + t0[f]) / mods[a]
    set_start_value(k[a], round(Int, k0))
end

# ---------------------------------------------------------------------
# 6. Optimization Execution
# ---------------------------------------------------------------------
optimize!(m)

if termination_status(m) in [OPTIMAL, TIME_LIMIT] && has_values(m)
    t_sol = round.(Int, value.(pi_))
    S = Solution(P, t_sol)
    println("\n=== Verification ===")
    println("Is Feasible: ", is_feasible(S))
    println("Final Score: ", score(S))
    println("Relative Gap: ", relative_gap(m))
else
    println("Optimization completed. No improved feasible solution found.")
end
