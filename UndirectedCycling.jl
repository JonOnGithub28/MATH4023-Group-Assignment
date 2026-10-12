# =====================================================================
# MATH4023: Mathematical Optimisation - Swiss Taktfahrplan Solver
# Undirected Cycle-Basis Bound Tightening Formulation (PESP Classic)
# =====================================================================

#=
include("Takt.jl")
using .Takt
=#

using JuMP, HiGHS, Graphs

"""
    solve_pesp_with_cycle_bounds(P)

Solves the PESP instance using undirected cycle basis cuts to bound integer variables k.
"""
function solve_pesp_with_cycle_bounds(P)
    try
        ccall((:Highs_resetGlobalScheduler, HiGHS.libhighs), Cvoid, (Cint,), 0)
    catch e
    end

    nE = number_of_events(P)
    nA = number_of_activities(P)

    # --- FIXED CACHING UNPACKING TUPLES INTO SINGLE INTEGERS ---
    from_events = [events_of(P, a)[1] for a in 1:nA]  
    to_events   = [events_of(P, a)[2] for a in 1:nA]  
    mods        = [modulus(P, a) for a in 1:nA]
    lowers      = [lower(P, a) for a in 1:nA]
    uppers      = [upper(P, a) for a in 1:nA]
    weights     = [weight(P, a) for a in 1:nA]

    t0 = times(published(P))

    # -----------------------------------------------------------------
    # STEP 1: Construct Graph Topology & Map Parallel Edges Securely
    # -----------------------------------------------------------------
    g_undirected = SimpleGraph(nE)
    
    # Map from (u, v) vertex pairs to ALL activity IDs connecting them.
    edge_to_activities = Dict{Tuple{Int,Int}, Vector{Int}}()

    for a in 1:nA
        f, t = from_events[a], to_events[a]
        if f != t
            add_edge!(g_undirected, f, t)
            u, v = min(f, t), max(f, t)
            # FIXED: Corrected default value from 'Int' type to a real empty array 'Int[]'
            push!(get!(edge_to_activities, (u, v), Int[]), a)
        end
    end

    # -----------------------------------------------------------------
    # STEP 2: Extract Integral Undirected Cycle Basis
    # -----------------------------------------------------------------
    cycles = cycle_basis(g_undirected)
    println("Graph Engine: Found $(length(cycles)) fundamental undirected cycles.")

    # -----------------------------------------------------------------
    # STEP 3: Model Setup & Baseline Multi-Periodic Bounds
    # -----------------------------------------------------------------
    m = Model(HiGHS.Optimizer)
    set_optimizer_attribute(m, "time_limit", 600.0) 
    set_optimizer_attribute(m, "mip_rel_gap", 1e-4)   
    set_optimizer_attribute(m, "output_flag", true)   
    set_optimizer_attribute(m, "presolve", "on")
    set_optimizer_attribute(m, "parallel", "on")
    set_optimizer_attribute(m, "mip_heuristic_effort", 0.5)

    # Initial element-by-element rigorous loose bounds
    lo_k = [ceil(Int, (lowers[a] - Takt.period(P, to_events[a]) - 1) / mods[a]) for a in 1:nA]
    hi_k = [floor(Int, (uppers[a] + Takt.period(P, from_events[a]) + 1) / mods[a]) for a in 1:nA]

    @variable(m, 0 <= pi_[e=1:nE] <= Takt.period(P, e) - 1, Int)   
    @variable(m, lo_k[a] <= k[a=1:nA] <= hi_k[a], Int)

    # Break translation gauge symmetry on the first global node reference
    fix(pi_[1], t0[1]; force=true) 

    # Core PESP Edge Constraints
    @constraint(m, link[a=1:nA], 
        lowers[a] <= pi_[to_events[a]] - pi_[from_events[a]] + mods[a] * k[a] <= uppers[a]
    )

    # -----------------------------------------------------------------
    # STEP 4: Inject Undirected Cycle Bounding Inequalities
    # -----------------------------------------------------------------
    println("Injecting cycle-basis integer cuts...")
    cuts_added = 0

    for cycle in cycles
        # Keep cycle constraints reasonably tight (<= 6 nodes) to prevent dense matrix rows
        if length(cycle) <= 6
            cycle_nodes = [cycle; cycle] # Explicitly close the circuit loop path
            
            # Construct cycle-wide bounding parameters
            int_combination = AffExpr(0.0)
            sum_lower = 0.0
            sum_upper = 0.0
            valid_cycle = true
            
            for i in 1:(length(cycle_nodes)-1)
                u_node = cycle_nodes[i]
                v_node = cycle_nodes[i+1]
                
                u_key, v_key = min(u_node, v_node), max(u_node, v_node)
                
                if haskey(edge_to_activities, (u_key, v_key))
                    # FIXED: Unpack the primary structural activity index out of the vector array safely
                    activities = edge_to_activities[(u_key, v_key)]
                    a = activities[1] 
                    
                    # Track direction of the cycle traversal relative to directed track data
                    if from_events[a] == u_node && to_events[a] == v_node
                        # Forward Edge traversal
                        add_to_expression!(int_combination, Float64(mods[a]), k[a])
                        sum_lower += lowers[a]
                        sum_upper += uppers[a]
                    else
                        # Backward Edge traversal: flips sign and lower/upper bounds
                        add_to_expression!(int_combination, -Float64(mods[a]), k[a])
                        sum_lower -= uppers[a]
                        sum_upper -= lowers[a]
                    end
                else
                    valid_cycle = false
                    break
                end
            end
            
            if valid_cycle
                @constraint(m, sum_lower <= int_combination <= sum_upper)
                cuts_added += 1
            end
        end
    end
    println("Successfully bound $cuts_added undirected fundamental cycle structures into the matrix.")

    # -----------------------------------------------------------------
    # STEP 5: Objective Function & Timetable Initialisation
    # -----------------------------------------------------------------
    @objective(m, Min, 
        sum(weights[a] * (pi_[to_events[a]] - pi_[from_events[a]] + mods[a] * k[a] - lowers[a]) for a in 1:nA)
    )

    # Seed warm start directly from the known-feasible published schedule
    set_start_value.(pi_, t0)
    for a in 1:nA
        d0 = duration(published(P), a)
        k0 = (d0 - t0[to_events[a]] + t0[from_events[a]]) / mods[a]
        set_start_value(k[a], round(Int, k0))
    end

    # -----------------------------------------------------------------
    # STEP 6: Optimization Execution
    # -----------------------------------------------------------------
    optimize!(m)

    if termination_status(m) in [OPTIMAL, TIME_LIMIT] && has_values(m)
        t_sol = round.(Int, value.(pi_))
        S = Solution(P, t_sol)
        println("\n=== Verification ===")
        println("Cycle-Bound Script Feasible: ", is_feasible(S))
        println("Final System Score:         ", score(S))
        println("Relative Gap:               ", relative_gap(m))
        return S
    else
        println("Optimization failed to extract improved values.")
        return solution(published(P))
    end
end


P = Problem(2)
s = solve_pesp_with_cycle_bounds(P)