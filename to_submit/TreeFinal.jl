# =====================================================================
# MATH4023: Mathematical Optimisation - Swiss Taktfahrplan Solver
# Corrected High-Performance Decomposed Parallel Solver
# =====================================================================

#=
include("Takt.jl")
using .Takt
=#

using JuMP, HiGHS, Graphs

"""
    solve_decomposed_pesp(P)

Full automated pipeline to partition, optimize, stitch, and verify a PESP problem instance.
"""
function solve_decomposed_pesp(P)
    try
        ccall((:Highs_resetGlobalScheduler, HiGHS.libhighs), Cvoid, (Cint,), 0)
    catch e
    end

    nE = number_of_events(P)
    nA = number_of_activities(P)

    # Cache lookups separating tuples into single integers
    from_events = [events_of(P, a)[1] for a in 1:nA]  
    to_events   = [events_of(P, a)[2] for a in 1:nA]  
    mods        = [modulus(P, a) for a in 1:nA]
    lowers      = [lower(P, a) for a in 1:nA]
    uppers      = [upper(P, a) for a in 1:nA]
    weights     = [weight(P, a) for a in 1:nA]

    t0 = times(published(P))

    # -----------------------------------------------------------------
    # STEP 1: Graph Partitioning (With Rigid Node Separation)
    # -----------------------------------------------------------------
    g_working = SimpleGraph(nE)
    for a in 1:nA
        if from_events[a] != to_events[a]
            add_edge!(g_working, from_events[a], to_events[a])
        end
    end

    tree_edges = Tuple{Int, Int, Int}[]
    bridge_edges = Tuple{Int, Int, Int}[]
    cycle_components = Vector{Int}[]

    # Keep a strict tracking set of nodes that belong to tree lines
    all_tree_nodes = Set{Int}()

    # Iterative leaf peeling to capture trees
    peeling = true
    while peeling
        peeling = false
        for v in 1:nE
            if degree(g_working, v) == 1
                neighbors_v = neighbors(g_working, v)
                if !isempty(neighbors_v)
                    parent_node = neighbors_v[1]
                    act_id = findfirst(a -> (from_events[a] == v && to_events[a] == parent_node) || 
                                            (from_events[a] == parent_node && to_events[a] == v), 1:nA)
                    push!(tree_edges, (v, parent_node, act_id))
                    push!(all_tree_nodes, v) # Log this node explicitly as a tree asset
                    rem_edge!(g_working, v, parent_node)
                    peeling = true
                end
            end
        end
    end

    # Extract cycle components and bridges
    raw_blocks = biconnected_components(g_working)
    for block_edges in raw_blocks
        block_nodes = Set{Int}()
        for edge in block_edges
            push!(block_nodes, src(edge))
            push!(block_nodes, dst(edge))
        end
        
        # --- FIXED: STRICT EXCLUSION INTERFACE ---
        # Strip any node that was flagged as a tree asset from entering the cycle block
        clean_nodes = [n for n in block_nodes if !(n in all_tree_nodes)]
        
        if length(clean_nodes) >= 3
            push!(cycle_components, clean_nodes)
        elseif length(clean_nodes) == 2
            u, v = clean_nodes[1], clean_nodes[2]
            act_id = findfirst(a -> (from_events[a] == u && to_events[a] == v) || 
                                    (from_events[a] == v && to_events[a] == u), 1:nA)
            push!(bridge_edges, (u, v, act_id))
        end
    end

    println("=== Network Decomposition Summary ===")
    println("  True Core Cycle Components : ", length(cycle_components))
    println("  Acyclic Tree Node Branches : ", length(all_tree_nodes))
    println("  Connecting Bridges         : ", length(bridge_edges))
    println("=====================================\n")

    # Pre-calculate global bound limits to feed subproblems
    lo_k_global = [ceil(Int, (lowers[a] - Takt.period(P, to_events[a]) - 1) / mods[a]) for a in 1:nA]
    hi_k_global = [floor(Int, (uppers[a] + Takt.period(P, from_events[a]) + 1) / mods[a]) for a in 1:nA]

    # -----------------------------------------------------------------
    # STEP 2: Independent Sub-Component Optimization
    # -----------------------------------------------------------------
    t_local_opt = zeros(Int, nE)
    component_map = Dict{Int, Int}() 

    for (c_idx, comp) in enumerate(cycle_components)
        println("Optimizing Cycle Component $c_idx (Pruned Size: $(length(comp)) nodes)...")
        for node in comp; component_map[node] = c_idx; end
        
        sub_m = Model(HiGHS.Optimizer)
        set_optimizer_attribute(sub_m, "time_limit", 600.0) 
        set_optimizer_attribute(sub_m, "mip_abs_gap", 0.0)   
        set_optimizer_attribute(sub_m, "output_flag", true)   
        set_optimizer_attribute(sub_m, "presolve", "on")
        set_optimizer_attribute(sub_m, "parallel", "on") 
        set_optimizer_attribute(sub_m, "mip_heuristic_effort", 1.0)

        @variable(sub_m, 0 <= pi_sub[e in comp] <= Takt.period(P, e) - 1, Int)
        
        internal_acts = filter(a -> (from_events[a] in comp && to_events[a] in comp), 1:nA)
        @variable(sub_m, lo_k_global[a] <= k_sub[a in internal_acts] <= hi_k_global[a], Int)

        # Single-node anchor breaks gauge symmetry cleanly over the pruned set
        local_anchor = comp[1]
        fix(pi_sub[local_anchor], t0[local_anchor]; force=true)

        @constraint(sub_m, link_sub[a in internal_acts], 
            lowers[a] <= pi_sub[to_events[a]] - pi_sub[from_events[a]] + mods[a] * k_sub[a] <= uppers[a]
        )
        
        @objective(sub_m, Min, 
            sum(weights[a] * (pi_sub[to_events[a]] - pi_sub[from_events[a]] + mods[a] * k_sub[a] - lowers[a]) for a in internal_acts)
        )

        for e in comp; set_start_value(pi_sub[e], t0[e]); end
        for a in internal_acts
            d0 = duration(published(P), a)
            k0 = (d0 - t0[to_events[a]] + t0[from_events[a]]) / mods[a]
            set_start_value(k_sub[a], round(Int, k0))
        end

        optimize!(sub_m)
        
        if has_values(sub_m)
            for e in comp
                t_local_opt[e] = round(Int, value(pi_sub[e]))
            end
        else
            println("  System Notice: Subproblem fell back onto baseline vector.")
            for e in comp; t_local_opt[e] = t0[e]; end
        end
    end

    # Handle isolated nodes
    for e in 1:nE
        if !haskey(component_map, e) && !(e in all_tree_nodes)
            push!(cycle_components, [e])
            component_map[e] = length(cycle_components)
            t_local_opt[e] = t0[e]
        end
    end

    # -----------------------------------------------------------------
    # STEP 3: SAFE BRIDGE STITCHING COORDINATION
    # -----------------------------------------------------------------
    nC = length(cycle_components)
    Delta_vals = zeros(Int, nC) 

    if !isempty(bridge_edges)
        println("\nOptimizing global stitching offsets over network bridges...")
        stitch_m = Model(HiGHS.Optimizer)
        set_optimizer_attribute(stitch_m, "time_limit", 180.0)
        set_optimizer_attribute(stitch_m, "output_flag", true)
        set_optimizer_attribute(stitch_m, "mip_abs_gap", 0.0)     
        set_optimizer_attribute(stitch_m, "presolve", "on")
        set_optimizer_attribute(stitch_m, "parallel", "on") 
        set_optimizer_attribute(stitch_m, "mip_heuristic_effort", 1.0)

        @variable(stitch_m, 0 <= Delta[1:nC] <= 119, Int) 
        fix(Delta[1], 0; force=true) 

        bridge_act_ids = [b[3] for b in bridge_edges]
        @variable(stitch_m, -10 <= k_bridge[a in bridge_act_ids] <= 10, Int)

        for b in bridge_edges
            u, v, a = b
            c_u = component_map[u]
            c_v = component_map[v]
            
            @constraint(stitch_m, uppers[a] >= (t_local_opt[v] + Delta[c_v]) - (t_local_opt[u] + Delta[c_u]) + mods[a] * k_bridge[a])
            @constraint(stitch_m, lowers[a] <= (t_local_opt[v] + Delta[c_v]) - (t_local_opt[u] + Delta[c_u]) + mods[a] * k_bridge[a])
        end

        @objective(stitch_m, Min, 
            sum(weights[a] * ((t_local_opt[to_events[a]] + Delta[component_map[to_events[a]]]) - 
                              (t_local_opt[from_events[a]] + Delta[component_map[from_events[a]]]) + 
                              mods[a] * k_bridge[a] - lowers[a]) for a in bridge_act_ids)
        )

        optimize!(stitch_m)
        if has_values(stitch_m)
            Delta_vals = round.(Int, value.(Delta))
        end
    else
        println("\nTopology Notice: Zero bridge cut-edges detected. Skipping stitching model...")
    end

    # Compile core component times
    t_final = fill(-1, nE)
    for e in 1:nE
        if haskey(component_map, e)
            c_id = component_map[e]
            t_final[e] = mod(t_local_opt[e] + Delta_vals[c_id], Takt.period(P, e))
        end
    end

    # -----------------------------------------------------------------
    # STEP 4: Recursive Linear Tree Re-Attachment
    # -----------------------------------------------------------------
    println("\nPropagating times to acyclic tree branches...")
    tree_resolved = true
    while tree_resolved
        tree_resolved = false
        for (child, parent, a) in tree_edges
            if t_final[parent] != -1 && t_final[child] == -1
                l_a = lowers[a]
                T_c = Takt.period(P, child)
                if from_events[a] == parent && to_events[a] == child
                    t_final[child] = mod(t_final[parent] + l_a, T_c)
                else
                    t_final[child] = mod(t_final[parent] - l_a, T_c)
                end
                tree_resolved = true
            end
        end
    end

    for e in 1:nE
        if t_final[e] == -1; t_final[e] = t0[e]; end
    end

    # -----------------------------------------------------------------
    # STEP 5: Verification & Execution
    # -----------------------------------------------------------------
    S = Solution(P, t_final)
    println("\n=== Verification ===")
    println("Decomposed Script Feasible: ", is_feasible(S))
    println("Stitched Network Final Score: ", score(S))
    
    return S
end

for i in 3:7
    P = Problem(i)
    s = solve_decomposed_pesp(P)
    #savesolution(s)
end
