using Graphs, JuMP, HiGHS
using Takt

# Builds the event-acitivies graph of a Takt problem. Assumes the problem is connected, which they all are.
function build_problem_graph(P::Problem)
    g = SimpleDiGraph()
    # todo prune based on which one(s) we actually use
    activity_to_edge = Dict{Int64,Tuple{Int64,Int64}}()
    edge_to_activity = Dict{Tuple{Int64,Int64},Int64}()
    add_vertices!(g, number_of_events(P))
    for a in 1:number_of_activities(P)
        # For vertices which have edges between them in both directions, we
        # want to arbitrarily choose one to keep (first come first serve)
        #
        # Upon reflection, does this ever happen? The structure of the evene-actities graph
        # should mean that one can only move from "incoming" events to "outgoing", so this
        # should never happen. Regardless, no need to change this.
        f,t = events_of(P,a)    
        if !has_edge(g,t,f)
            add_edge!(g,f,t)
            activity_to_edge[a] = (f,t)
            edge_to_activity[(f,t)] = a
        end
    end
    return g, activity_to_edge, edge_to_activity
end

# Graphs.jl does not support producing a directed fundamental cycle basis,
# so we need to construct such a thing out of an undirected one.
function cycle_basis_directed(g::SimpleDiGraph)
    B = cycle_basis(SimpleGraph(g))
    B_directed = []
    for i in eachindex(B)
        b = B[i]
        # Indices here are ((vertex,vertex), orientation). Some redundancy but convenient
        b_directed = Dict{Tuple{Int64,Int64},Int64}()
        for i in 1:(length(b)-1)
            e_0 = b[i]
            e_1 = b[i+1]
            if has_edge(g,e_0,e_1)
                b_directed[(e_0,e_1)] = 1
            else
                @assert has_edge(g,e_1,e_0)  "bug in cycle_basis_directed()"
                b_directed[(e_0,e_1)] = -1
            end
        end

        # Need to add last edge manually (ask me how I know...)
        e_0 = b[end]
        e_1 = b[1]
        if has_edge(g,e_0,e_1)
            b_directed[(e_0,e_1)] = 1
        else
            @assert has_edge(g,e_1,e_0)  "bug in cycle_basis_directed()"
            b_directed[(e_0,e_1)] = -1
        end

        push!(B_directed, b_directed)
    end
    return B_directed
end

# Compute the potentials (i.e., the times/pis) from the tensions
function potential_from_tensions(P, g, x)
    vertices_to_search = Set{Int64}(1)
    pi_dict = Dict{Int64, Int64}(1 => 0)
    while !isempty(vertices_to_search)
        e = pop!(vertices_to_search)
        ins = inneighbors(g,e)
        for f in ins
            if !haskey(pi_dict, f)
                pi_dict[f] = pi_dict[e] - x[(f,e)]
                push!(vertices_to_search, f)
            end
        end
        outs = outneighbors(g,e)
        for t in outs
            if !haskey(pi_dict, t)
                pi_dict[t] = pi_dict[e] + x[(e,t)]
                push!(vertices_to_search, t)
            end
        end

        #= DEBUG v
        println(pi_dict)
        for a in 1:number_of_activities(p)
            f,t = events_of(p,a)
            if haskey(pi_dict, f) && haskey(pi_dict, t)
                if mod(pi_dict[t] - pi_dict[f] - x_vals[(f,t)], T) != 0  
                    println("x not recovered correctly from pi during conversion")
                    println("offending activity (index $a, from index $f, to index $t):")
                    println(activity(p,a))
                    println("pi difference: $(pi_dict[t] - pi_dict[f])")
                    println("x_value: $(x_vals[(f,t)])")
                    error()
                end
            end
        end
        =#
    end

    # Need to reduce potential modulo the period
    n_keys = length(keys(pi_dict))
    @assert n_keys == number_of_events(p) "did not capture all events in potential conversion"
    pi_vect = [mod(pi_dict[e], Takt.period(p,e)) for e in 1:n_keys]

    # DEBUG v
    for a in 1:number_of_activities(p)
        f,t = events_of(p,a)
        if haskey(pi_dict, f) && haskey(pi_dict, t)
            if mod(pi_dict[t] - pi_dict[f] - x_vals[(f,t)], T) != 0  
                println("x not recovered correctly from pi during conversion, after modular reduction")
                println("offending activity (index $a, from index $f, to index $t):")
                println(activity(p,a))
                println("pi difference: $(pi_dict[t] - pi_dict[f])")
                println("x_value: $(x_vals[(f,t)])")
                error()
            end
        end
    end
    #

    return pi_vect
end

function incidence_matrix(B, edges)
    ne = length(edges)
    Gamma = Array{Int64}(undef,0,ne)
    for b in B
        r = Vector{Int64}()
        for (f,t) in edges
            if haskey(b,(f,t))
                push!(r,1)
            elseif haskey(b,(t,f))
                push!(r,-1)
            else
                push!(r,0)
            end
        end
        Gamma = [Gamma; r']
    end
    return Gamma
end

function min_period(p)
    n = number_of_events(p)
    return minimum(e -> Takt.period(p,e), 1:n)
end

pn = 7
p = Problem(pn)
pg, a2e, e2a = build_problem_graph(p)
pg_edges = [(src(ed),dst(ed)) for ed in edges(pg)]
B = cycle_basis_directed(pg)
Gamma = incidence_matrix(B, pg_edges)
T = min_period(p)

m = Model(HiGHS.Optimizer)
@variable(m, lower(p,e2a[e]) <= x[e in pg_edges] <= upper(p,e2a[e]), Int)
@variable(m, z[b in eachindex(B)], Int)
@constraint(m, Gamma * x - T * z == 0)
@objective(m, Min, sum(weight(p,e2a[e]) * (x[e] - lower(p,e2a[e])) for e in pg_edges))

set_optimizer_attribute(m, "time_limit", 600.0)
set_optimizer_attribute(m, "mip_rel_gap", 0.0)
set_optimizer_attribute(m, "presolve", "on")
set_optimizer_attribute(m, "parallel", "on")
set_optimizer_attribute(m, "threads", Sys.CPU_THREADS)

# If time allowed I would add a warm-start option that converted the published
# solution into x-z form.

optimize!(m)

x_vals = Dict(e => round(Int, value(x[e])) for e in pg_edges)
xv = collect(values(x_vals))
pi_ = potential_from_tensions(p, pg, x_vals)

# DEBUG
for a in 1:number_of_activities(p)
    f,t = events_of(p,a)
    if mod(pi_[t] - pi_[f] - x_vals[(f,t)], T) != 0  
        println("x not recovered correctly from pi after conversion")
        println("offending activity (index $a, from index $f, to index $t):")
        println(activity(p,a))
        println("pi difference: $(pi_[t] - pi_[f])")
        println("x_value: $(x_vals[(f,t)])")
        error()
    end
end

for a in 1:number_of_activities(p)
    f,t = events_of(p,a)
    if mod(pi_[t] - pi_[f] - lower(p,a), T) + lower(p,a) > upper(p,a) 
        println("constraint violation after conversion")
        println("offending activity (index $a, from index $f, to index $t):")
        println("upper: $(upper(p,a))")
        println("lower: $(lower(p,a))")
        println("pi difference: $(pi_[t] - pi_[f])")
        println("modulo: $(mod(pi_[t] - pi_[f] - lower(p,a), T))")
        error()
    end
end

#println(x_vals)
#println(pi_)
#println([activity(p,i) for i in 1:number_of_activities(p)])

S = Solution(p,pi_)
@show is_feasible(S)
@show score(S)
@show gap(S, objective_bound(m))
savesolution(S, "relaxed_p$(pn)_cyclebasis.json")