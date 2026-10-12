include(joinpath(@__DIR__, "Takt", "src", "Takt.jl"))
using .Takt
using Random

# Score of a list of event times 
function total_score(P, t)
    return score(Solution(P, t))
end

# Return a copy of t with every event of one line moved c minutes later
function shift_line(P, t, events, c)
    new_t = copy(t)
    for e in events
        new_t[e] = mod(t[e] + c, period(P, e))
    end
    return new_t
end

# gets all possible moves along a line 
function all_moves(P, line_events)
    moves = []
    for events in line_events
        T = period(P, events[1])
        push!(moves, (events, 1:T-1)) # all shifts of c minutes (up to period)
        for i in 2:length(events) # wiggle the actual event durations  
            for a in 1:number_of_activities(P)
                if events_of(P, a) == (events[i-1], events[i])
                    span = upper(P, a) - lower(P, a)
                    if span > 0
                        push!(moves, (events[i:end], -span:span))
                    end
                end
            end
        end
    end
    return moves
end 

# Try every shift of every line, keeping any that lowers the score.
# Repeat until a full pass finds nothing better.
function improve(P, t, moves)
    best = total_score(P, t)
    improved = true
    while improved
        improved = false
        for (events, shifts) in moves
            for c in shifts
                if c == 0
                    continue
                end
                candidate = shift_line(P, t, events, c)
                s = total_score(P, candidate)
                if s < best
                    t = candidate
                    best = s
                    improved = true
                end
            end
        end
    end
    return t, best
end

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
n       = 12        # problem number
seconds = 60      # how long to search (seconds)

P = Problem(n)

# The events belonging to each line
line_events = [[e for e in 1:number_of_events(P) if linename(P, e) == l] for l in lines(P)]
moves = all_moves(P, line_events)

start_t = times(published(P))
println("Problem $n, published score: ", round(Int, total_score(P, start_t)))

# Step 1: improve the published timetable until no single line shift helps.
best_t, best = improve(P, start_t, moves)
println("After first pass: ", round(Int, best))

# Step 2: move a random line by random amount, improve again, only keep the result if its better
stop_time = time() + seconds
while time() < stop_time
    events = rand(line_events)
    c = rand(1:period(P, events[1]) - 1)
    t = shift_line(P, best_t, events, c)
    if total_score(P, t) == Inf
        continue                      
    end
    t, s = improve(P, t, moves)
    if s < best
        global best_t = t
        global best = s
        println("New best: ", round(Int, best))
    end
end

S = Solution(P, best_t)
println("Feasible: ", is_feasible(S), "   final score: ", round(Int, score(S)))
println("Saved to ", savesolution(S))
