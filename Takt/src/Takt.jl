"""
    Takt

Twelve periodic timetabling problems on the Swiss railway network.

Each problem is a set of **events** — a train on one line arriving at, or
departing from, one station — joined by **activities** which constrain the gap
between two events.  Choosing a time for every event is choosing a timetable.

Lines run every 15, 30, 60 or 120 minutes and the pattern repeats, so all
arithmetic is modular; an activity joining two lines has modulus `gcd` of their
two periods.

See the assignment for what the problems mean.  This module provides:

**Loading**

| | |
|:--|:--|
| [`number_of_problems`](@ref) | how many problems there are |
| [`problem_numbers`](@ref) | their numbers, `0:12` (problem 0 is an unscored warm-up) |
| [`Problem`](@ref) | load one |
| [`published`](@ref) | the timetable Switzerland actually runs |

**Asking about a problem**

| | |
|:--|:--|
| [`number_of_events`](@ref), [`number_of_activities`](@ref) | sizes |
| [`lines`](@ref), [`period`](@ref) | which lines, and how often each runs |
| [`event`](@ref) | what an event is, in one call |
| [`linename`](@ref), [`stationname`](@ref), [`eventkind`](@ref) | its line, station, arrival-or-departure |
| [`events_of`](@ref) | which two events an activity joins |
| [`activities_at`](@ref) | which activities touch an event |
| [`stations`](@ref) | the stations in the problem |
| [`activity`](@ref) | what an activity is, in one call |
| [`describe`](@ref) | either of those, in words |
| [`lower`](@ref), [`upper`](@ref), [`modulus`](@ref) | an activity's bounds and modulus |
| [`weight`](@ref), [`activitytype`](@ref) | its weight and kind |

**Making and judging a timetable**

| | |
|:--|:--|
| [`Solution`](@ref) | build one from a vector of times |
| [`times`](@ref) | get the times back out |
| [`duration`](@ref), [`slack`](@ref) | what an activity costs in it |
| [`is_feasible`](@ref), [`violations`](@ref) | is it admissible, and if not why |
| [`score`](@ref) | weighted avoidable delay — the thing to minimise |
| [`timetable`](@ref) | print one line of it, in running order |
| [`scoreby`](@ref) | where the score is going, by activity type |
| [`costliest`](@ref) | the activities costing you the most |
| [`gap`](@ref) | how far above a lower bound you are |
| [`savesolution`](@ref), [`loadsolution`](@ref) | write and read a submission |

Every function has its own documentation: try `?score`.
"""
module Takt

using JSON, Printf

export Problem, Solution, number_of_problems, problem_numbers, number_of_events, number_of_activities,
       period, modulus, lower, upper, weight, activitytype, duration, slack,
       score, is_feasible, violations, timetable, savesolution, loadsolution,
       times, activity, describe, scoreby, costliest, gap, stations, activities_at,
       published,
       lines, stationname, linename, eventkind, event, events_of

struct Activity
    from::Int; to::Int; lower::Int; upper::Int; modulus::Int
    weight::Float64; type::Symbol
end
struct Event
    line::String; station::String; kind::Symbol
end
struct Problem
    n::Int
    events::Vector{Event}
    activities::Vector{Activity}
    periods::Dict{String,Int}
    published::Vector{Int}
end
struct Solution
    problem::Problem
    times::Vector{Int}
    # Inner constructor, so the checks cannot be bypassed: an outer method is
    # skipped whenever the argument is already a Vector{Int}.
    function Solution(P::Problem, t::AbstractVector{<:Integer})
        length(t) == length(P.events) ||
            error("Problem $(P.n) has $(length(P.events)) events, got $(length(t)) times")
        for (e, x) in enumerate(t)
            T = P.periods[P.events[e].line]
            0 <= x < T || error(
                "time $(x) for event $(e) is outside 0..$(T-1); that event is on " *
                "line $(P.events[e].line), which runs every $T minutes. " *
                "Reduce your times modulo the period of their line.")
        end
        new(P, collect(Int, t))
    end
end

const _DATA = Ref{Any}(nothing)
_data() = (_DATA[] === nothing && (_DATA[] = JSON.parsefile(joinpath(@__DIR__, "takt.json"))); _DATA[])

"""
    Problem(n) -> Problem

Load problem `n`.  The problems are numbered **0 to 12** --- see
[`problem_numbers`](@ref) --- and are **nested**: every activity of problem `n`
is also an activity of problem `n+1`.

Problem 0 is a single line with no transfers, so nothing trades off against
anything: its optimum is exactly zero.  It is a warm-up, it **does not count for
the competition**, and you should run it first to check a formulation before
trying anything larger.

```julia
julia> P = Problem(5)
Problem 5: 15 lines, periods 30, 60, 120 min
  348 events, 443 activities
    drive       174
    dwell       159
    ...
```

See also [`number_of_problems`](@ref), [`published`](@ref), [`score`](@ref).
"""
function Problem(n::Integer)
    d = _data(); hi = length(d["problems"]) - 1
    0 <= n <= hi || error("problems are numbered 0 to $(hi); got $(n)")
    p = d["problems"][n + 1]          # problem 0 is the warm-up, stored first
    ev = [Event(e["line"], e["station"], Symbol(e["kind"])) for e in p["events"]]
    ac = [Activity(a["from"], a["to"], a["lower"], a["upper"], a["modulus"],
                   a["weight"], Symbol(a["type"])) for a in p["activities"]]
    Problem(n, ev, ac, Dict{String,Int}(k => v for (k,v) in p["periods"]),
            Int.(p["real"]))
end
"""
    number_of_problems() -> Int

How many problems this package contains, **counting the warm-up**.  They are
numbered `0` to `number_of_problems() - 1`; iterate with
[`problem_numbers`](@ref) rather than `1:number_of_problems()`.

See also [`Problem`](@ref), [`problem_numbers`](@ref).
"""
number_of_problems() = length(_data()["problems"])
"""
    problem_numbers() -> UnitRange{Int}

The problem numbers, `0:12`.  Problem `0` is an unscored warm-up whose optimum is
zero; problems `1` upward are the competition.

```julia
for n in problem_numbers()          # all of them, warm-up included
    P = Problem(n)
end
```

See also [`Problem`](@ref), [`number_of_problems`](@ref).
"""
problem_numbers() = 0:(length(_data()["problems"]) - 1)
"""
    number_of_events(P) -> Int

How many events problem `P` has.  A `Solution` needs exactly this many times.

See also [`Solution`](@ref), [`number_of_activities`](@ref).
"""
number_of_events(P::Problem) = length(P.events)
"""
    number_of_activities(P) -> Int

How many activities problem `P` has.  Activities are indexed `1:number_of_activities(P)`.

See also [`activitytype`](@ref), [`events_of`](@ref).
"""
number_of_activities(P::Problem) = length(P.activities)
"""
    lines(P) -> Vector{String}

The names of the lines in problem `P`, sorted.  A name looks like `"IC1:0"`: the
service, then `:0` or `:1` for the two directions, which are separate lines here
and joined only by their turnarounds.

```julia
julia> lines(Problem(2))
8-element Vector{String}:
 "IC1:0"
 "IC1:1"
 ...
```

See also [`period`](@ref), [`timetable`](@ref).
"""
lines(P::Problem) = sort(collect(keys(P.periods)))
"""
    linename(P, e) -> String

Which line the train in event `e` is running on.

See also [`stationname`](@ref), [`period`](@ref).
"""
linename(P::Problem, e::Integer) = P.events[e].line
"""
    stationname(P, e) -> String

Which station event `e` happens at.

See also [`linename`](@ref).
"""
stationname(P::Problem, e::Integer) = P.events[e].station

"""
    eventkind(P, e) -> Symbol

Whether event `e` is an `:arrival` or a `:departure`.

Which kind an event is determines what it can be joined to: a `:drive` runs from
a departure to an arrival, a `:dwell` from an arrival to a departure, a
`:transfer` from one train's arrival to another's departure.

```julia
julia> eventkind(Problem(3), 1)
:departure
```

See also [`event`](@ref), [`activitytype`](@ref).
"""
eventkind(P::Problem, e::Integer) = P.events[e].kind

"""
    event(P, e) -> NamedTuple

Everything about event `e` at once: its `line`, its `station`, and its `kind`.

```julia
julia> event(Problem(3), 4)
(line = "IC1:0", station = "Bern", kind = :arrival)

julia> event(Problem(3), 4).kind
:arrival
```

Equivalent to `(line = linename(P,e), station = stationname(P,e), kind = eventkind(P,e))`,
but rather easier to read when you are inspecting a timetable.

See also [`linename`](@ref), [`stationname`](@ref), [`eventkind`](@ref), [`events_of`](@ref).
"""
event(P::Problem, e::Integer) =
    (line = P.events[e].line, station = P.events[e].station, kind = P.events[e].kind)

"""
    period(P, line) -> Int
    period(P, e)    -> Int

How often a line runs, in minutes — 15, 30, 60 or 120.  Given an integer, the
period of the line that event belongs to.

A time for event `e` must lie in `0:period(P,e)-1`.

```julia
julia> period(Problem(5), "IC1:0")
60
```

See also [`lines`](@ref), [`modulus`](@ref).
"""
period(P::Problem, line::AbstractString) = P.periods[line]
period(P::Problem, e::Integer) = P.periods[P.events[e].line]

"""
    lower(P, a) -> Int

The shortest time activity `a` may take, in minutes.  For a drive or a dwell this
is the shortest SBB themselves schedule for it; for a transfer it is the walking
time; for a headway the minimum separation.

See also [`upper`](@ref), [`duration`](@ref), [`slack`](@ref).
"""
lower(P::Problem, a::Integer) = P.activities[a].lower
"""
    upper(P, a) -> Int

The longest time activity `a` may take.  A timetable is admissible when every
activity's duration is at most its upper bound.

Transfers and turnarounds have `upper == lower + modulus - 1`, so they span a
whole period and can never be violated — they cost, but they do not constrain.
Drives, dwells and headways are what make a timetable infeasible.

See also [`lower`](@ref), [`is_feasible`](@ref).
"""
upper(P::Problem, a::Integer) = P.activities[a].upper
"""
    modulus(P, a) -> Int

The modulus to which activity `a`'s duration is determined: the `gcd` of the
periods of the lines of the two events it joins.

Event `i` happens at every time `pi[i] + T_i * k`, and event `j` at every
`pi[j] + T_j * m`.  The gaps between them are therefore

    pi[j] - pi[i] + T_j * m - T_i * k

which, as `k` and `m` range over the integers, takes exactly the values
congruent to `pi[j] - pi[i]` modulo `gcd(T_i, T_j)`.  So the duration is pinned
down only to within that gcd — not to within either line's period.

This is **not** the interval at which the activity repeats.  A 30-minute line
feeding a 60-minute one presents the same configuration only every 60 minutes,
the *lcm*; but the gaps it offers are spaced 30 apart, the *gcd*, and it is the
spacing that the arithmetic cares about.

For a drive or a dwell both events lie on one line, so the modulus is that
line's period.

```julia
julia> P = Problem(7); i, j = events_of(P, 1);
julia> modulus(P, 1) == gcd(period(P, i), period(P, j))
true
```

See also [`period`](@ref), [`duration`](@ref), [`eventkind`](@ref).
"""
modulus(P::Problem, a::Integer) = P.activities[a].modulus
"""
    weight(P, a) -> Float64

Passengers per hour affected by activity `a`: riding it, if a drive or dwell;
changing trains, if a transfer.

Turnarounds and headways have weight zero — nobody is riding the train round the
corner, and nobody is riding the gap between two trains.  They constrain the
timetable without costing anything.

See also [`score`](@ref), [`activitytype`](@ref).
"""
weight(P::Problem, a::Integer) = P.activities[a].weight
"""
    activitytype(P, a) -> Symbol

What kind of activity `a` is:

| | from | to |
|:--|:--|:--|
| `:drive` | a train departing | the same train arriving at the next station |
| `:dwell` | a train arriving | the same train departing |
| `:turnaround` | a train arriving at a terminus | its departure back the other way |
| `:transfer` | a train on one line arriving | a train on another departing, for changing |
| `:headway` | a train departing | a train on another line departing over the same track |

See also [`weight`](@ref), [`upper`](@ref).
"""
activitytype(P::Problem, a::Integer) = P.activities[a].type
"""
    events_of(P, a) -> (Int, Int)

The two events activity `a` joins, as `(from, to)`.  Its duration is measured
from the first to the second.

See also [`duration`](@ref), [`linename`](@ref), [`stationname`](@ref).
"""
events_of(P::Problem, a::Integer) = (P.activities[a].from, P.activities[a].to)

"""
    Solution(P, times) -> Solution

A timetable for problem `P`.  `times[e]` is the minute at which event `e`
happens, and must lie in `0:period(P,e)-1`; times outside that range are an
error, so reduce them modulo the period of their line first.  `length(times)`
must equal `number_of_events(P)`.

```julia
julia> S = Solution(P, zeros(Int, number_of_events(P)))
NOT an admissible timetable for Problem 1: 54 activities violated.
```

Trains cannot teleport, so the all-zero timetable is not admissible.  A better
first attempt is to let every train run as fast as it is allowed and propagate
along each line.

See also [`published`](@ref), [`is_feasible`](@ref), [`score`](@ref).
"""
Solution

"""
    times(S)    -> Vector{Int}
    times(S, e) -> Int

The event times in timetable `S`: `times(S)[e]` is the minute at which event `e`
happens.  This is the inverse of the `Solution` constructor —

    times(Solution(P, t)) == t

— and it returns a copy, so modifying it cannot corrupt `S`.

Its main use is handing a known timetable to a solver as a starting point.  You
are given one, so use it:

```julia
P  = Problem(7)
t0 = times(published(P))                  # a feasible timetable, free
@variable(model, 0 <= pi[e=1:number_of_events(P)] <= period(P, e) - 1, Int)
for e in 1:number_of_events(P)
    set_start_value(pi[e], t0[e])
end
```

and afterwards, to read a solution back out and check it:

```julia
S = Solution(P, round.(Int, value.(pi)))
is_feasible(S) || error("the solver returned something inadmissible")
```

See also [`Solution`](@ref), [`published`](@ref), [`savesolution`](@ref).
"""
times(S::Solution) = copy(S.times)
times(S::Solution, e::Integer) = S.times[e]

"""
    published(P) -> Solution

The timetable Switzerland actually runs, reconstructed from the same feed as
everything else in this package.

It is data, not an answer.  It is admissible on every problem, every group has
it, and its score is a long way from zero — so beating it is the exercise, and it
makes a good incumbent to hand a solver as a starting point.

```julia
julia> S = published(Problem(1));
julia> is_feasible(S), Int(score(S))
(true, 222246)
```

See also [`Solution`](@ref), [`score`](@ref), [`timetable`](@ref).
"""
published(P::Problem) = Solution(P, P.published)

"""
    duration(S, a) -> Int

How long activity `a` takes under timetable `S`, in minutes:

    lower(P,a) + mod(times[to] - times[from] - lower(P,a), modulus(P,a))

that is, the smallest value at least `lower(P,a)` consistent with the two event
times, given that the gaps between them are spaced `modulus(P,a)` apart.

For a transfer this is the wait faced by the **luckiest** train in the cycle; the
mean over all trains is higher, and [`score`](@ref) adds the difference.

See also [`slack`](@ref), [`modulus`](@ref), [`score`](@ref).
"""
function duration(S::Solution, a::Integer)
    A = S.problem.activities[a]
    A.lower + mod(S.times[A.to] - S.times[A.from] - A.lower, A.modulus)
end
"""
    slack(S, a) -> Int

`duration(S, a) - lower(P, a)`: how much longer activity `a` takes than it must.
Always in `0:modulus(P,a)-1`.

Summed with weights, this *is* the score — the whole of what a timetable can
influence, and the objective you hand a solver.

See also [`duration`](@ref), [`score`](@ref).
"""
slack(S::Solution, a::Integer) = duration(S, a) - S.problem.activities[a].lower

"""
    violations(S) -> Vector{Int}

The activities whose duration exceeds their upper bound under `S`, as indices.
Empty exactly when the timetable is admissible.

Only drives, dwells and headways can appear: transfers and turnarounds span a
whole period, so they cost but never constrain.

See also [`is_feasible`](@ref), [`upper`](@ref).
"""
function violations(S::Solution)
    P = S.problem
    [a for a in 1:number_of_activities(P) if duration(S, a) > P.activities[a].upper]
end
"""
    is_feasible(S) -> Bool

Whether every activity is satisfied.  **Check this before submitting**: a
timetable that is not admissible scores nothing.

See also [`violations`](@ref), [`score`](@ref).
"""
is_feasible(S::Solution) = isempty(violations(S))

"""
    score(S) -> Float64

**Weighted avoidable delay** under timetable `S`, in passenger-minutes, and the
quantity to minimise:

    score(S) = sum over activities of  weight(a) * slack(S, a)

`slack(S,a)` is how much longer activity `a` takes than it has to, so this is
the waiting every passenger suffers *on top of* what the best possible timetable
would give them.  `score(S) >= 0`, and 0 is attainable only when nothing trades off against
anything — on Problem 1, for instance.

Returns `Inf` if `S` is not admissible.

```julia
julia> score(published(Problem(1))), score(published(Problem(7)))
(222246.0, 6255010.0)
```

See also [`slack`](@ref), [`gap`](@ref).
"""
function score(S::Solution)
    is_feasible(S) || return Inf
    sum(S.problem.activities[a].weight * slack(S, a)
        for a in 1:number_of_activities(S.problem); init = 0.0)
end

function Base.show(io::IO, P::Problem)
    c = Dict(t => count(a -> a.type === t, P.activities)
             for t in (:drive,:dwell,:turnaround,:transfer,:headway))
    per = join(sort(unique(values(P.periods))), ", ")
    println(io, "Problem $(P.n): $(length(P.periods)) lines, periods $(per) min")
    println(io, "  $(number_of_events(P)) events, $(number_of_activities(P)) activities")
    for t in (:drive,:dwell,:turnaround,:transfer,:headway)
        c[t] > 0 && println(io, "    ", rpad(String(t), 12), c[t])
    end
end

function Base.show(io::IO, S::Solution)
    P = S.problem
    v = violations(S)
    if !isempty(v)
        println(io, "NOT an admissible timetable for Problem $(P.n): $(length(v)) activities violated.")
        for a in v[1:min(3, end)]
            f, t = P.activities[a].from, P.activities[a].to
            println(io, "  ", rpad(String(P.activities[a].type), 12),
                    "$(P.events[f].station) ($(P.events[f].kind), $(P.events[f].line)) -> ",
                    "$(P.events[t].station) ($(P.events[t].kind), $(P.events[t].line))")
            println(io, "     needs $(P.activities[a].lower)..$(P.activities[a].upper) minutes, ",
                    "timetable gives $(duration(S, a))")
        end
        length(v) > 3 && println(io, "  ... and $(length(v)-3) more; see violations(S)")
        return
    end
    @printf(io, "Solution to Problem %d, score %.0f\n", P.n, score(S))
    @printf(io, "  (weighted avoidable delay, in passenger-minutes)\n\n")
    for t in (:drive,:dwell,:turnaround,:transfer,:headway)
        idx = [a for a in 1:number_of_activities(P) if P.activities[a].type === t]
        isempty(idx) && continue
        @printf(io, "  %-12s %12.0f\n", String(t),
                sum(P.activities[a].weight * slack(S, a) for a in idx))
    end
    println(io, "\n  use timetable(S, linename) to see the timetable itself")
end


"""
    activity(P, a) -> NamedTuple

Everything about activity `a` at once: its `type`, the `from` and `to` events it
joins (as [`event`](@ref) tuples), its `lower` and `upper` bounds, its `modulus`
and its `weight`.

```julia
julia> activity(Problem(7), 412)
(type = :transfer,
 from = (line = "IR35:1", station = "Bern", kind = :arrival),
 to   = (line = "IC1:0",  station = "Bern", kind = :departure),
 lower = 4, upper = 63, modulus = 60, weight = 1203.0)
```

For the endpoints as *indices* — which is what you want when walking the graph —
use [`events_of`](@ref).

See also [`event`](@ref), [`describe`](@ref), [`events_of`](@ref).
"""
activity(P::Problem, a::Integer) =
    (type = P.activities[a].type,
     from = event(P, P.activities[a].from),
     to   = event(P, P.activities[a].to),
     lower = P.activities[a].lower, upper = P.activities[a].upper,
     modulus = P.activities[a].modulus, weight = P.activities[a].weight)

"""
    describe(x) -> String

An [`event`](@ref) or an [`activity`](@ref), in words.  Which one is worked out
from the shape of the tuple, so the same call does for both.

```julia
julia> describe(event(P, 4))
"IC1:0 arrival Bern"

julia> describe(activity(P, 412))
"transfer    IR35:1 arrival Bern -> IC1:0 departure Bern   4..63 min (mod 60)  weight 1203"
```

Intended for reading diagnostic output; [`costliest`](@ref) uses it.

See also [`event`](@ref), [`activity`](@ref).
"""
describe(e::NamedTuple{(:line, :station, :kind)}) =
    "$(e.line) $(String(e.kind)) $(e.station)"

describe(a::NamedTuple{(:type, :from, :to, :lower, :upper, :modulus, :weight)}) =
    @sprintf("%-11s %s -> %s   %d..%d min (mod %d)  weight %.0f",
             String(a.type), describe(a.from), describe(a.to),
             a.lower, a.upper, a.modulus, a.weight)

"""
    scoreby(S) -> Dict{Symbol,Float64}

Where the score is going, split by activity type.

```julia
julia> scoreby(published(Problem(9)))
  transfer   5883999
  dwell      3475941
  drive       685266
```

Most of the score is usually in the connections; the running times contribute
little, because a train can rarely be made to leave much later than it must.
Worth looking at before deciding what to attack.

See also [`score`](@ref), [`costliest`](@ref).
"""
function scoreby(S::Solution)
    P = S.problem
    out = Dict{Symbol,Float64}()
    for t in (:drive, :dwell, :turnaround, :transfer, :headway)
        idx = [a for a in 1:number_of_activities(P) if P.activities[a].type === t]
        isempty(idx) && continue
        out[t] = sum(P.activities[a].weight * slack(S, a) for a in idx)
    end
    out
end

"""
    costliest(S; n = 10, type = nothing) -> Vector{Int}

The `n` activities costing the most **removable** time under `S` — largest
`weight * slack` — as indices, worst first.  Pass `type` to look at one kind
only, e.g. `type = :transfer`.

Prints them via [`describe`](@ref), so they are readable at the REPL:

```julia
julia> costliest(published(Problem(9)); n = 3, type = :transfer);
  1.  482,112  transfer  S3:1 arrival Zürich HB -> IC1:0 departure Zürich HB ...
  2.  301,554  transfer  IR16:0 arrival Bern -> IC51:1 departure Bern ...
  3.  277,010  transfer  ...
```

This is the question "where is my timetable losing time?", and the answer is
usually a handful of heavy connections at the big interchanges.

See also [`scoreby`](@ref), [`slack`](@ref), [`describe`](@ref).
"""
function costliest(S::Solution; n::Integer = 10, type = nothing)
    P = S.problem
    idx = [a for a in 1:number_of_activities(P)
           if (type === nothing || P.activities[a].type === type) &&
              P.activities[a].weight > 0]
    cost = [P.activities[a].weight * slack(S, a) for a in idx]
    ord = sortperm(cost; rev = true)[1:min(n, length(idx))]
    for (rank, k) in enumerate(ord)
        @printf("%3d. %10.0f  %s\n", rank, cost[k], describe(activity(P, idx[k])))
    end
    idx[ord]
end

"""
    gap(S, bound) -> Float64

How far timetable `S` is above a lower `bound` on the score, relatively:

    (score(S) - bound) / score(S)

**This is the number the assignment asks you to get below 0.01**, and it is also
what a solver reports as its relative gap when you give it [`score`](@ref) to
minimise — so the two agree, and no conversion is needed.

`bound` is a lower bound on the *avoidable* delay, in the same units `score`
reports.

```julia
julia> gap(S, 6_800_000.0)
0.0039
```

See also [`score`](@ref).
"""
function gap(S::Solution, bound::Real)
    sc = score(S)
    sc == 0 && return 0.0
    (sc - bound) / sc
end

"""
    stations(P) -> Vector{String}

The stations appearing in problem `P`, sorted.

See also [`stationname`](@ref), [`activities_at`](@ref).
"""
stations(P::Problem) = sort(unique(e.station for e in P.events))

"""
    activities_at(P, e) -> Vector{Int}

Every activity with event `e` at either end.  Useful for walking the
event–activity graph without rebuilding the incidence structure yourself.

```julia
julia> activities_at(Problem(3), 17)
3-element Vector{Int64}: ...
```

See also [`events_of`](@ref), [`describe`](@ref).
"""
activities_at(P::Problem, e::Integer) =
    [a for a in 1:number_of_activities(P)
     if P.activities[a].from == e || P.activities[a].to == e]

"""
    timetable(S, line)

Print one line's timetable under `S`, in the order the train runs, marking where
the times wrap past the period.

A line whose end-to-end journey is longer than its period has more than one train
working along it at once — which is exactly why the arithmetic has to be modular.

```julia
julia> timetable(published(Problem(7)), "IR95:0")
IR95:0  (every 60 minutes)
    14   departure Genève-Aéroport
    21   arrival   Genève
    ...
     2   arrival   Renens VD  <- wraps
```

See also [`lines`](@ref), [`published`](@ref).
"""
function timetable(S::Solution, line::AbstractString)
    P = S.problem
    idx = [e for e in 1:number_of_events(P) if P.events[e].line == line]
    isempty(idx) && error("no line $line; try lines(P)")
    # follow the chain of drives and dwells from the event nothing leads into
    nxt = Dict{Int,Int}()
    incoming = Set{Int}()
    for a in P.activities
        a.type in (:drive, :dwell) || continue
        P.events[a.from].line == line || continue
        nxt[a.from] = a.to; push!(incoming, a.to)
    end
    start = findfirst(e -> !(e in incoming), idx)
    order = Int[]
    e = start === nothing ? idx[1] : idx[start]
    while true
        push!(order, e)
        haskey(nxt, e) && length(order) <= length(idx) || break
        e = nxt[e]
    end
    for e in idx; e in order || push!(order, e); end
    T = P.periods[line]
    println("$line  (every $T minutes)")
    prev = nothing
    for e in order
        t = S.times[e]
        wrap = prev !== nothing && t < prev ? "  <- wraps" : ""
        @printf("  %4d   %-9s %s%s\n", t, String(P.events[e].kind), P.events[e].station, wrap)
        prev = t
    end
end

"""
    savesolution(S, path)
    savesolution(S)

Write timetable `S` to `path` as JSON, recording the problem number, the score
and the event times.  **This is the submission format** --- one file per problem.
Returns the path written.

Omit `path` and the file is named `solution_<problem>_<score>.json`, which is a
good habit: the name then carries what the file contains, and saving an improved
timetable cannot silently overwrite a better one.  The score is always a whole
number, so the name is exact.

```julia
julia> savesolution(S)
"solution_7_2093594.json"

julia> savesolution(S, "my_best_attempt.json")
"my_best_attempt.json"
```

Omitting `path` requires an admissible timetable, since an inadmissible one has
no score to name the file after; check `is_feasible(S)` first.  Saving one
explicitly is allowed, and records `"score": null`.

See also [`loadsolution`](@ref), [`is_feasible`](@ref).
"""
function savesolution(S::Solution, path::AbstractString)
    ok = is_feasible(S)
    open(path, "w") do io
        JSON.print(io, Dict("problem" => S.problem.n,
                            "score"   => ok ? Int(score(S)) : nothing,
                            "times"   => S.times))
    end
    path
end
function savesolution(S::Solution)
    is_feasible(S) || error("that timetable is not admissible, so it has no score " *
                            "to name the file after. Check violations(S); or pass a " *
                            "filename explicitly if you want to save it anyway.")
    savesolution(S, "solution_$(S.problem.n)_$(Int(score(S))).json")
end
"""
    loadsolution(path) -> Solution

Read back a timetable written by [`savesolution`](@ref).  The problem is
reloaded from its recorded number, so the result scores identically.

See also [`savesolution`](@ref).
"""
function loadsolution(path::AbstractString)
    d = JSON.parsefile(path)
    Solution(Problem(d["problem"]), Int.(d["times"]))
end

end # module
