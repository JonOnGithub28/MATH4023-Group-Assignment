module TaktMap

# UTILS
function push_unique!(vect, item)
    !(item in vect) && push!(vect, item)
end

using Takt  # Local packages
using Statistics, CSV, DataFrames, CairoMakie

struct TrainLine
    name::String;
    stations::Vector{String};
end

struct Station
    name::String;
    # Offset LR95 coordinates, with Bern now at (0,0)
    E::Float64;
    N::Float64;
end

# Data on all lines and stations needed to map out the problem
struct ProblemMapData
    lines::Vector{TrainLine};
    stations::Vector{Station};
end

# The projection center of the LR95 system is the old Bern Observatory 
# (*not* the Zimmerwold observatory, which is a refence point in a different sense).
# The LR95 coordinates of Bern are:
BERN_LR95_E = 2_600_000.0
BERN_LR95_N = 1_200_000.0

function update_from_event(
    e::Takt.Event,
    station_name_list::Vector{String},
    line_list::Vector{TrainLine},
)
    # DEBUG: check which lines contain Feldkirch
    #if e.station == "Feldkirch"
    #    println("line $e.line contains FeldKirch")
    #end

    # Check if we've seen the line before
    push_unique!(station_name_list, e.station)
    for line in line_list
        if line.name == e.line
            # Already have this line
            push_unique!(line.stations, e.station)
            return
        end
    end

    # This is a new line
    new_line = TrainLine(e.line, [e.station])

    push!(line_list, new_line)
end

function load_railway_df()
    script_dir = @__DIR__
    file_name = "actual-date-swiss-service-point-MODIFIED.csv"
    file_path = joinpath(script_dir, file_name)
    return CSV.read(
        file_path, 
        DataFrame,
        select = [:designationOfficial, :lv95East, :lv95North]
    )
end

function fetch_station_info(station_name, railway_df)
    # Deliberately will raise an error if we do not find EXACTLY ONE record of a station with this name

    # There is EXACTLY ONE station (Feldkirch) whose name does not show up 
    # in this data. After some research, I've determined it is Feldkirch, Bahnhof.
    if station_name == "Feldkirch"
        station_data = only(railway_df[railway_df.designationOfficial .== "Feldkirch, Bahnhof", :])
    else
        # The original data is missing LR coordinates for:
        #    Erzingen (Baden): 2,674,481E, 1,279,237
        # I've manually added this data into the csv file attached with this code
        station_data = only(railway_df[railway_df.designationOfficial .== station_name, :])
    end

    return Station(
        station_name, 
        station_data.lv95East - BERN_LR95_E,
        station_data.lv95North - BERN_LR95_N,
    )
end

"""
Given a Takt.Problem(), returns a TaktMap.ProblemMapData() object
which on which TaktMap.plot_map_data() may be called to produce
a visual.
"""
function generate_problem_map_data(P::Takt.Problem)
    # Fetch the station names and lines``
    station_name_list = String[]
    line_list = TrainLine[]
    for e in P.events
        update_from_event(e, station_name_list, line_list)
    end

    # Fetch coordinates of each station and form the Station() list
    railway_df = load_railway_df()
    station_list = Station[]
    for station_name in station_name_list
        push!(station_list, fetch_station_info(station_name, railway_df))
    end

    return ProblemMapData(line_list, station_list)
end

function coords_of_station(data::ProblemMapData, station_name::String)
    for station in data.stations
        if station.name == station_name
            return (station.E, station.N)
        end
    end
    error("station not in data")
end

# Simple angle between vectors function
function angle_between(x_1, y_1, x_2, y_2)
    dot_prod = x_1 * x_2 + y_1 * y_2
    norm_1 = sqrt(x_1^2 + y_1^2)
    norm_2 = sqrt(x_2^2 + y_2^2)
    return acos(dot_prod/(norm_1 * norm_2))
end

# Tests to see whether the second vector is to the left
# of the first vector, in the sense that it lies to the left of
# the directed line the first one spans
function second_left_of_first(x_1, y_1, x_2, y_2)
    # Strategy is to rotate the first vector counterclockwise
    # and then take its dot product with the second
    x_1_rot = -y_1
    y_1_rot = x_1
    dot_prod = x_1_rot * x_2 + y_1_rot * y_2
    return dot_prod >= 0  # for our purposes we include the zero case
end

function rotate_vector(x, y, theta)
    x_rot = cos(theta) * x - sin(theta) * y 
    y_rot = sin(theta) * x + cos(theta) * y 
    return x_rot, y_rot
end

function vector_norm(x, y)
    return sqrt(x^2 + y^2)
end

# Gets the unit-length vector angular bisector on the left side
# (here we imagine vector 1 ending where vector 2 begins)
function unit_left_side_bisector(x_1, y_1, x_2, y_2)
    angle = angle_between(-x_1, -y_1, x_2, y_2)
    small_angle_on_left = second_left_of_first(x_1, y_1, x_2, y_2)
    
    if small_angle_on_left
        x_2_rot, y_2_rot = rotate_vector(x_2, y_2, angle/2)
        norm_2 = vector_norm(x_2, y_2)
        x_3 = x_2_rot/norm_2
        y_3 = y_2_rot/norm_2
        return (x_3, y_3)
    else
        x_2_rot, y_2_rot = rotate_vector(x_2, y_2, -angle/2)
        norm_2 = vector_norm(x_2, y_2)
        x_3 = x_2_rot/norm_2
        y_3 = y_2_rot/norm_2
        return (-x_3, -y_3)
    end
end

# Shifts the points on a line to the "left", to model the fact
# swiss trains drive on the left (and to prevent overlap between lines going 
# in distinct directions)
function left_shift_points(E_vals, N_vals, shift)
    # We need to have at least two points for this to make sense
    @assert length(E_vals) >= 2
    @assert length(N_vals) >= 2

    @assert length(N_vals) == length(E_vals)
    n = length(N_vals)
    i = 1

    shifted_E_vals = Float64[]
    shifted_N_vals = Float64[]

    local delta_E_last
    local delta_N_last
    local delta_E_next
    local delta_N_next

    # The first and last point need to be handled separately

    let  # First point
        delta_E_next = E_vals[2] - E_vals[1]
        delta_N_next = N_vals[2] - N_vals[1]

        delta_norm = vector_norm(delta_E_next, delta_N_next)
        delta_E_rot = -delta_N_next/delta_norm
        delta_N_rot = delta_E_next/delta_norm

        push!(shifted_E_vals, E_vals[1] + shift * delta_E_rot)
        push!(shifted_N_vals, N_vals[1] + shift * delta_N_rot)

        i += 1
        delta_E_last = delta_E_next
        delta_N_last = delta_N_next
    end

    while i <= n-1
        delta_E_next = E_vals[i+1] - E_vals[i]
        delta_N_next = N_vals[i+1] - N_vals[i]

        (b_E, b_N) = unit_left_side_bisector(
            delta_E_last, delta_N_last, 
            delta_E_next, delta_N_next
        )

        push!(shifted_E_vals, E_vals[i] + shift * b_E)
        push!(shifted_N_vals, N_vals[i] + shift * b_N)

        delta_E_last = delta_E_next
        delta_N_last = delta_N_next
        i += 1
    end

    let  # Last point
        delta_norm = vector_norm(delta_E_next, delta_N_next)
        delta_E_rot = -delta_N_last/delta_norm
        delta_N_rot = delta_E_last/delta_norm

        push!(shifted_E_vals, E_vals[n] + shift * delta_E_rot)
        push!(shifted_N_vals, N_vals[n] + shift * delta_N_rot)
    end

    return shifted_E_vals, shifted_N_vals
end



MAP_FONT_SIZE = 8

"""
Produces a plot to visualize a given TaktMap.ProblemMapData() object.

Options:
- tight: If true, the bounding box is set based on the specific problem given. Otherwise uses a bounding box for P12. Set to true if you want a tight graph for one problem, set to false if you want graphs that are more easily comparable between problems.
"""
function plot_map_data(
    data::ProblemMapData, 
    tight::Bool,  # whether to use tight square bounds for problem in question (defaults to using the square bounds for P12 otherwise)
)
    fig = Figure(size = (2400,2400))
    ax = Axis(fig[1,1], aspect = DataAspect())  # anything except 1 to 1 scaling will break the directional marker code

    # Plot the stations
    station_E_vals = [station.E for station in data.stations]
    station_N_vals = [station.N for station in data.stations]
    scatter!(ax, station_E_vals, station_N_vals,
        color = :black,
        markersize = 14,
    )
    station_names = [station.name for station in data.stations]
    #annotation!(ax, E_values, N_values, text = station_names,
    #    fontsize = 8
    #)


    # Plot the train lines
    line_number = 0
    for line in data.lines
        line_number += 1
        line_color = Cycled(line_number) 

        line_E_vals = Float64[]
        line_N_vals = Float64[]
        for station in line.stations
            (E,N) = coords_of_station(data, station)
            push!(line_E_vals, E)
            push!(line_N_vals, N)
        end
        n = length(line_E_vals)

        LEFT_RIGHT_HALF_SPACING = 600  # half the distance between the left and right tracks used for plotting, in meters
        line_E_vals_shifted, line_N_vals_shifted = left_shift_points(
            line_E_vals, line_N_vals, LEFT_RIGHT_HALF_SPACING
        )

        # Determine where we want to place the directional markers
        E_dir_vals = Float64[]  # for directional (arrow) markers
        N_dir_vals = Float64[]
        rotation_angles = Float64[]  # rotational angles for directional markers
        DIRECTIONAL_STEP_PROPORTION = 0.33  # how far along each segment the direction marker should be placed
        for i in 1:(n-1)
            # Could optimise, as we compute these deltas anyway in shifting the points 
            # but this is unimportant
            delta_E = line_E_vals[i+1] - line_E_vals[i]
            delta_N = line_N_vals[i+1] - line_N_vals[i]

            push!(E_dir_vals, line_E_vals_shifted[i] + delta_E * DIRECTIONAL_STEP_PROPORTION)
            push!(N_dir_vals, line_N_vals_shifted[i] + delta_N * DIRECTIONAL_STEP_PROPORTION)
            push!(rotation_angles, atan(delta_N, delta_E))
        end

        lines!(ax, line_E_vals_shifted, line_N_vals_shifted, 
            label = line.name,
            color = line_color,
            alpha = 0.7
        )
        scatter!(ax, E_dir_vals, N_dir_vals,
            markersize = 8,
            rotation = rotation_angles,
            marker = :rtriangle,
            color = line_color,
        )
    end


    # Set x and y lims for plot
    if tight
        min_E = minimum(station_E_vals)
        max_E = maximum(station_E_vals)
        E_range = max_E - min_E
        min_N = minimum(station_N_vals)
        max_N = maximum(station_N_vals)
        N_range = max_N - min_N
        PADDING = 0.1
        xlims!(ax, 
            min_E - PADDING * E_range,
            max_E + PADDING * E_range
        )
        ylims!(ax, 
            min_N - PADDING * N_range,
            max_N + PADDING * N_range
        )
    else
        # Result of performing above for P12
        E_low = -116794
        E_high = 201197
        N_low = -176671
        N_high = 141320

        xlims!(ax, E_low, E_high)
        ylims!(ax, N_low, N_high)
    end

    #=
    min_E = minimum(station_E_vals)
    max_E = maximum(station_E_vals)
    E_range = max_E - min_E
    min_N = minimum(station_N_vals)
    max_N = maximum(station_N_vals)
    N_range = max_N - min_N
    padding = 0.1
    xlims!(ax, 
        min_E - padding * E_range,
        max_E + padding * E_range
    )
    ylims!(ax, 
        min_N - padding * N_range,
        max_N + padding * N_range
    )
    =#


    Legend(fig[1, 2], ax)
    save("map.svg", fig)
end

export generate_problem_map_data
export plot_map_data

#P = Problem(12)
#data = generate_problem_map_data(P)
#plot_map_data(data, true)

end # module TaktMap
