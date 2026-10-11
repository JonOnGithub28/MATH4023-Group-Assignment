module TaktMap

# UTILS
function push_unique!(vect, item)
    !(item in vect) && push!(vect, item)
end

using Takt  # Local packages
using LinearAlgebra, Statistics, CSV, DataFrames, CairoMakie
using Colors #ColorSchemes

struct TrainLine
    name::String;
    stations::Vector{String};
    lanes::Vector{Int};  # Which "lane" does the train run on in each step
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
    route_usage::Dict{Tuple{String, String}, Int},
)
    # Check if we've seen the line before
    push_unique!(station_name_list, e.station)
    for line in line_list
        if line.name == e.line
            # Already have this line
            if !(e.station in line.stations)
                last_station = line.stations[end]
                route = (last_station, e.station)
                if haskey(route_usage, route)
                    route_usage[route] += 1
                else
                    route_usage[route] = 1
                end
                push!(line.stations, e.station)
                push!(line.lanes, route_usage[route])
            end
            return
        end
    end

    # This is a new line
    new_line = TrainLine(e.line, [e.station], [])

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

    route_usage = Dict{Tuple{String, String}, Int}()  # How many trains we've seen that run from station A to station B directly
    for e in Iterators.reverse(P.events)  # For whatever reason, the new lines from each problem are added at the START of events vector. To get a consistent order we need to reverse this.
        update_from_event(e, station_name_list, line_list, route_usage)
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

# Given two parameterized lines (in {p + tv : t free} form),
# returns the point of intersection
function parametrized_lines_bisector(
    p1_x, p1_y, v1_x, v1_y,
    p2_x, p2_y, v2_x, v2_y,
)
    A = [v1_x  -v2_x; v1_y  -v2_y]
    b = [p2_x - p1_x, p2_y - p1_y]
    t,s = A \ b
    return p1_x + t * v1_x, p1_y + t * v1_y
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
function left_shift_points(
    E_vals, 
    N_vals,
    lane_numbers,
    shift
)
    # We need to have at least two points for this to make sense
    @assert length(E_vals) >= 2
    @assert length(N_vals) >= 2

    @assert length(N_vals) == length(E_vals)
    n = length(N_vals)
    i = 1

    # Shift needs to be handled in two stages:
    # 1. Shift each segment individually
    # 2. Reconcile the different shifts of adjacent segments

    # Stage 1
    stage_1_shifted_E_vals = Float64[]
    stage_1_shifted_N_vals = Float64[]
    delta_E_vals = Float64[]
    delta_N_vals = Float64[]
    for i in 1:(n-1)
        delta_E = E_vals[i+1] - E_vals[i]
        delta_N = N_vals[i+1] - N_vals[i]
        push!(delta_E_vals, delta_E)
        push!(delta_N_vals, delta_N)

        delta_norm = vector_norm(delta_E, delta_N)
        delta_E_rot = -delta_N/delta_norm
        delta_N_rot = delta_E/delta_norm

        lane_num = lane_numbers[i]

        push!(stage_1_shifted_E_vals, E_vals[i] + shift * lane_num * delta_E_rot)
        push!(stage_1_shifted_N_vals, N_vals[i] + shift * lane_num * delta_N_rot)
        push!(stage_1_shifted_E_vals, E_vals[i+1] + shift * lane_num * delta_E_rot)
        push!(stage_1_shifted_N_vals, N_vals[i+1] + shift * lane_num * delta_N_rot)
    end

    # Stage 2
    shifted_E_vals = Float64[stage_1_shifted_E_vals[1]]
    shifted_N_vals = Float64[stage_1_shifted_N_vals[1]]
    for i in 1:(n-2)
        # The ideal approach is to replace the two shifted points at each
        # intermediate station with the point of intersection between the 
        # corresponding segments. However, the linear system needed to calculate
        # this becomes unstable when the shifts are close to parallel, causing stops to be quite far off.
        # In that case, we just use the average position instead.

        local new_E
        local new_N
        a = dot((delta_E_vals[i], delta_N_vals[i]), (delta_E_vals[i+1], delta_N_vals[i+1]))
        if 1 - abs(a) < 0.1  # arbitrary cutoff, chosen by eye
            # Shifts are essentially parallel
            new_E = (stage_1_shifted_E_vals[2*i] + stage_1_shifted_E_vals[2*i+1])/2
            new_N = (stage_1_shifted_N_vals[2*i] + stage_1_shifted_N_vals[2*i+1])/2
        else 
            # Shifts are different enough, so we use the intersection
            new_E, new_N = parametrized_lines_bisector(
                stage_1_shifted_E_vals[2*i], stage_1_shifted_N_vals[2*i], delta_E_vals[i], delta_N_vals[i],
                stage_1_shifted_E_vals[2*i+1], stage_1_shifted_N_vals[2*i+1], delta_E_vals[i+1], delta_N_vals[i+1],
            )
        end

        push!(shifted_E_vals, new_E)
        push!(shifted_N_vals, new_N)
    end
    push!(shifted_E_vals, stage_1_shifted_E_vals[end])
    push!(shifted_N_vals, stage_1_shifted_N_vals[end])

    return shifted_E_vals, shifted_N_vals
end

MAP_FONT_SIZE = 8
LINE_WIDTH = 300 # Measured in data scale
LANE_SPACING = 700  # Distance (meters) between any two lanes on the same side, or twice the distance between the innermost left and right lanes.


# Determines the assignment of train lines to colors, based on their 3-letter prefix,
# so the same route in different directions has the same color.
function generate_color_key(line_names::Vector{String})
    prefixes_seen = String[]
    for name in line_names
        prefix = String(split(name, ':')[1])
        push_unique!(prefixes_seen, prefix)
    end

    num_prefixes = length(prefixes_seen)
    color_pool = distinguishable_colors(num_prefixes, [colorant"white"];
        dropseed = true,
        lchoices = range(0, 50, length = 15),
        cchoices = range(30, 100, length = 15)
    )
    return Dict(zip(prefixes_seen, color_pool))
end

"""
Produces a plot to visualize a given TaktMap.ProblemMapData() object.

Options:
- tight: If true, the bounding box is set based on the specific problem given. Otherwise uses a bounding box for P12. Set to true if you want a tight graph for one problem, set to false if you want graphs that are more easily comparable between problems.
"""
function save_plot_of_map_data(
    data::ProblemMapData,
    name::String,
    tight::Bool,  # whether to use tight square bounds for problem in question (defaults to using the square bounds for P12 otherwise)
    labels::Bool,  # whether to include station labels
)
    fig = Figure(size = (600,600))
    ax = Axis(fig[1,1], aspect = DataAspect())  # anything except 1 to 1 scaling will break the directional marker code

    # reactive screen-to-data scale (only needed for plotting the lines, which CairoMakie does not allow to be set to data scale natively)
    s2d = lift(
        (flims, vp) -> widths(vp)[1]/widths(flims)[1], 
        ax.finallimits,    # limits of the plot in screen coordinates
        ax.scene.viewport  #                       data
    )

    line_size = @lift(LINE_WIDTH * $s2d)
    line_marker_size = LINE_WIDTH * 4  # size used for marks along a particular line (i.e. the directional and start/end marks)
    station_size = LINE_WIDTH * 10
    font_size = LINE_WIDTH * 3

    # Plot the stations
    station_E_vals = [station.E for station in data.stations]
    station_N_vals = [station.N for station in data.stations]
    scatter!(ax, station_E_vals, station_N_vals,
        markerspace = :data,
        markersize = station_size,
        color = :gray,
    )

    # Determine the colors assigned to each line, based on their 3 letter prefix
    line_names = [line.name for line in data.lines]
    line_color_key = generate_color_key(line_names)

    # Plot the train lines
    for line in data.lines
        prefix = String(split(line.name, ':')[1])
        line_color = line_color_key[prefix]

        line_opacity = 0.7

        line_E_vals = Float64[]
        line_N_vals = Float64[]
        for station in line.stations
            (E,N) = coords_of_station(data, station)
            push!(line_E_vals, E)
            push!(line_N_vals, N)
        end

        line_E_vals_shifted, line_N_vals_shifted = left_shift_points(
            line_E_vals, line_N_vals, line.lanes, LANE_SPACING
        )

        # Determine where we want to place the directional markers
        n = length(line_E_vals_shifted)
        E_dir_vals = Float64[]  # for directional (arrow) markers
        N_dir_vals = Float64[]
        rotation_angles = Float64[]  # rotational angles for directional markers
        DIRECTIONAL_STEP_PROPORTION = 0.33  # how far along each segment the direction marker should be placed
        for i in 1:(n-1)
            delta_E = line_E_vals_shifted[i+1] - line_E_vals_shifted[i]
            delta_N = line_N_vals_shifted[i+1] - line_N_vals_shifted[i]

            push!(E_dir_vals, line_E_vals_shifted[i] + delta_E * DIRECTIONAL_STEP_PROPORTION)
            push!(N_dir_vals, line_N_vals_shifted[i] + delta_N * DIRECTIONAL_STEP_PROPORTION)
            push!(rotation_angles, atan(delta_N, delta_E))
        end

        # The line itself
        lines!(ax, line_E_vals_shifted, line_N_vals_shifted,
            color = line_color,
            linewidth = line_size,
            label = line.name,
            alpha = line_opacity
        )

        # Start point
        scatter!(ax, line_E_vals_shifted[1], line_N_vals_shifted[1],
            markerspace = :data,
            markersize = line_marker_size,
            marker = :circle,
            color = :transparent,
            strokecolor = line_color,
            strokewidth = 0.25,
            alpha = line_opacity
        )

        # End point
        scatter!(ax, line_E_vals_shifted[end], line_N_vals_shifted[end],
            markerspace = :data,
            markersize = line_marker_size,
            marker = :cross,
            color = line_color,
            alpha = line_opacity
        )

        # Intermediate stops
        scatter!(ax, line_E_vals_shifted[2:end-1], line_N_vals_shifted[2:end-1], 
            markerspace = :data,
            markersize = line_marker_size,
            marker = :circle,
            color = line_color,
            alpha = line_opacity,
        )

        # Directional markers
        scatter!(ax, E_dir_vals, N_dir_vals,
            markerspace = :data,
            markersize = line_marker_size,
            rotation = rotation_angles,
            marker = :rtriangle,
            color = line_color,
        )
    end

    # Station names
    if labels
        station_names = [station.name for station in data.stations]
        annotation!(ax, station_E_vals, station_N_vals,
            text = station_names,
            labelspace = :data,
            fontsize = 3,
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
        max_range = max(E_range, N_range)
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
        E_low = -131248
        E_high = 215651
        N_low = -139393
        N_high = 104042

        xlims!(ax, E_low, E_high)
        ylims!(ax, N_low, N_high)
    end

    hidedecorations!(ax)
    hidespines!(ax)

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

    legend_rows = cld(length(data.lines), 12)
    Legend(fig[2, 1], ax, 
        orientation = :horizontal, 
        nbanks = legend_rows, 
        framevisible = false,
        labelsize = 5,
        patchsize = (5, 2.5)
    )
    save("$name.svg", fig)
end

export generate_problem_map_data
export save_plot_of_map_data

end # module TaktMap
