scalars = dict(
    earth_r = 6371
)

fiber_path_analysis = dict(
    n_clusters = 1,
    use_road_data = True,
    max_connection_length = 50,
    max_distance_for_direct_road_merge = 2,
)

visibility_analysis = dict(
    n_clusters = 1,
    poi_building_height = 15,
    tower_height = 45,
    max_tower_reach = 35,
    n_visible = 3,
)

mapping_analysis = dict(
    
    population = dict(
        data_year = 2020,
        dataset_1km_resolution = False,
        dataset_UNadj = True,
        area_radius = 5,
    ),

    coverage = dict(
        with_variable_signal_strength = False,
    )
)