poi = dict(
    school = dict(
        id_column = 'giga_school_id',
        filepath = '/data/poi/zw_schools_iinfrastructure_attributes_added.xlsx'
    ),
)

tower = dict(
    LTE = dict(
        id_column = 'ict_id',
        filepath = '/data/tower/zwe_4G_processed.csv'
    ),
    non_LTE = dict(
        id_column = 'ict_id',
        filepath = '/data/tower/'
    ),
)

fiber = dict(
    exact = dict(
        id_column = 'ict_id',
        filepath = '/data/fiber/zwe_exact.csv'
    ),
    geo_referenced = dict(
        id_column = 'ict_id',
        filepath = '/data/fiber/zwe_georef.csv'
    ),
)