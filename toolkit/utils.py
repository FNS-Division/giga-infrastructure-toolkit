import pandas as pd
import geopandas as gp
from shapely.geometry import Point
from shapely import wkt


def data_read(filepath):

    file_format = filepath.split('.')[-1]

    if file_format == 'csv':
        df = pd.read_csv(filepath)
    elif file_format == 'xlsx':
        df = pd.read_excel(filepath, engine='openpyxl')
    elif file_format == 'json':
        df = pd.read_json(filepath)
    elif file_format == 'parquet':
        df = pd.read_parquet(filepath, engine='fastparquet')
    elif file_format =='geoparquet':
        df = gp.read_parquet(filepath)
    elif file_format == 'geojson':
        df = gp.read_file(filepath)
    elif file_format == 'zip' and filepath.split('.')[-2] == 'shp':
        df = gp.read_file(filepath)
    elif file_format =='gpkg':
        df = gp.read_file(filepath)
    else:
        raise ValueError('Allowed file formats: "csv", "xlsx", "json", "parquet", "geoparquet", "geojson", "shp.zip" and "gpkg". Please update the file format and try again!')
    
    return df


def get_geo_column_names(df):

    geo_cols = list(filter(lambda a: a.casefold() in ['lon', 'lat', 'longitude', 'latitude', 'long', 'x', 'y', 'lat_', 'lon_', 'lat(s)', 'long(e)', 'lon(e)'], df.columns))

    if len(geo_cols)>2:
        raise ValueError('Please keep only the relevant geo reference columns. Multiple matches are found: ' + str(geo_cols))
    elif len(geo_cols) <2:
        raise ValueError('Dataframe does not include any relevant geometry column names! Example geo reference column names: "lat", "Lat", "Latitude", "LATITUDE", "y"')

    if geo_cols[0] in ['y', 'Y']:
        geo_cols.sort()
    else:
        geo_cols.sort(reverse=True)
    
    return geo_cols


def rename_geo_cols(df):
    
    geo_cols = get_geo_column_names(df)
    if geo_cols != ['lon', 'lat']:
        df.rename(columns = {geo_cols[0]: 'lon', geo_cols[1]: 'lat'}, inplace = True)
        return print('Georeference columns are renamed as "lat", "lon"!')


def initialize_geometry_column(df):

    if 'geometry' in df:
        return print('Geometry column has already been initialized! If you want to reinitialize please drop and recall the function.')

    geo_cols = get_geo_column_names(df)

    try:    
        df['geometry'] = [Point(i, j) for i,j in zip(df[geo_cols[0]], df[geo_cols[1]]) if i]
    except:
        df[geo_cols[0]] = df[geo_cols[0]].astype(float)
        df[geo_cols[1]] = df[geo_cols[1]].astype(float)
        df['geometry'] = [Point(i, j) for i,j in zip(df[geo_cols[0]], df[geo_cols[1]]) if i]
    
    return df



def df_to_gdf(df, rename = True, crs = 'EPSG:4326'):

    if 'geometry' not in df:
        df = initialize_geometry_column(df)
    elif df['geometry'].dtypes != 'geometry':
        df['geometry'] = df['geometry'].apply(wkt.loads)
    
    if rename:
        rename_geo_cols(df)

    return gp.GeoDataFrame(df, crs = crs)