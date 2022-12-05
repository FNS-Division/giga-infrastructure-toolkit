# 2022 Giga

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from shapely.geometry import Point, box, LineString
import numpy as np
from shapely import wkt
import geopandas as gp
import pandas as pd
from osgeo import gdal
from pyrosm import get_data, OSM
from pyrosm.data import sources
from sklearn.cluster import KMeans
from urllib import request
import datetime
from keplergl import KeplerGl
from sklearn.neighbors import KDTree
from scipy.sparse.csgraph import minimum_spanning_tree
from srtm.height_map_collection import Srtm1HeightMapCollection
import grispy as gsp
import itertools
import requests
import gzip
import logging
import http.cookiejar as cookielib
import pycountry
from bs4 import BeautifulSoup
import math
import networkx as nx
import geonetworkx as gnx
from pathlib import Path
import base64
from tqdm import tqdm
import sys
import os



def obfuscate(plainText):
    plainBytes = plainText.encode('ascii')
    encodedBytes = base64.b64encode(plainBytes)
    encodedText = encodedBytes.decode('ascii')
    return encodedText


def deobfuscate(obfuscatedText):
    obfuscatedBytes = obfuscatedText.encode('ascii')
    decodedBytes = base64.b64decode(obfuscatedBytes)
    decodedText = decodedBytes.decode('ascii')
    return decodedText


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
    elif file_format == 'geojson':
        df = gp.read_file(filepath)
    elif file_format == 'zip' and filepath.split('.')[-2] == 'shp':
        df = gp.read_file(filepath)
    elif file_format =='gpkg':
        df = gp.read_file(filepath)
    else:
        raise ValueError('Allowed file formats: "csv", "xlsx", "json", "parquet", "geojson", "shp.zip" and "gpkg". Please update the file format and try again!')
    
    return df


def get_geo_column_names(df):

    geo_cols = list(filter(lambda a: a.casefold() in ['lon', 'lat', 'longitude', 'latitude', 'long', 'x', 'y', 'lat_', 'lon_', 'lat(s)', 'long(e)', 'lon(e)'], df.columns))

    if len(geo_cols)>2:
        raise ValueError('Please keep only the relevant geo refernce columns. We find multiple matches: ' + str(geo_cols))
    elif len(geo_cols) <2:
        raise ValueError('Dataframe does not include any relevant geometry column names! Example geo reference column names: "lat", "Lat", "Latitude", "LATITUDE", "y"')

    geo_cols.sort(reverse=True)

    if geo_cols[0] in ['y', 'Y']:
        geo_cols.sort()
    
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


def tif_to_df(tif):

    try:
        srcband = tif.GetRasterBand(1).ReadAsArray()
    except RuntimeError as e:
        print('Band 1 not found')
        print(e)
        sys.exit(1)

    flat = srcband.flatten()
    flat = np.where(flat == -99999, 0, flat)

    gt = tif.GetGeoTransform()

    res = gt[1]

    xmin = gt[0]
    ymax = gt[3]

    xsize = tif.RasterXSize
    ysize = tif.RasterYSize

    xstart = xmin + res/2
    ystart = ymax - res/2

    x = np.arange(xstart, xstart + xsize*res, res)
    y = np.arange(ystart, ystart - ysize*res, -res)

    if len(x) != xsize:
        x = x[:-1]

    if len(y) != ysize:
        y = y[:-1]

    x = np.tile(x, ysize)
    y = np.repeat(y, xsize)

    tif_dict = {'lon': x, 'lat': y, 'population': flat}
    df_pop = pd.DataFrame(tif_dict)

    return df_pop
    

def get_osm_road_data(dataset, locations = None, network_type = 'all', buffer_around_bbox = 0, nodes_id_column = 'id', verbose = True):

    if dataset in sources._all_sources:
        file_x = get_data(dataset)
    elif os.path.exists(dataset):
        file_x = dataset
    else:
        raise RuntimeError('Either OSM dataset name or OSM file path should be provided! Available datasets can be checked using methods in the following page: https://pyrosm.readthedocs.io/en/latest/basics.html#available-datasets')
    
    if verbose:
        print('Reading OSM file...')

    if buffer_around_bbox>0:
        assert locations is not None, 'If use_bbox set as True then set of locations geodataframe should be provided!'
        assert type(locations) == gp.geodataframe.GeoDataFrame, 'Provided locations should be in the type of geopandas geodataframe!'

        xmin, ymin, xmax, ymax = locations.total_bounds
        bbox_ = box(xmin, ymin, xmax, ymax)
        osm = OSM(file_x, bounding_box=bbox_.buffer(km2deg(buffer_around_bbox, 6371)))
    else:
        osm = OSM(file_x)
    
    if verbose:
        print('Extracting nodes and edges in the OSM data...')

    nodes, edges = osm.get_network(nodes=True, network_type=network_type)

    assert len(edges)>0, 'OSM road data is missing or non-exist for given locations!'

    nodes = nodes.loc[:, ['lon', 'lat', 'id', 'geometry']]
    nodes['label'] = None
    edges = edges.loc[:, ['u','v', 'id', 'geometry', 'length']]
    edges['label'] = None


    nodes.set_index(nodes_id_column, drop = False, inplace = True)
    #nodes.lon[nodes.lon.isnull()], nodes.lat[nodes.lat.isnull()] = nodes.geometry.x, nodes.geometry.y

    if verbose:
        print('OSM nodes and edges are extracted!')
    
    return osm, nodes, edges



def initialize_geograph(nodes, edges, crs = gnx.WGS84_CRS, is_undirected = True, edges_from_to_columns = ['u', 'v'], verbose = True):

    if verbose:
        print('Initializing geograph using nodes and edges...')

    gx = gnx.GeoGraph(crs = crs)
    gx.add_nodes_from_gdf(nodes)

    try:
        edges.set_index(edges_from_to_columns)
    except:
        raise ValueError('From to column names are not specified correctly.')

    gx.add_edges_from_gdf(edges.set_index(edges_from_to_columns))
    
    if is_undirected:
        gx = gx.to_undirected()
    
    gx_nodes, gx_edges = gx.nodes_to_gdf(), gx.edges_to_gdf()

    lon_missing_idx= gx_nodes.loc[gx_nodes.lon.isnull()].index
    gx_nodes.loc[lon_missing_idx, 'lon'], gx_nodes.loc[lon_missing_idx, 'lat'] = gx_nodes.loc[lon_missing_idx, 'geometry'].x, gx_nodes.loc[lon_missing_idx, 'geometry'].y

    if verbose:
        print('Geograph is initialized!')
    
    #gnx.fill_elevation_attribute(gx)
    
    return gx, gx_nodes, gx_edges


def upper_triangle_to_full_dmx(val_list, n):
    mask = ~np.tri(n, k=0, dtype = bool)
    out = np.zeros((n,n))
    out[mask] = val_list
    out.T[mask] = val_list
    return out


def geo_dist(coords_1, coords_2):
    R = 6371.0 # radius of the earth
    lon1 = math.radians(coords_1[0])
    lat1 = math.radians(coords_1[1])
    lon2 = math.radians(coords_2[0])
    lat2 = math.radians(coords_2[1])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2 # Haversine formula
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c
    return distance

def km2deg(x, R):
    return x * np.rad2deg(1/R)

def deg2km(x, R):
    return np.deg2rad(x) * R


def download_srtm_data(username, password, url, path):

    password_manager = request.HTTPPasswordMgrWithDefaultRealm()
    password_manager.add_password(None, "https://urs.earthdata.nasa.gov", username, password)

    cookie_jar = cookielib.CookieJar()

    # Install all the handlers.

    opener = request.build_opener(
        request.HTTPBasicAuthHandler(password_manager),
        #request.HTTPHandler(debuglevel=1),    # Uncomment these two lines to see
        #request.HTTPSHandler(debuglevel=1),   # details of the requests/responses
        request.HTTPCookieProcessor(cookie_jar))
    request.install_opener(opener)


    # retrieve the file from url

    request.urlretrieve(url, path)



def get_opencellid_urls(country_code, opencellid_access_token):
    
    try:
        country_alpha2 = pycountry.countries.get(alpha_3 = country_code.upper()).alpha_2
    except:
        raise ValueError('ISO3 country code is not valid! Please make sure you have entered valid ISO3 country code.')

    url = "https://opencellid.org/downloads.php?token="+str(opencellid_access_token)

    # find table
    html_content = requests.get(url).text 
    soup = BeautifulSoup(html_content, "lxml")
    table = soup.find("table", {'id':"regions"})

    # get header
    t_headers = []
    for th in table.find_all("th"):
        t_headers.append(th.text.replace('\n', ' ').strip())

    table_data = []
    # for all the rows of table
    for tr in table.tbody.find_all('tr'):
        t_row = {}
        
        for td, th in zip(tr.find_all("td"), t_headers):
            if 'Files' in th:
                t_row[th]=[]
                for a in td.find_all('a'):
                    t_row[th].append(a.get('href'))
            else:
                t_row[th] = td.text.replace('\n', '').strip()
        
        table_data.append(t_row)

    cell_dict = pd.DataFrame(table_data)

    ## get the links for the country code
    if country_alpha2 not in cell_dict['Country Code'].values:
        print('Country code is invalid or not exist in OpenCelliD database!')
        sys.exit(1)
    else:
        links = cell_dict[cell_dict['Country Code']==country_alpha2]['Files (grouped by MCC)'].values[0]
    
    return links


def get_opencellid_data(country_code, data_path, write_data = False):
    links = get_opencellid_urls(country_code)
    colnames = ['radio', 'mcc', 'net', 'area', 'cell', 'unit', 'lon', 'lat', 'range', 'samples', 'changeable', 'created', 'updated', 'averageSignal']
    df_cell = pd.DataFrame()

    for link in links:
        response = requests.get(link, stream=True)
        temp_file = os.path.join(data_path, 'raw', 'opencellid_' + country_code.lower()+'.csv.gz.tmp')

        totes_chunks = 0
        with open(temp_file, 'wb') as feed_file:
            for chunk in tqdm(response.iter_content(chunk_size=1024)):
                if chunk:
                    feed_file.write(chunk)
                    totes_chunks += 1024
        try:
            with gzip.open(temp_file, 'rt') as feed_data:
                df_cell = pd.concat([df_cell, pd.read_csv(feed_data, names=colnames, header=None)], ignore_index=True)
        except IOError:
            rate_limit = 'RATE_LIMITED'
            bad_token = 'INVALID_TOKEN'
            with open(temp_file, 'r') as eggs_erroneous:
                contents = eggs_erroneous.readline()
            if rate_limit in contents:
                logging.error("Feed did not update. You're rate-limited!")
            elif bad_token in contents:
                logging.error("API token rejected by Unwired Labs!!")
            else:
                logging.error("Non-specific error.  Details in %s", temp_file)
            raise
    os.remove(temp_file)

    if write_data:
        file_name = 'opencellid_' + country_code.lower() + '.parquet'
        print('Writing country cell tower data to the data directory...')
        df_cell.to_parquet(os.path.join(data_path, 'raw', file_name), index=False)
        print('Done! Dataset is saved with a name: ' + file_name)

    return df_cell



def get_population_data(country_code, dataset_year, data_path, one_km_res = False, un_adjusted = True, worldpop_base_url = 'https://data.worldpop.org/'):

    try:
        pycountry.countries.get(alpha_3 = country_code.upper()).name
    except:
        raise ValueError('ISO3 country code is not valid! Please make sure you have entered valid ISO3 country code.')

    worldpop_datasets = pd.read_csv(worldpop_base_url + 'assets/wpgpDatasets.csv')

    assert sum(worldpop_datasets.Covariate.str.contains(str(dataset_year)))>0, 'Worldpop dataset for given does not exist!'
    assert country_code.upper() in worldpop_datasets.ISO3.tolist(), 'Country code does not exist in the worldpop database!'

    dataset_url = worldpop_base_url + worldpop_datasets[(worldpop_datasets.ISO3 == country_code.upper()) & 
                                                            (worldpop_datasets.Covariate == 'ppp_' + str(dataset_year) + ('_UNadj' if un_adjusted else ''))].PathToRaster.values[0]
    if one_km_res:
        dataset_url = dataset_url.split('/')
        dataset_url[5] = dataset_url[5] + '_1km' + ('_UNadj' if un_adjusted else '')
        dataset_url[8] = dataset_url[8].replace(str(dataset_year), str(dataset_year) + '_1km_Aggregated')
        dataset_name = dataset_url[-1]
        dataset_url = '/'.join(dataset_url)
    else:
        dataset_name = dataset_url.split('/')[-1]
    

    dataset_path = os.path.join(data_path, dataset_url.split('/')[-1])

    if not os.path.exists(dataset_path):
        request.urlretrieve(dataset_url, dataset_path)
        print('Dataset download complete!')

    if os.path.exists(dataset_path):
        print('Reading country population tif file...')
        try:
            pop_tif = gdal.Open(dataset_path)
        except:
            raise RuntimeError('Unable to open country population tif file!')
    else:
        print('Country population tif file could not be downloaded! Please download it manually and place it under the data folder.')

    print('Processing raster data...')
    df_pop = tif_to_df(pop_tif)

    print('Data is extracted to pandas dataframe!')

    return df_pop, dataset_name
