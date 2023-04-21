# 2022 Giga

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from scipy.spatial import cKDTree
import pycountry
import gdal
from urllib import request
import pandas as pd
import numpy as np
import os

from giga_toolkit.utils import data_read, setup_logging, df_to_gdf, deg2km, km2deg, tif_to_df, get_opencellid_data
from giga_toolkit.toolkit import GigaTools


class Mapping(GigaTools):

    def __init__(self,
                 school_filename,
                 country_code,
                 path = os.getcwd(),
                 school_subfoldername = '',
                 school_id_column_name = 'giga_id_school',
                 tower_filename = None,
                 opencellid_token = None,
                 pop_radius = [3],
                 pop_dataset_year = 2020,
                 pop_un_adjusted = True,
                 pop_one_km_res = False,
                 coverage_foldername = 'MCE_4G',
                 accept_variable_signal_strength = False,
                 ):
        super().__init__(path)
        self.path = path
        self.data_path = os.path.join(path, 'data')

        self.country_code = country_code

        self.school_filename = school_filename
        self.school_subfoldername = school_subfoldername
        self.school_file_path = os.path.join(self.data_path, 'school', school_subfoldername,  school_filename)
        self.school_id_column_name = school_id_column_name
        
        self.tower_filename = tower_filename

        self.pop_radius = pop_radius
        self.pop_dataset_year = pop_dataset_year
        self.pop_un_adjusted = pop_un_adjusted
        self.pop_one_km_res = pop_one_km_res

        self.coverage_foldername = coverage_foldername
        self.accept_variable_signal_strength = accept_variable_signal_strength
        
        self.logger = setup_logging('mapping_logger')
        self.logger.info('New mapping object created.')

        try:
            self.country_alpha2 = pycountry.countries.get(alpha_3 = self.country_code.upper()).alpha_2
        except:
            raise ValueError('ISO3 country code is not valid! Please make sure you have entered valid ISO3 country code.')


        if opencellid_token is not None:

            try:
                self.opencellid_token = open(os.path.join(path, 'assets', 'opencellid.token')).read()
            except:
                self.opencellid_token = None
                self.logger.warn('OpenCellId token is not provided. Tower data will be read from data folder!')
            

    ### CELL TOWER

    def set_tower_data(self, write_data = True):

        if self.tower_filename is not None:
            self.logger.info('Tower data is being read from data folder!')
            self.tower_data = data_read(os.path.join(self.data_path, 'raw', self.tower_filename))
        else:
            self.logger.info('Downloading tower data from OpenCellId server!')
            self.tower_data = get_opencellid_data(self.country_code, self.opencellid_token, self.data_path, 'raw', write_data=write_data)
        
        self.logger.info('Tower data is set!')

    def get_closest_towers(self):

        self.logger.info('Locating closest towers of each type to the schools...')
        
        rad_types = ['LTE', 'GSM', 'UMTS']
        cell_types = self.tower_data.radio.unique()

        for rad_type in rad_types:
            if rad_type in cell_types:
                kdtree = cKDTree(self.tower_data.loc[self.tower_data.radio == rad_type, ['lon', 'lat']].to_numpy())
                dist, ind = kdtree.query(self.school_data[['lon', 'lat']].to_numpy(), k = 1)
                self.school_data['dist_' + rad_type] = deg2km(dist)
                self.logger.info(f'Closest {rad_type} towers are located!')
        
        dist_cols = [col for col in self.school_data.columns if col.startswith('dist')]
        self.school_data['closest_tower_distance'] = self.school_data[dist_cols].min(axis=1)
        self.school_data['closest_radio_type'] = self.school_data[dist_cols].idxmin(axis=1).apply(lambda x: x.strip('dist_'))


    ### POPULATION

    @staticmethod
    def get_population_data(country_code, dataset_year, path, one_km_res = False, un_adjusted = True, worldpop_base_url = 'https://data.worldpop.org/'):

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
        

        dataset_path = os.path.join(path, dataset_url.split('/')[-1])

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
        df_pop, res = tif_to_df(pop_tif)
        df_pop.rename(columns ={'value': 'population'}, inplace=True)

        print('Data is extracted to pandas dataframe!')

        return df_pop, dataset_name


    def set_population_data(self):
        
        self.pop_data, self.pop_dataset_name = Mapping.get_population_data(country_code = self.country_code, 
                                                                           dataset_year = self.pop_dataset_year, 
                                                                           path = os.path.join(self.data_path, 'raw'), 
                                                                           one_km_res=self.pop_one_km_res, 
                                                                           un_adjusted = self.pop_un_adjusted
        )
    

    def bubble_population(self, radius):

        self.logger.info('Filtering population data pixels around schools...')

        pop_clean = self.pop_data.loc[self.pop_data.population>0].reset_index(drop=True)

        kdtree = cKDTree(pop_clean[['lon', 'lat']].to_numpy())
        
        radius = km2deg(radius+1)
        
        pop_neighbor = kdtree.query_ball_point(self.school_data[['lon', 'lat']].to_numpy(), r = km2deg(radius+1))

        return pop_clean.loc[set(np.concatenate(pop_neighbor))]


    def prepare_population(self, df_pop, crs = 'epsg:4326'):
        
        gdf_pop = df_to_gdf(df_pop)
        gdf_pop = gdf_pop.to_crs(crs)

        if self.pop_one_km_res:
            gdf_pop['geometry'] = gdf_pop.geometry.buffer(.5 * 1000, cap_style = 3)
        else:
            gdf_pop['geometry'] = gdf_pop.geometry.buffer(.05 * 1000, cap_style = 3)
        
        gdf_pop['area_pop_tile'] = gdf_pop.area / 10**6

        return gdf_pop
    

    def overlay_population(self, radius):

        if type(radius) != list:
            radius = [radius]

        df_pop = self.bubble_population(max(radius))

        gdf_pop = self.prepare_population(df_pop, crs = 'epsg:3395')

        for radius_ in radius:

            self.logger.info(f'Overlaying population data for {radius_}km radius around schools...')

            # Performing overlay funcion and join the larger tile
            gdf_overlayed = gdf_pop.reset_index().overlay(self.school_data.to_crs('epsg:3395').buffer(radius_ * 1000).reset_index(), how='intersection')
            
            # Calculating the areas of the newly-created geometries - i.e. how much of the larger tile that is in the WorldPop tile
            gdf_overlayed['area_joined'] = gdf_overlayed.area / 10**6
            
            # Calculating the estimated population inside the larger tile
            gdf_overlayed[f'pop_{radius_}km'] = (gdf_overlayed['area_joined'] / gdf_overlayed['area_pop_tile']) * gdf_overlayed['population']

            self.school_data = self.school_data.join(gdf_overlayed.groupby(self.school_id_column_name).sum(numeric_only=True)[f'pop_{radius_}km'])
            
            self.logger.info(f'pop_{radius_}km attribute added to the school data!')
    

    #### COVERAGE

    def set_coverage_data(self):

        self.logger.info('Reading country coverage data...')
        mc_filename = f'MCE_{self.country_alpha2.upper()}4G_2020.tif'
        mc_file_path = os.path.join(self.data_path, 'raw', self.coverage_foldername, mc_filename)
    
        mc_tif = gdal.Open(mc_file_path)

        df_mc, res = tif_to_df(mc_tif)

        # filter coverage data with coverage level
        # 1: strong signal strength, 2: variable signal strength, 3: no signal
        if self.accept_variable_signal_strength:
            df_mc = df_mc[df_mc.value!=3]
        else:
            df_mc = df_mc[df_mc.value==1]

        # transform point data to polygon geodataframe
        gdf_mc_covered = df_to_gdf(df_mc, crs = 'epsg:3857')
        gdf_mc_covered.geometry = gdf_mc_covered.geometry.buffer(res, cap_style= 3)
        self.coverage_data = gdf_mc_covered.to_crs('epsg:4326')
        self.logger.info('Coverage data property is set!')
    
    def overlay_coverage(self):
        
        self.logger.info('Overlaying coverage data on schools...')
        # spatial join school data to MC data
        self.school_data = self.school_data.sjoin(self.coverage_data[['geometry']], how='left', predicate = 'intersects').rename(columns = {'index_right': 'mobile_coverage'})
        self.school_data = self.school_data[~self.school_data.index.duplicated(keep='first')]
        self.school_data['mobile_coverage'] = self.school_data['mobile_coverage'].apply(lambda x: 1 if x>0 else 0)
        self.logger.info('Mobile coverage attribute added to the school data!')

    
    def run_mapping(self, celltower = True, population = True, coverage = True):
        
        self.set_school_data()

        if celltower:
            if self.tower_filename is not None or self.opencellid_token is not None:
                self.set_tower_data()
                self.get_closest_towers()
            else:
                self.logger.info('Cell tower mapping will not run. Either tower filename or opencellid token should be provided to run celltower mapping!')
        
        if population:
            self.set_population_data()
            self.overlay_population(radius = self.pop_radius)
        
        if coverage:
            self.set_coverage_data()
            self.overlay_coverage()
