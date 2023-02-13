# 2022 Giga

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from giga_toolkit.toolkit import *


class CellTower(GigaTools):

        def __init__(self, 
                    school_filename,
                    country_code,
                    path = os.getcwd(),
                    school_subfoldername = '', 
                    school_id_column_name = 'giga_school_id',
                    ):

            super().__init__(path)
            self.path = path
            self.data_path = os.path.join(path, 'data')
            self.school_filename = school_filename
            self.school_subfoldername = school_subfoldername
            self.school_file_path = os.path.join(self.data_path, 'school', school_subfoldername,  school_filename)
            self.country_code = country_code
            self.school_id_column_name = school_id_column_name
            self.tower_file_path = os.path.join(self.data_path, 'raw', 'opencellid_' + country_code.lower() + '.parquet')
            
        

        def set_tower_data(self, write_data = True):

            try:
                self.tower_data = data_read(self.tower_file_path)
            except:
                self.tower_data = get_opencellid_data(self.country_code, self.data_path, write_data = write_data)
            
            print(f'Tower data is set!')
        

        def get_closest_tower(self):
            
            grid = gsp.GriSPy(self.school_data[['lon', 'lat']].to_numpy(), metric='vincenty')

            rad_types = ['LTE', 'GSM', 'UMTS']
            cell_types = self.tower_data.radio.unique()

            for rad_type in rad_types:
                if rad_type in cell_types:
                    grid = gsp.GriSPy(self.tower_data.loc[self.tower_data.radio == rad_type, ['lon', 'lat']].to_numpy(), metric='vincenty')
                    bubble_dist, bubble_ind = grid.nearest_neighbors(self.school_data[['lon', 'lat']].to_numpy(), n = 1)
                    self.school_data['dist_' + rad_type] = deg2km(bubble_dist, self.earth_r)
            
            dist_cols = [col for col in self.school_data.columns if col.startswith('dist')]
            self.school_data['dist_closest'] = self.school_data[dist_cols].min(axis=1)
            self.school_data['radio_closest'] = self.school_data[dist_cols].idxmin(axis=1).apply(lambda x: x.strip('dist_'))

            print(f'Nearest OpenCelliD towers are mapped to the school locations.')
        

        def run_cell(self, write = False):
        
            self.set_tower_data(write_data=write)
            self.get_closest_tower()

            if write:
                output_filename = f'{self.school_filename[:-4]}_cell_{datetime.date.today().strftime("%m%d%Y")}.csv'
                self.school_data.to_csv(os.path.join(self.data_path, 'output', output_filename), index = True)
                print(f'Data is saved with a name {output_filename}')



class Population(GigaTools):

    def __init__(self, 
            school_filename,
            country_code,
            path = os.getcwd(),
            school_subfoldername = '', 
            school_id_column_name = 'giga_school_id',
            search_range = [5],
            pop_dataset_year = 2020,
            pop_un_adjusted = True,
            pop_one_km_res = False,
            ):

        super().__init__(path)
        self.path = path
        self.data_path = os.path.join(path, 'data')
        self.school_file_path = os.path.join(self.data_path, 'school', school_subfoldername,  school_filename)
        self.school_filename = school_filename
        self.school_subfoldername = school_subfoldername
        self.school_id_column_name = school_id_column_name
        self.country_code = country_code
        self.search_range = search_range
        self.pop_dataset_year = pop_dataset_year
        self.pop_un_adjusted = pop_un_adjusted
        self.pop_one_km_res = pop_one_km_res
    

    def run_population(self, write = False):

        self.population, self.pop_dataset_name = get_population_data(country_code = self.country_code, 
                                                                        dataset_year = self.pop_dataset_year, 
                                                                        data_path = os.path.join(self.data_path, 'raw'), 
                                                                        one_km_res=self.pop_one_km_res, 
                                                                        un_adjusted = self.pop_un_adjusted
                                                                    )

        if type(self.search_range) != list:
            self.search_range = [self.search_range]

        grid = gsp.GriSPy(self.population[['lon', 'lat']].to_numpy(), metric='vincenty')
        upper_radii = km2deg(max(self.search_range) + 1, 6371)
        bubble_dist, bubble_ind = grid.bubble_neighbors(self.school_data[['lon', 'lat']].to_numpy(), distance_upper_bound=upper_radii)

        gdf_pop = df_to_gdf(self.population.loc[list(dict.fromkeys(np.concatenate(bubble_ind)).keys())])
        gdf_pop = gdf_pop.to_crs('epsg:3395')
        
        if self.pop_one_km_res:
            gdf_pop.geometry = gdf_pop.geometry.buffer(.5 * 1000, cap_style = 3)
        else:
            gdf_pop.geometry = gdf_pop.geometry.buffer(.05 * 1000, cap_style = 3)

        gdf_pop['area_pop_tile'] = gdf_pop.area / 10**6

        gdf_school = self.school_data.to_crs('epsg:3395')

        for range_ in self.search_range:
            
            #gdf_school.geometry = gdf_school.buffer(range_ * 1000)
            
            # Performing overlay funcion and join the larger tile
            gdf_joined = gdf_pop.reset_index().overlay(gdf_school.buffer(range_ * 1000).reset_index(), how='intersection')


            # Calculating the areas of the newly-created geometries - i.e. how much of the larger tile that is in the WorldPop tile
            gdf_joined['area_joined'] = gdf_joined.area / 10**6

            
            # Calculating the estimated population inside the larger tile
            gdf_joined[f'pop_{range_}km'] = (gdf_joined['area_joined'] / gdf_joined['area_pop_tile']) * gdf_joined['population']

            self.school_data = self.school_data.join(gdf_joined.groupby('source_id').sum(numeric_only=True)[f'pop_{range_}km'])
        
            
        if write:
            output_filename = f'{self.school_filename[:-4]}_pop_{datetime.date.today().strftime("%m%d%Y")}.csv'
            self.school_data.to_csv(os.path.join(self.data_path, 'output', output_filename), index = True)
            print(f'Data is saved with a name {output_filename}')


class MobileCoverage(GigaTools):
    
    def __init__(self, 
            country_code, 
            school_filename, 
            school_subfoldername='',
            path = os.getcwd(), 
            school_id_column_name='giga_school_id',
            coverage_with_variable_signal_strength = False,
            coverage_col_name = '4G_coverage'):
        
        super().__init__(path)
        self.path = path
        self.data_path = os.path.join(path, 'data')
        self.school_filename = school_filename
        self.school_subfoldername = school_subfoldername
        self.school_file_path = os.path.join(self.data_path, 'school', school_subfoldername,  school_filename)
        self.country_code = country_code
        self.school_id_column_name = school_id_column_name
        self.coverage_with_variable_signal_strength = coverage_with_variable_signal_strength
        self.coverage_col_name = coverage_col_name

        try:
            country_alpha2 = pycountry.countries.get(alpha_3 = self.country_code.upper()).alpha_2
        except:
            raise ValueError('ISO3 country code is not valid! Please make sure you have entered valid ISO3 country code.')
        
        self.mc_filename = f'MCE_{country_alpha2.upper()}4G_2020.tif'
        self.mc_file_path = os.path.join(self.path, 'data', 'raw', 'MCE_4G', self.mc_filename)
    

    def run_coverage(self, write = False):

        mc_tif = gdal.Open(self.mc_file_path)

        df_mc, res = tif_to_df(mc_tif)

        # filter coverage data with coverage level
        # 1: strong signal strength, 2: variable signal strength, 3: no signal
        if self.coverage_with_variable_signal_strength:
            df_mc = df_mc[df_mc.value!=3]
        else:
            df_mc = df_mc[df_mc.value==1]

        # transform point data to polygon geodataframe
        gdf_mc_covered = df_to_gdf(df_mc, crs = 'epsg:3857')
        gdf_mc_covered.geometry = gdf_mc_covered.geometry.buffer(res, cap_style= 3)
        gdf_mc_covered = gdf_mc_covered.to_crs('epsg:4326')

        # spatial join school data to MC data
        self.school_data = self.school_data.sjoin(gdf_mc_covered[['geometry']], how='left', predicate = 'intersects').rename(columns = {'index_right': self.coverage_col_name})
        self.school_data = self.school_data[~self.school_data.index.duplicated(keep='first')]
        self.school_data[self.coverage_col_name] = self.school_data[self.coverage_col_name].apply(lambda x: 1 if x>0 else 0)

        if write:
            output_filename = f'{self.school_filename[:-4]}_mc_{datetime.date.today().strftime("%m%d%Y")}.csv'
            self.school_data.to_csv(os.path.join(self.data_path, 'output', output_filename), index = True)
            print(f'Data is saved with a name {output_filename}')