# Copyright 2022 Giga

# Your use of this software is subject to the Giga Principles and
# Policies. This copyright notice shall be included in all copies
# or substantial portions of the software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from giga_toolkit.toolkit import *

class Visiblity(GigaTools):

    def __init__(self, 
                    school_filename,
                    tower_filename, 
                    path = os.getcwd(), 
                    school_subfoldername = '', 
                    school_id_column_name = 'giga_school_id',
                    tower_id_column_name = 'source_id',
                    srtm_dict_filename = 'srtm30m_bounding_boxes.json',
                    srtm_folder_name = 'srtm1',
                    avg_school_height = 15,
                    max_tower_reach = 35,
                    n_visible = 1,
                    avg_tower_height = 0,
                    los_correction = 0,
                    country_code = '',
                    n_clusters = 1
                    ):
        
        super().__init__(path)
        self.path = path
        self.data_path = os.path.join(path, 'data')
        self.school_file_path = os.path.join(self.data_path, 'school', school_subfoldername,  school_filename)
        self.school_filename = school_filename
        self.tower_filename = tower_filename
        self.school_subfoldername = school_subfoldername
        self.tower_id_column_name = tower_id_column_name
        self.tower_file_path = os.path.join(self.data_path, 'raw', tower_filename)
        self.school_id_column_name = school_id_column_name
        self.srmt_dict = gp.read_file(os.path.join(path, 'assets', srtm_dict_filename))
        self.srtm_folder_path = os.path.join(self.data_path, srtm_folder_name)
        self.avg_school_height = avg_school_height
        self.max_tower_reach = max_tower_reach
        self.n_visible = n_visible
        self.avg_tower_height = avg_tower_height
        self.los_correction = los_correction
        self.country_code = country_code
        self.n_clusters = n_clusters

        assert os.path.exists(os.path.join(self.path, 'assets', 'earthdata_account.txt')), 'Please provide EarthData account details in a text file named earthdata_account.txt under assets folder!'
    

    def set_tower_data(self):
        
        assert os.path.exists(self.tower_file_path), 'Please make sure you specify the correct file name to the tower data.'

        tower = data_read(self.tower_file_path)

        if self.tower_id_column_name not in tower.columns: 
            print(tower.columns)
            raise ValueError('Given tower id column is not in the tower dataset. Please initialize tower_id_column_name parameter with one of the above!')
        

        tower.set_index(self.tower_id_column_name, inplace=True)

        assert tower.index.duplicated().sum() == 0, 'Duplicate ids exist in the tower data! Please make sure each row has unique id and re-run.'
        
        if type(tower) != gp.geodataframe.GeoDataFrame:
            tower = df_to_gdf(tower, rename=True)
        else:
            rename_geo_cols(tower)

        if 'height' not in tower.columns:
            tower['height'] = self.avg_tower_height
            print(f'Column "heigh" is not in the dataset. Therefore, "height" column is initialized with avg_tower_height which is set as {self.avg_tower_height}!')
        
        self.tower_data = tower.loc[['lat', 'lon', 'height', 'geometry']]
    

    def download_matching_srtm_tiles(self):

        f = open(os.path.join(self.path, 'assets', 'earthdata_account.txt'))
        lines = f.readlines()
        username = deobfuscate(lines[0].strip())
        password = deobfuscate(lines[1].strip())
        f.close()

        print('Locating SRTM tiles...')

        all_loc = pd.concat([self.school_data[['geometry']], self.tower_data[['geometry']]])
        all_loc.geometry = all_loc.geometry.buffer(km2deg(self.max_tower_reach, self.earth_r))
        area_srtm_files = all_loc.sjoin(self.srtm_dict, how='left', predicate='intersects')

        print('In total ' + str(len(area_srtm_files.dataFile.unique())) + ' SRTM tiles are matched to the school and tower locations.')

        self.unmatched_locations = area_srtm_files[area_srtm_files.dataFile.isnull()]

        print('Downloading matched SRTM tiles...')
        for file in area_srtm_files.dataFile.unique():
            file_path = os.path.join(self.srtm_folder_path, file)
            if not os.path.exists(file_path):
                download_srtm_data(username, password, self.srtm_base_url + file, file_path)
        
        print('SRTM data collection is complete!')
    

    def bubble_towers(self, bubble_r, metric = 'vincenty'):

        print(f'Implementing bubble search with the radius of {bubble_r} km...')

        grid = gsp.GriSPy(self.tower_data[['lon', 'lat']].to_numpy(), metric = metric)
        search_r = km2deg(bubble_r, self.earth_r)
        bubble_dist, bubble_ind = grid.bubble_neighbors(self.school_data[['lon', 'lat']].to_numpy(), sorted = True, distance_upper_bound=search_r)

        print('Bubble search is implemented!')

        return bubble_dist, bubble_ind
    

    def get_service_towers(self, towers, bubble_dist, bubble_ind, school_pos):
        
        tower_match = pd.DataFrame(zip(bubble_ind[school_pos], bubble_dist[school_pos]), columns = ['tower_pos', 'dist'])
        
        if len(tower_match) != 0:
            tower_match.reset_index(drop=True, inplace=True)
            tower_match[['lat', 'lon', 'height']] = tower_match.tower_pos.apply(lambda x: towers.iloc[x][['lat', 'lon', 'height']])
            tower_match['idx'] = tower_match.tower_pos.apply(lambda x: towers.iloc[x].name)
            tower_match['dist_km'] = tower_match.dist.apply(lambda x: deg2km(x, self.earth_r))
        
        return tower_match

    
    def get_visibility(self):

        self.download_matching_srtm_tiles()

        if len(self.unmatched_locations) != 0:
            print(f'# of unmatched locations: {len(self.unmatched_locations)}')
            print('The unmatched school/tower geo locations are not valid and will be discarded from the dataset(s). Discarded locations are kept in "unmatched_locations" attribute.')

        school_unmatched, tower_unmatched = [a for a in self.unmatched_locations.index if a in self.school_data.index], [a for a in self.unmatched_locations.index if a in self.tower_data.index]
        self.school_data.drop(index = school_unmatched, inplace = True)
        self.tower_data.drop(index = tower_unmatched, inplace = True)

        self.school_data['within_tower_reach'] = False
        
        srtm1_data = Srtm1HeightMapCollection(auto_build_index=True, hgt_dir=Path(self.srtm_folder_path))
        
        bubble_dist, bubble_ind = self.bubble_towers(bubble_r = self.max_tower_reach)

        print('Running visibility test for all schools...')
        self.n_checks = 0

        for school in tqdm(self.school_data.reset_index().itertuples()):
            school_pos = school.Index
            school_idx = self.school_data.index[school_pos]
            tower_match = self.get_service_towers(self.tower_data, bubble_dist, bubble_ind, school_pos)
            visible_count = 0

            if len(tower_match) == 0:
                continue

            for twr in tower_match.itertuples():
                self.n_checks += 1
                e_profile, d_profile = zip(*[(i.elevation, i.distance) for i in srtm1_data.get_elevation_profile(school.lat, school.lon, twr.lat, twr.lon)])
                df_elev = pd.DataFrame(zip(np.linspace(e_profile[0] + self.avg_school_height, e_profile[-1] + twr.height, len(e_profile)), e_profile, d_profile), columns = ['los', 'dep', 'dist'])
                df_elev['dif'] = df_elev.los - df_elev.dep
                t_visible = np.all(df_elev.dif > -self.los_correction)
                visible_count += t_visible
                if t_visible:
                    self.school_data.loc[school_idx, ['tower_' + str(visible_count), 
                        'tower_' + str(visible_count) + '_lat', 
                        'tower_' + str(visible_count) + '_lon', 
                        'tower_' + str(visible_count) + '_dist',
                        'tower_' + str(visible_count) + '_los_geometry']] = twr.idx, twr.lat, twr.lon, twr.dist_km, LineString([self.tower_data.loc[twr.idx, 'geometry'], school.geometry])
                #tower_match.at[twr.Index,'visible'] = t_visible
                if visible_count == self.n_visible:
                    break
                
            if visible_count > 0:
                self.school_data.at[school_idx, 'within_tower_reach'] = True
            
        print('Visibility check is complete; school_data attribute is updated with the results!')
        
        print(f'Average # of checks per school: {self.n_checks/len(self.school_data)}')
    

    def run_pipeline(self, write = True):

        self.set_tower_data()

        if self.n_clusters > 1:

            self.cluster_locations(self.n_clusters)

            for cluster in self.cluster:

                self.school_data = self.cluster[cluster]
                print(f'Cluster {cluster} data is set!')

                self.get_visibility()

                if write:
                    output_filename = f'{self.school_filename[:-4]}_visibility_cluster_{cluster}_{datetime.date.today().strftime("%m%d%Y")}.xlsx'
                    self.school_data.to_csv(os.path.join(self.data_path, 'output', output_filename), index = True)
                    print(f'Data is saved with a name {output_filename}')
        
        else:

            self.get_visibility()

            if write:
                output_filename = f'{self.school_filename[:-4]}_visibility_cluster_{datetime.date.today().strftime("%m%d%Y")}.xlsx'
                self.school_data.to_csv(os.path.join(self.data_path, 'output', output_filename), index = True)
                print(f'Data is saved with a name {output_filename}')