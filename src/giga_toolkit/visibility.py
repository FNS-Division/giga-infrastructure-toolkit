# 2022 Giga

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from giga_toolkit.toolkit import GigaTools
from giga_toolkit.utils import *
import http.cookiejar as cookielib
from scipy.spatial import cKDTree
from srtm.height_map_collection import Srtm1HeightMapCollection
from shapely.geometry import LineString

class Visibility(GigaTools):

    """
    A module for computing visibility between schools and telecommunication towers.
    """

    def __init__(self, 
                    school_filename,
                    tower_filename, 
                    path = os.getcwd(), 
                    school_subfoldername = '', 
                    school_id_column_name = 'poi_id',
                    tower_id_column_name = 'ict_id',
                    srtm_folder_name = 'srtm1',
                    earthdata_username = '',
                    earthdata_password = '',
                    earthdata_account_file = 'earthdata_account.txt',
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
        self.srtm_folder_path = os.path.join(self.data_path, srtm_folder_name)
        self.earthdata_username = earthdata_username
        self.earthdata_password = earthdata_password
        self.earthdata_account_file = earthdata_account_file
        self.avg_school_height = avg_school_height
        self.max_tower_reach = max_tower_reach
        self.n_visible = n_visible
        self.avg_tower_height = avg_tower_height
        self.los_correction = los_correction
        self.country_code = country_code
        self.n_clusters = n_clusters
        self.logger = setup_logging('visibility_logger')
        

    

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
            self.logger.warn(f'Column "height" is not in the dataset. Therefore, "height" column is initialized with avg_tower_height which is set as {self.avg_tower_height}!')
        
        self.tower_data = tower.loc[:, ['lat', 'lon', 'height', 'geometry']]

    
    @staticmethod
    def get_srtm_dict(dict_url = 'https://dwtkns.com/srtm30m/srtm30m_bounding_boxes.json'):
        req = request.Request(dict_url)
        req.add_header('User-Agent', 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:77.0) Gecko/20100101 Firefox/77.0')
        content = request.urlopen(req)
        if content.status == 200:
            return gp.read_file(content)
        else:
            raise RuntimeError(f'SRTM dictionary cannot be read from following url: {dict_url}')
    

    @staticmethod
    def locate_srtm_tiles(df, srtm_dict):
    
        # left spatial join srtm dictionary to school and tower locations
        matched_tiles = df.sjoin(srtm_dict, how='left', predicate='intersects')

        unmatched_locations = matched_tiles[matched_tiles.dataFile.isnull()]

        return matched_tiles, unmatched_locations
    
    @staticmethod
    def srtm_directory_check(srtm_folder_path, srtm_tiles):

        srtm_files_to_download = set()
        
        for file in srtm_tiles.dataFile.unique():
            file_path = os.path.join(srtm_folder_path, file)
            if not os.path.exists(file_path):
                srtm_files_to_download.add(file_path)

        return srtm_files_to_download

    
    @staticmethod
    def download_srtm_tile(username, password, url, path):

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
    


    def download_matching_srtm_tiles(self):

        # concatenate school and tower data
        all_loc = pd.concat([self.school_data[['geometry']], self.tower_data[['geometry']]])
        all_loc['geometry'] = all_loc.geometry.buffer(km2deg(self.max_tower_reach))
        
        # get SRTM dictionary
        srtm_dict = Visibility.get_srtm_dict()


        # locate srtm tiles
        self.logger.info('Locating SRTM tiles...')        
        matched_tiles, unmatched_locations = Visibility.locate_srtm_tiles(all_loc, srtm_dict)
        matched_tiles.dropna(subset='dataFile',  inplace=True)
        self.logger.info('In total ' + str(len(matched_tiles.dataFile.unique())) + ' SRTM tiles are matched to the school and tower locations.')

        # drop unmatched schools and towers from respective datasets
        if len(unmatched_locations) != 0:
            self.logger.info(f'# of unmatched locations: {len(unmatched_locations)}')
            self.logger.warn('The unmatched school/tower geo locations are not valid and will be discarded from the dataset(s). Discarded locations are kept in "school_unmatched" and "tower_unmatched" attribute.')

        self.school_unmatched, self.tower_unmatched = [a for a in unmatched_locations.index if a in self.school_data.index], [a for a in unmatched_locations.index if a in self.tower_data.index]
        self.school_data.drop(index = self.school_unmatched, inplace = True)
        self.tower_data.drop(index = self.tower_unmatched, inplace = True)

        srtm_files_to_download = Visibility.srtm_directory_check(self.srtm_folder_path, matched_tiles)

        if len(srtm_files_to_download) > 0:

            self.logger.info('Initializing EarthData credentials...')
            if len(self.earthdata_account_id) == 0 or len(self.earthdata_pwd) == 0:
                assert os.path.exists(os.path.join(self.path, 'assets', self.earthdata_account_file)), f'Please provide EarthData account details in a text file named {self.earthdata_account_file} under assets folder!'

                # open file containing EarthData username and password
                f = open(os.path.join(self.path, 'assets', self.earthdata_account_file))
                lines = f.readlines()
                username = deobfuscate(lines[0].strip())
                password = deobfuscate(lines[1].strip())
                f.close()
            else:
                username = self.earthdata_username
                password = self.earthdata_password

            self.logger.info('Downloading matched SRTM tiles...')
            for file_path in srtm_files_to_download:
                Visibility.download_srtm_tile(username, password, self.srtm_base_url + file_path.split('/')[-1], file_path)

        
        self.logger.info('SRTM data collection is complete!')
    
    @staticmethod
    def bubble_towers(kdtree_of_towers :cKDTree, query_instance, radius):
        """
        Finds towers within a specified radius of a query point using a KDTree.

        Args:
            kdtree_of_towers (scipy.spatial.cKDTree): A KDTree of tower coordinates.
            query_instance (numpy.ndarray): The query point.
            radius (float): The radius of the bubble around the query point, in kilometers.

        Returns:
            Tuple: A tuple containing the indices of the towers within the bubble, and their distances from the query point in kilometers.
        """
        
        # Convert radius from kilometers to degrees.
        radius = km2deg(radius)
        
        # Get the indices of the towers within the specified radius.
        neighbors = kdtree_of_towers.query_ball_point(query_instance, r=radius)

        # Get the distances of the towers within the specified radius from the query point.
        dist, ind = kdtree_of_towers.query(query_instance, len(neighbors))

        return ind, deg2km(dist)

    @staticmethod
    def check_visibility(srtm1_data: Srtm1HeightMapCollection, lat1, lon1, height1, lat2, lon2, height2, data = False):

        """
        Calculates the line of sight visibility between two points using SRTM data.

        Args:
            srtm1_data (Srtm1HeightMapCollection): An SRTM1 height map collection.
            lat1 (float): The latitude of the first point.
            lon1 (float): The longitude of the first point.
            height1 (float): The height of the first point.
            lat2 (float): The latitude of the second point.
            lon2 (float): The longitude of the second point.
            height2 (float): The height of the second point.
            data (bool): Whether to return a Pandas DataFrame with additional information.

        Returns:
            bool or DataFrame: Whether there is line of sight visibility between the two points. If data is True, a Pandas DataFrame with additional information is returned.
        """

        # get elevation and distance profiles
        e_profile, d_profile = zip(*[(i.elevation, i.distance) for i in srtm1_data.get_elevation_profile(lat1, lon1 , lat2, lon2)])

        # calculate line of sight profile
        los_profile = np.linspace(e_profile[0] + height1, e_profile[-1] + height2, len(e_profile))
        has_line_of_sight = np.all(los_profile >= e_profile)

        if data:
            # return line of sight profile as Pandas DataFrame
            return pd.DataFrame(zip(los_profile, e_profile, d_profile), columns = ['los_profile', 'elevation_profile', 'distance'])
        
        return has_line_of_sight
    
    
    def get_visibility(self):

        self.download_matching_srtm_tiles()

        height_cols = [col for col in self.school_data.columns if 'height' in str(col).lower()]

        if len(height_cols) > 1:
            raise RuntimeError('There are more than one column in the school dataset indicating the school building height! Make sure there is only one height column.')
        elif len(height_cols) ==1:
            self.school_data.rename(columns={height_cols[0]: 'height'}, inplace = True)
        else:
            self.school_data['height'] = self.avg_school_height
        
        srtm1_data = Srtm1HeightMapCollection(auto_build_index=True, hgt_dir=Path(self.srtm_folder_path))
        
        self.n_checks = 0

        kdtree = cKDTree(self.tower_data[['lon', 'lat']].to_numpy())

        # Create a dictionary to store the visibility information for each school
        visibility_dict = dict.fromkeys(self.school_data.index)

        for school in self.school_data.itertuples():
            self.logger.debug(f"Calculating visibility for school {school.Index}")
            
            # find towers within maximum tower reach
            neighbors, dist_km = Visibility.bubble_towers(kdtree, np.array([school.lon, school.lat]), self.max_tower_reach)

            if len(neighbors) == 0:
                continue

            # initialize visibility dictionary for current school
            visible_count = 0
            visibility_dict.update({school.Index:
                                    dict(is_visible = False)
                                    })

            # iterate over towers within maximum tower reach and check visibility
            tower_match = self.tower_data.iloc[neighbors].copy()
            tower_match['dist_km'] = dist_km
            for twr in tower_match.itertuples():
                self.n_checks += 1
                has_line_of_sight = Visibility.check_visibility(srtm1_data, school.lat, school.lon, self.avg_school_height, twr.lat, twr.lon, twr.height)
                visible_count += has_line_of_sight
                
                if has_line_of_sight:
                    # add tower information to visibility dictionary for current school
                    twr_idx = 'tower_' + str(visible_count)
                    visibility_dict[school.Index].update({
                        twr_idx: twr.Index,
                        twr_idx + '_lat': twr.lat,
                        twr_idx + '_lon': twr.lon,
                        twr_idx + '_dist': twr.dist_km,
                        twr_idx + '_los_geom': LineString([twr.geometry, school.geometry])
                    })
                
                if visible_count == self.n_visible:
                    break
                
            visibility_dict[school.Index].update(is_visible = visible_count>0)
            
        self.logger.info('Visibility check is complete!')
        
        self.logger.info(f'Average # of checks per school: {self.n_checks/len(self.school_data)}')

        return visibility_dict
    
    def write_visibility_data(self, vis_dict, filename, flex_format = False):
        if flex_format:
            output = pd.DataFrame(vis_dict.items(), vis_dict.keys(), columns =['id', 'visibility']).drop(columns='id', inplace=True)
        else:
            output = pd.DataFrame(vis_dict.values(), vis_dict.keys())
        output.to_csv(os.path.join(self.data_path, 'output', filename), index= True)
        self.logger.info(f'Data is saved with a name {filename}')

    def run_visibility(self):
        
        # set school data
        self.set_school_data()

        # set tower data
        self.set_tower_data()

        if self.n_clusters > 1:

            self.cluster_locations(self.n_clusters)

            output = dict()

            for cluster in self.cluster:

                self.school_data = self.cluster[cluster]
                print(f'Cluster {cluster} data is set!')

                output.update(self.get_visibility())
            
            return output
        
        else:

            return self.get_visibility()
