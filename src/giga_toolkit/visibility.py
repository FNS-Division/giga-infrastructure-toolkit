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
from typing import Dict

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
                    avg_school_height: float = 15,
                    max_tower_reach: float = 35,
                    n_visible: int = 3,
                    avg_tower_height = 0,
                    los_correction = 0,
                    country_code = '',
                    n_clusters = 1
                    ):
        """
        Initializes a new instance of the Visibility class.
        
        Args:
        school_filename (str): A filename to the school data which includes columns for latitude, longitude, 
                                    and building height.
        tower_filename (str): A filename to the tower data which includes columns for latitude, longitude, and 
                                   tower height.
        srtm_folder_name (str or Path): The name to the folder containing SRTM elevation data.
        max_tower_reach (float): The maximum distance (in kilometers) from a school to a cell phone tower to be considered
                                 in the analysis.
        n_visible (int, optional): The minimum number of cell phone towers that must be visible from a school in order for 
                                   it to be considered "covered". Default is 3.
        avg_school_height (float, optional): The average height of school buildings, in meters. Default is 15.
        """
        
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
        self.logger.info('New visibility object created.')
        

    

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
        """
        Retrieve the SRTM dictionary file from the provided url.

        Args:
            dict_url (str): The url of the SRTM dictionary file.

        Returns:
            A GeoDataFrame containing the SRTM dictionary information.

        Raises:
            RuntimeError: If the SRTM dictionary cannot be read from the provided url.
        """
        # Create the request object with user agent header
        req = request.Request(dict_url)
        req.add_header('User-Agent', 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:77.0) Gecko/20100101 Firefox/77.0')
        content = request.urlopen(req)
        # Read the content if status code is 200
        if content.status == 200:
            return gp.read_file(content)
        else:
            raise RuntimeError(f'SRTM dictionary cannot be read from following url: {dict_url}')
    

    @staticmethod
    def locate_srtm_tiles(df, srtm_dict):
        """
        Spatially join the SRTM dictionary with school and tower locations.

        Args:
            df (pandas.DataFrame): The DataFrame containing school and tower locations.
            srtm_dict (geopandas.GeoDataFrame): The GeoDataFrame containing SRTM dictionary information.

        Returns:
            A tuple of two GeoDataFrames - matched_tiles and unmatched_locations.
            matched_tiles contains the rows of srtm_dict that intersect with df.
            unmatched_locations contains the rows of df that don't intersect with srtm_dict.
        """
        # Perform spatial join using 'intersects' predicate
        matched_tiles = df.sjoin(srtm_dict, how='left', predicate='intersects')
        # Find the rows with null values for dataFile
        unmatched_locations = matched_tiles[matched_tiles.dataFile.isnull()]

        return matched_tiles, unmatched_locations
    
    @staticmethod
    def srtm_directory_check(srtm_folder_path, srtm_tiles):
        """
        Check if SRTM files corresponding to srtm_tiles exist in the provided directory.

        Args:
            srtm_folder_path (str): The path of the directory where SRTM files are stored.
            srtm_tiles (geopandas.GeoDataFrame): The GeoDataFrame containing the SRTM tiles to check.

        Returns:
            A set containing the file paths of SRTM files that need to be downloaded.
        """
        # Initialize an empty set to store file paths of SRTM files to download
        srtm_files_to_download = set()

        # Check if each file in srtm_tiles exists in the directory
        for file in srtm_tiles.dataFile.unique():
            file_path = os.path.join(srtm_folder_path, file)
            # If the file doesn't exist, add it to the set
            if not os.path.exists(file_path):
                srtm_files_to_download.add(file_path)

        return srtm_files_to_download

    
    @staticmethod
    def download_srtm_tile(username, password, url, path):
        """
        Downloads a SRTM tile from the specified url and saves it to the specified path.

        Args:
            username (str): The Earthdata username.
            password (str): The Earthdata password.
            url (str): The url of the SRTM tile to download.
            path (str): The path where the SRTM tile should be saved.

        Returns:
            None
        """
        # Create password manager to handle authentication
        password_manager = request.HTTPPasswordMgrWithDefaultRealm()
        password_manager.add_password(None, "https://urs.earthdata.nasa.gov", username, password)

        # Create cookie jar to handle cookies
        cookie_jar = cookielib.CookieJar()

        # Install all the handlers.
        opener = request.build_opener(
            request.HTTPBasicAuthHandler(password_manager),
            #request.HTTPHandler(debuglevel=1),    # Uncomment these two lines to see
            #request.HTTPSHandler(debuglevel=1),   # details of the requests/responses
            request.HTTPCookieProcessor(cookie_jar))
        request.install_opener(opener)

        # Retrieve the file from url
        request.urlretrieve(url, path)
    


    def download_matching_srtm_tiles(self):

        """
        Downloads SRTM tiles matching school and tower locations, and discards unmatched locations.
        
        Parameters:
        -----------
        None
        
        Returns:
        --------
        None
        """
        
        # concatenate school and tower data
        all_loc = pd.concat([self.school_data[['geometry']], self.tower_data[['geometry']]])

        # buffer the locations by max tower reach and convert to degrees
        all_loc['geometry'] = all_loc.geometry.buffer(km2deg(self.max_tower_reach))
        
        # get SRTM dictionary
        srtm_dict = Visibility.get_srtm_dict()


        # locate srtm tiles
        self.logger.info('Locating SRTM tiles...')        
        matched_tiles, unmatched_locations = Visibility.locate_srtm_tiles(all_loc, srtm_dict)

        # drop tiles that don't match any location
        matched_tiles.dropna(subset='dataFile',  inplace=True)
        self.logger.info('In total ' + str(len(matched_tiles.dataFile.unique())) + ' SRTM tiles are matched to the school and tower locations.')

        # drop unmatched schools and towers from respective datasets
        if len(unmatched_locations) != 0:
            self.logger.info(f'# of unmatched locations: {len(unmatched_locations)}')
            self.logger.warn('The unmatched school/tower geo locations are not valid and will be discarded from the dataset(s). Discarded locations are kept in "school_unmatched" and "tower_unmatched" attribute.')

        # get indices of unmatched schools and towers
        self.school_unmatched, self.tower_unmatched = [a for a in unmatched_locations.index if a in self.school_data.index], [a for a in unmatched_locations.index if a in self.tower_data.index]
        
        # drop unmatched schools and towers from respective datasets
        self.school_data.drop(index = self.school_unmatched, inplace = True)
        self.tower_data.drop(index = self.tower_unmatched, inplace = True)

        # get list of SRTM tiles that need to be downloaded
        srtm_files_to_download = Visibility.srtm_directory_check(self.srtm_folder_path, matched_tiles)

        if len(srtm_files_to_download) > 0:
            self.logger.info('Initializing EarthData credentials...')

            # if account details are not provided, read them from file
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

            # download each SRTM tile
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
    

    @staticmethod
    def calculate_three_dimension_haversine(srtm1_data: Srtm1HeightMapCollection, lat1, lon1, height1, lat2, lon2, height2):
        """
        Calculate the three-dimensional Haversine distance between two points in latitude, longitude, and altitude.
        
        Args:
        - srtm1_data: Srtm1HeightMapCollection - SRTM1 data for the area
        - lat1: float - latitude of point 1 (in degrees)
        - lon1: float - longitude of point 1 (in degrees)
        - height1: float - altitude of point 1 (in meters)
        - lat2: float - latitude of point 2 (in degrees)
        - lon2: float - longitude of point 2 (in degrees)
        - height2: float - altitude of point 2 (in meters)
        
        Returns:
        - float: the three-dimensional Haversine distance between the two points (in meters)
        """
        
        # Get the elevation of point 1 and point 2

        h1 = srtm1_data.get_altitude(lat1, lon1) + height1

        h2 = srtm1_data.get_altitude(lat2,lon2) + height2

        # Calculate the great circle distance between the points (in kilometers)
        distance_km = haversine_([lat1,lat2],[lon1,lon2],upper_tri=True)

        # Convert the distance to meters
        distance_m = distance_km[0] * 1000

        # Calculate the difference in height between the points (in meters)
        dheight = h2 - h1

        # Calculate the three-dimensional Haversine distance (in meters)
        d3 = np.sqrt(distance_m ** 2 + dheight ** 2)
        
        return d3
    
    
    def get_visibility(self) -> Dict:
        """
        Calculate visibility of schools from nearby towers.
        
        Returns:
        -------
        Dict:
            A dictionary containing the visibility information for each school. 
            Keys are the index of the schools, and values are another dictionary containing 
            whether the school is visible or not, and information about the towers within 
            range and visible to the school.
        """
        # download SRTM tiles that match the school and tower data
        self.download_matching_srtm_tiles()

        # find the column containing the school building height information
        height_cols = [col for col in self.school_data.columns if 'height' in str(col).lower()]

        # Check if there is more than one height column, and raise an error if so
        if len(height_cols) > 1:
            raise RuntimeError('There are more than one column in the school dataset indicating the school building height! Make sure there is only one height column.')
        # If there is only one height column, rename it to 'height'
        elif len(height_cols) ==1:
            self.school_data.rename(columns={height_cols[0]: 'height'}, inplace = True)
        # If there is no height column, assign the average school height to a new column 'height'
        else:
            self.school_data['height'] = self.avg_school_height
        
        # load SRTM elevation data
        srtm1_data = Srtm1HeightMapCollection(auto_build_index=True, hgt_dir=Path(self.srtm_folder_path))
        
        self.n_checks = 0

        # create a k-d tree of the tower locations for efficient nearest neighbor search
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
            #tower_match['dist_km'] = dist_km
            for twr in tower_match.itertuples():
                self.n_checks += 1
                has_line_of_sight = Visibility.check_visibility(srtm1_data, school.lat, school.lon, school.height, twr.lat, twr.lon, twr.height)
                visible_count += has_line_of_sight
                
                # If the tower is visible, add tower information to visibility dictionary for current school
                if has_line_of_sight:
                    twr_idx = 'tower_' + str(visible_count)
                    visibility_dict[school.Index].update({
                        twr_idx: twr.Index,
                        twr_idx + '_lat': twr.lat,
                        twr_idx + '_lon': twr.lon,
                        #twr_idx + '_dist': twr.dist_km,
                        twr_idx + '_anternna_dist': Visibility.calculate_three_dimension_haversine(srtm1_data, school.lat, school.lon, school.height, twr.lat, twr.lon, twr.height),
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
