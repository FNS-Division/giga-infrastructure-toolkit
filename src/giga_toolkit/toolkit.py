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

from giga_toolkit.utils import *
from giga_toolkit import apiconfig

class GigaTools:

    TOOLKIT_VERSION = apiconfig.giga_toolkit_config['TOOLKIT_VERSION']

    def __init__(self, 
                    path,
                    country_code = 'none',
                    school_filename = 'none', 
                    school_subfoldername = '', 
                    school_id_column_name = 'giga_school_id', 
                    random_state = 0
                    ):

        self.path = path
        self.data_path = os.path.join(path, 'data')
        self.school_file_path = os.path.join(self.data_path, 'school', school_subfoldername,  school_filename)
        self.school_filename = school_filename
        self.school_subfoldername = school_subfoldername
        self.school_id_column_name = school_id_column_name
        self.country_code = country_code
        self.random_state = random_state
        self.earth_r = 6371
        self.srtm_base_url = 'https://e4ftl01.cr.usgs.gov//DP133/SRTM/SRTMGL1.003/2000.02.11/'

        print('Please make sure that you have given the correct path to your main folder: ')
        print(self.path)
    

    def set_school_data(self):

        assert os.path.exists(self.school_file_path), 'Please make sure you specify the correct file name to the school data. If the data in the subfolder of school folder please specify school_file_folder!'

        self.school_data = data_read(self.school_file_path)

        try:
            self.school_data.set_index(self.school_id_column_name, inplace=True)
        except:
            print('Columns in the dataset: ')
            print(self.school_data.columns)
            raise ValueError('Given school id column is not in the school dataset: please initialize school_id_column_name parameter with one of the above!')

        assert self.school_data.index.duplicated().sum() == 0, 'Duplicate ids exist in the school data! Please make sure each row has unique id and re-run.'
        
        self.school_data = df_to_gdf(self.school_data, rename = True)

        print('School data is set!')
    

    def cluster_locations(self, n_clusters):

        print('Clustering schools based on location...')
        
        kmeans_ = KMeans(n_clusters=n_clusters, random_state=self.random_state).fit(self.school_data[['lat', 'lon']])

        self.cluster = {}

        for sg_idx in range(n_clusters):
            self.cluster[sg_idx] = self.school_data[kmeans_.labels_ == sg_idx]
        
        print('Clusters are initialized!')