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

class FiberPath(GigaTools):

    def __init__(self, 
                    school_filename, 
                    osm_dataset = None,
                    path = os.getcwd(),
                    fiber_filename = 'none', 
                    school_subfoldername = '', 
                    school_id_column_name = 'giga_school_id',
                    fiber_id_column_name = 'source_id',
                    country_code = '',
                    max_dist_from_road = 2000,
                    n_clusters = 1
                    ):
        
        super().__init__(path)
        self.path = path
        self.data_path = os.path.join(path, 'data')
        self.school_file_path = os.path.join(self.data_path, 'school', school_subfoldername,  school_filename)
        self.school_filename = school_filename
        self.fiber_filename = fiber_filename
        self.school_subfoldername = school_subfoldername
        self.fiber_id_column_name = fiber_id_column_name
        self.fiber_file_path = os.path.join(self.data_path, 'raw', fiber_filename)
        self.school_id_column_name = school_id_column_name
        self.osm_dataset = osm_dataset
        self.country_code = country_code
        self.giga_edge_idx = 0
        self.max_dist_from_road = max_dist_from_road
        self.n_clusters = n_clusters

    
    def set_fiber_data(self):

        assert os.path.exists(self.fiber_file_path), 'Please make sure you specify the correct file name to the fiber data.'

        fiber = data_read(self.fiber_file_path)

        if self.fiber_id_column_name not in fiber.columns: 
            print(fiber.columns)
            raise ValueError('Given fiber id column is not in the fiber dataset. Please initialize fiber_id_column_name parameter with one of the above!')

        print('Note that fiber node id column will be used as school id column in the school dataset.')        
        fiber[self.school_id_column_name] = fiber[self.fiber_id_column_name]
        fiber.set_index(self.school_id_column_name, inplace=True)

        if type(fiber) != gp.geodataframe.GeoDataFrame:
            fiber = df_to_gdf(fiber, rename=True)
        else:
            rename_geo_cols(fiber)

        fiber['label'] = 'fiber'

        self.school_data = pd.concat([self.school_data, fiber[['lat', 'lon', 'geometry', self.fiber_id_column_name, 'label']]])

        assert self.school_data.index.duplicated().sum() == 0, 'Some fiber ids are already exist in the school dataset! Please make sure fiber ids do not overlap with school ids and re-run.'



    def prepare_data(self):

        if self.fiber_filename != 'none':
            #self.fiber_file_path = os.path.join(self.data_path, 'raw',  self.fiber_filename)
            self.set_fiber_data()

        assert 'label' in self.school_data.columns, 'School data should include "label" column indicating if school is connected (fiber) or not (school)'

        self.fiber_data = self.school_data.loc[self.school_data.label == 'fiber'].drop(columns='label')

        self.fiber_idx = self.fiber_data.index

        self.school_data.drop(columns='label', inplace = True)
        self.no_fiber = (len(self.fiber_data) == 0)
        print('School/Node data is ready for fiber path analysis!')
    

    def set_cluster_data(self, cluster_idx):
        
        self.school_data = self.cluster[cluster_idx]
        self.fiber_data = self.school_data.loc[self.fiber_idx]
        self.no_fiber = len(self.fiber_data) == 0
        print('Cluster ' + str(cluster_idx) + ' data is set!')


    
    def set_graph(self, buffer_around_bbox=0):

        if self.osm_dataset is not None:

            osm_dataset_path = os.path.join(self.data_path, 'raw', self.osm_dataset)
            if not os.path.exists(osm_dataset_path):
                osm_dataset_path = self.osm_dataset


            self.osm, self.nodes, self.edges = get_osm_road_data(self.osm_dataset, 
                                                                    locations = self.school_data, 
                                                                    network_type='all',
                                                                    buffer_around_bbox=buffer_around_bbox,
                                                                    nodes_id_column='id',
                                                                    verbose = True
            )
        
            self.graph, self.nodes, self.edges = initialize_geograph(self.nodes, 
                                                                        self.edges, 
                                                                        is_undirected = True, 
                                                                        crs = gnx.WGS84_CRS, 
                                                                        edges_from_to_columns = ['u', 'v'], 
                                                                        verbose = True,
            )

    def make_osm_graph_conected(self):
        subgraph_nodes = [i for i in nx.connected_components(self.graph)]
        num_subgraphs = len(subgraph_nodes)
        print('# of subgraphs in the graph: ' + str(num_subgraphs))
        
        if num_subgraphs > 1:

            # label nodes with the subgraph
            self.nodes['subgraph'] = 0
            for sub_idx in range(num_subgraphs):
                self.nodes.loc[list(subgraph_nodes[sub_idx]), 'subgraph'] = sub_idx

            print('Finding closest pair of nodes for each subraph pair...')
            # solve closest pair of points problem for each subgraph to find distances between each pair of subgraphs
            subgraph_pairs = list(itertools.combinations(range(num_subgraphs), 2))
            subgraphs_closest_pairs = {'u': [], 'v': [], 'distance': [], 'closest_in_u': [], 'closest_in_v':[]}

            for pair in subgraph_pairs:
                tree = KDTree(self.nodes.loc[self.nodes.subgraph == pair[0], ['lon', 'lat']])
                (distances, neighbors) = tree.query(self.nodes.loc[self.nodes.subgraph == pair[1], ['lon', 'lat']], k=1)
                min_dist_idx = distances.argmin()
                closest_in_u = self.nodes.loc[self.nodes.subgraph == pair[0], 'id'].iloc[neighbors[min_dist_idx][0]]
                closest_in_v = self.nodes.loc[self.nodes.subgraph == pair[1], 'id'].iloc[min_dist_idx]
                min_dist = distances[min_dist_idx][0]

                subgraphs_closest_pairs['u'].append(pair[0])
                subgraphs_closest_pairs['v'].append(pair[1])
                subgraphs_closest_pairs['distance'].append(min_dist)
                subgraphs_closest_pairs['closest_in_u'].append(closest_in_u)
                subgraphs_closest_pairs['closest_in_v'].append(closest_in_v)
                
            subgraphs_closest_pairs = pd.DataFrame(subgraphs_closest_pairs)

            print('Creating distance matrix of subgraphs...')
            # get subgraph distance matrix
            dmx_subgraph = np.zeros((num_subgraphs, num_subgraphs))
            for pair in subgraphs_closest_pairs.itertuples():
                dmx_subgraph[pair.u, pair.v] = pair.distance
                dmx_subgraph[pair.v, pair.u] = pair.distance

            print('Solving MST to add minimum cost edges to make the graph connected')
            # Solve MST to connect subgraphs each other with minimum cost
            # Calculate minimum spanning tree based on the square distance matrix
            mst_subgraph = minimum_spanning_tree(dmx_subgraph, overwrite = False)

            # Convert minimum spanning tree to array
            mst_subgraph = mst_subgraph.toarray()

            print('Adding subgraph connection edges to the graph...')
            for u, v in zip(*np.nonzero(mst_subgraph)):
                if u > v:
                    idx_u, idx_v = subgraphs_closest_pairs.loc[(subgraphs_closest_pairs.u == v) & (subgraphs_closest_pairs.v == u), ['closest_in_u','closest_in_v']].values[0]
                    self.graph.add_edge(idx_u, idx_v, label = 'graph_connection', id = 'giga_'+str(self.giga_edge_idx).zfill(7))
                    self.giga_edge_idx += 1
                else:
                    idx_u, idx_v = subgraphs_closest_pairs.loc[(subgraphs_closest_pairs.u == u) & (subgraphs_closest_pairs.v == v), ['closest_in_u','closest_in_v']].values[0]
                    self.graph.add_edge(idx_u, idx_v, label = 'graph_connection', id = 'giga_'+str(self.giga_edge_idx).zfill(7))
                    self.giga_edge_idx += 1

            gnx.fill_edges_missing_geometry_attributes(self.graph)
            gnx.fill_length_attribute(self.graph)
            print('Done! OSM graph is connected!')
    
    


    def merge_schools_to_graph(self):
        print('Spatial merging school/node locations to the graph...')
        tree = KDTree(self.nodes[['lon', 'lat']])
        dist, neighbors = tree.query(self.school_data[['lon', 'lat']])
        df_distance_to_graph = pd.DataFrame({'id': self.school_data.index, 'neighbour' : neighbors.reshape(-1), 'distance': dist.reshape(-1)}).sort_values(by = 'distance').reset_index(drop=True)
        df_distance_to_graph['geo_dist'] = df_distance_to_graph.apply(lambda x: gnx.get_distance(self.nodes.iloc[int(x.neighbour)].geometry, self.school_data.loc[x.id, 'geometry'], 'geodesic'), axis =1)
        
        school_max_distance = self.school_data.loc[df_distance_to_graph.loc[df_distance_to_graph['geo_dist']<=self.max_dist_from_road, 'id'], ['lat', 'lon', 'geometry']]
        self.graph = gnx.spatial_points_merge(self.graph, school_max_distance, intersection_nodes_attr={'label': 'school_merge'})

        for j in tqdm(df_distance_to_graph.loc[df_distance_to_graph['geo_dist']>self.max_dist_from_road, 'id']):
            self.graph = gnx.spatial_points_merge(self.graph, self.school_data.loc[j, ['lat', 'lon', 'geometry']], intersection_nodes_attr={'label': 'school_merge'})
        
        #self.graph = self.graph.to_undirected()
        gnx.fill_length_attribute(self.graph)

        node_labels = {idx: {'label': 'school'} if idx not in self.fiber_data.index else {'label': 'fiber'} for idx in self.school_data.index}
        nx.set_node_attributes(self.graph, node_labels)

        for node1, node2, data in self.graph.edges(data=True):
            try:
                data['id']
            except:
                nx.set_edge_attributes(self.graph, {(node1, node2): {'label': 'school_merge', 'id': 'giga_' + str(self.giga_edge_idx).zfill(7)}})
                self.giga_edge_idx +=1

        self.edges = self.graph.edges_to_gdf()
        self.nodes = self.graph.nodes_to_gdf()
        self.nodes.lon[self.nodes.lon.isnull()], self.nodes.lat[self.nodes.lat.isnull()] = self.nodes.geometry.x, self.nodes.geometry.y

        self.graph_pdn = self.osm.to_graph(self.nodes, self.edges, graph_type = 'pandana', direction = 'twoway', retain_all=True)

        print('Connected graph with school/node vertices')

    
    def set_school_distance_matrix(self):

        print('Calculating the school/node shortest path distance matrix...')
        
        # create pair combinations of schools/nodes
        ind = list(itertools.combinations(self.school_data.index, 2))
        ind_unz = list(zip(*ind))
        orig_nodes, dest_nodes = ind_unz[0], ind_unz[1]

        dmx_upper_triangle = self.graph_pdn.shortest_path_lengths(orig_nodes, dest_nodes)

        self.school_distance_matrix = upper_triangle_to_full_dmx(dmx_upper_triangle, len(self.school_data))
        print('Distance matrix is calculated!')
    

    def mst_schools(self):

        print('Finding minimum spanning tree of schools/nodes using shortest path distance matrix...')

        mst_schools = minimum_spanning_tree(self.school_distance_matrix, overwrite = False)

        # Convert minimum spanning tree to array
        mst_schools = mst_schools.toarray()

        # Get origin and destination road nodes ids from the minimum spanning tree array
        u_mst, v_mst = np.nonzero(mst_schools)
        mst_orig = self.school_data.iloc[u_mst].index
        mst_dest = self.school_data.iloc[v_mst].index

        mst_sp_geoms = self.graph_pdn.shortest_paths(mst_orig, mst_dest, imp_name=None)

        nx.set_edge_attributes(self.graph, False, 'mst')
        for path_ in mst_sp_geoms:
            nx.set_edge_attributes(self.graph, {edge_: {'mst': True} for edge_ in zip(path_[:-1], path_[1:])})
        
        if self.no_fiber:
            self.edges = self.graph.edges_to_gdf()

        self.mst_edges = self.edges[list(nx.get_edge_attributes(self.graph, 'mst').values())]

        self.mst_nodes = self.nodes.loc[list(dict.fromkeys(self.mst_edges[['u','v']].values.reshape(-1,)).keys())]
        nx.set_node_attributes(self.graph, False, 'mst')
        nx.set_node_attributes(self.graph, {idx: {'mst': True} for idx in self.mst_nodes.index})

        self.mst_graph = self.osm.to_graph(self.mst_nodes, self.mst_edges, graph_type = 'pandana', direction = 'twoway', retain_all=True)

        print('MST graph is initialized!')

        if self.no_fiber:
            print('Graph has no fiber nodes! Extracting MST paths...')

            nx_mst = nx.Graph()
            nx_mst.add_weighted_edges_from(zip(self.mst_edges['u'], self.mst_edges['v'], self.mst_edges['length']))
            nx.set_edge_attributes(nx_mst, self.mst_edges.set_index(['u', 'v'])['osm_type'], 'osm_type')
            
            mst_splitters = [node_[0] for node_ in nx_mst.degree() if node_[1]>2]
            
            nx.set_node_attributes(self.graph, False, 'mst_splitter')
            nx.set_node_attributes(self.graph, {idx: {'mst_splitter': True} for idx in mst_splitters})
            self.nodes = self.graph.nodes_to_gdf()
            self.mst_nodes = self.nodes.loc[self.mst_nodes.index]

            self.mst_path = pd.DataFrame(zip(mst_orig, mst_dest, [self.school_distance_matrix[u][v] for u, v in zip(u_mst, v_mst)]), columns= ['u', 'v', 'edge_length'])

            self.mst_path['mst_path'] = pd.Series(mst_sp_geoms).apply(lambda x: [el for el in x if el in list(self.school_data.index) + mst_splitters]).values
            self.mst_path['closest_vertice_on_mst_path'] = self.mst_path['mst_path'].apply(lambda x: x[1] if len(x) > 1 else x[0])
            self.mst_path['distance_to_closest_vertice_on_mst_path'] = self.mst_graph.shortest_path_lengths(tuple(self.mst_path.u), tuple(self.mst_path.closest_vertice_on_mst_path))

            self.mst_path['osm_length_of_mst_edge'] = [
                sum(
                    [nx_mst.edges[edge_]['weight'] for edge_ in zip(path_[:-1], path_[1:]) if nx_mst.edges[edge_]['label'] != nx_mst.edges[edge_]['label']]
                        ) for path_ in mst_sp_geoms
                        ]

            node_col = ['id', 'lat', 'lon', 'tags', 'mst_splitter', 'geometry']
            edge_col = ['u', 'v', 'label', 'length', 'geometry']

            self.mst_nodes = self.mst_nodes[node_col].rename(columns={'mst_splitter': 'splitter'})
            self.mst_edges = self.mst_edges[edge_col]
            self.fiber_path_length = self.mst_edges['length'].sum()*(0.001)
            
            print('MST paths are extracted!')

    

    def compute_fiber_path(self):

        print('Framing the closest node to each school...')

        ind = list(itertools.product(self.school_data.index, self.fiber_data.index))
        ind_unz = list(zip(*ind))
        orig_nodes, dest_nodes = ind_unz[0], ind_unz[1]

        # calculate distance matrix
        dmx_school_fiber = pd.DataFrame(np.array(self.mst_graph.shortest_path_lengths(orig_nodes, dest_nodes)).reshape(
            len(self.school_data), len(self.fiber_data)
            ), index = self.school_data.index, columns= self.fiber_data.index)

        # get closest fiber node for each school/fiber node
        self.closest_nodes_x = pd.DataFrame(zip(dmx_school_fiber.min(axis=1), dmx_school_fiber.idxmin(axis=1)), index = self.school_data.index, columns= ['distance_to_closest_fiber_node', 'closest_fiber_node_id']).sort_values(by='distance_to_closest_fiber_node')
        self.closest_nodes_x['connected_fiber_node_id'] = self.closest_nodes_x.closest_fiber_node_id
        self.closest_nodes_x['distance_to_connected_fiber_node'] = self.closest_nodes_x.distance_to_closest_fiber_node
        self.closest_nodes_x['distance_to_upstream_node'] = self.closest_nodes_x.distance_to_closest_fiber_node
        self.closest_nodes_x['connected_through'] = np.empty((len(self.closest_nodes_x), 0)).tolist()

        print('Iteratively connecting schools...')
        # iteratively find the closest connection node to each school over MST
        connected_idx = []
        for row in self.closest_nodes_x.itertuples():

            if row.distance_to_closest_fiber_node != 0:
                dist_to_connected = self.mst_graph.shortest_path_lengths(np.repeat(row.Index, len(connected_idx)), connected_idx)
                min_dist_to_connected = min(dist_to_connected)
                
                if min_dist_to_connected < row.distance_to_closest_fiber_node:
                    self.closest_nodes_x.at[row.Index, 'distance_to_upstream_node'] = min_dist_to_connected
                    connected_through = connected_idx[dist_to_connected.index(min_dist_to_connected)]
                    self.closest_nodes_x.at[row.Index, 'connected_through'] = self.closest_nodes_x.loc[connected_through, 'connected_through'] + [connected_through]
                    self.closest_nodes_x.at[row.Index, 'connected_fiber_node_id'] = self.closest_nodes_x.loc[connected_through, 'connected_fiber_node_id']
                    self.closest_nodes_x.at[row.Index, 'distance_to_connected_fiber_node'] = dmx_school_fiber.loc[row.Index, self.closest_nodes_x.at[row.Index, 'connected_fiber_node_id']]

            connected_idx.append(row.Index)
        
        print('Extracting fiber path...')
        school_to_fiber_paths = self.mst_graph.shortest_paths(tuple(self.closest_nodes_x.index), tuple(self.closest_nodes_x.connected_fiber_node_id))

        nx.set_edge_attributes(self.graph, False, 'fiber_path')
        for path_ in school_to_fiber_paths:
            nx.set_edge_attributes(self.graph, {edge_: {'fiber_path': True} for edge_ in zip(path_[:-1], path_[1:])})
        
        #self.edges = self.graph.edges_to_gdf() 

        self.fiber_path_edges = self.edges[list(nx.get_edge_attributes(self.graph, 'fiber_path').values())]

        self.fiber_path_length = self.fiber_path_edges['length'].sum()*(0.001) # in kilometers

        nx_fp = nx.Graph()
        nx_fp.add_weighted_edges_from(zip(self.fiber_path_edges['u'], self.fiber_path_edges['v'], self.fiber_path_edges['length']))
        nx.set_edge_attributes(nx_fp, self.fiber_path_edges.set_index(['u', 'v'])['label'], 'label')

        splitters = [node_[0] for node_ in nx_fp.degree() if node_[1]>2]

        nx.set_node_attributes(self.graph, False, 'fiber_path_splitter')
        nx.set_node_attributes(self.graph, {idx: {'fiber_path_splitter': True} for idx in splitters})
        
        #fiber_path_nodes_idx = dict.fromkeys(self.fiber_path_edges[['u','v']].values.reshape(-1,)).keys()
        fiber_path_nodes_idx = list(nx_fp.nodes)
        nx.set_node_attributes(self.graph, False, 'fiber_path')
        nx.set_node_attributes(self.graph, {idx: {'fiber_path': True} for idx in fiber_path_nodes_idx})
        
        self.edges = self.graph.edges_to_gdf() 
        self.nodes = self.graph.nodes_to_gdf()
        self.fiber_path_nodes = self.nodes.loc[fiber_path_nodes_idx]

        self.closest_nodes_x['fiber_path'] = pd.Series(school_to_fiber_paths).apply(lambda x: [el for el in x if el in list(self.school_data.index) + splitters]).values
        self.closest_nodes_x['closest_vertice_on_fiber_path'] = self.closest_nodes_x['fiber_path'].apply(lambda x: x[1] if len(x) > 1 else x[0])
        self.closest_nodes_x['distance_to_closest_vertice_on_fiber_path'] = self.mst_graph.shortest_path_lengths(tuple(self.closest_nodes_x.index), tuple(self.closest_nodes_x.closest_vertice_on_fiber_path))

        self.closest_nodes_x['osm_distance_to_connected_fiber_node'] = [
            sum(
                [nx_fp.edges[edge_]['weight'] for edge_ in zip(path_[:-1], path_[1:]) if nx_fp.edges[edge_]['label'] != nx_fp.edges[edge_]['label']]
                    ) for path_ in school_to_fiber_paths
                    ]

        node_col = ['id', 'lat', 'lon', 'label', 'fiber_path_splitter', 'geometry']
        edge_col = ['u', 'v', 'label', 'length', 'geometry']

        self.fiber_path_nodes = self.fiber_path_nodes[node_col].rename(columns={'fiber_path_splitter': 'splitter'})
        self.fiber_path_edges = self.fiber_path_edges[edge_col]
        
        print('Fiber path is computed!')
    

    def run_pipeline(self, buffer_around_bbox = 0):

        if self.n_clusters > 1:

            self.cluster_locations(self.n_clusters)

            for cluster in self.cluster:

                self.school_data = self.cluster[cluster]
                self.fiber_data = self.school_data.loc[[idx for idx in self.fiber_idx if idx in self.school_data.index]]
                self.no_fiber = len(self.fiber_data) == 0
                print('Cluster ' + str(cluster) + ' data is set!')
                
                self.set_graph(buffer_around_bbox = 1)
                self.make_osm_graph_conected()
                self.merge_schools_to_graph()
                self.set_school_distance_matrix()
                self.mst_schools()
                self.compute_fiber_path()

                output_filename = self.school_filename[:-4] + '_cluster_' + str(cluster) + '_' + datetime.date.today().strftime('%m%d%Y') + '.xlsx'
                self.write_output(output_filename)
        
        else:

            self.set_graph(buffer_around_bbox = buffer_around_bbox)
            self.merge_schools_to_graph()
            self.set_school_distance_matrix()
            self.mst_schools()
            self.compute_fiber_path()

            output_filename = self.school_filename[:-4] + '_' + datetime.date.today().strftime('%m%d%Y') + '.xlsx'
            self.write_output(output_filename)

    


    def write_output(self, output_filename):

        print('Writing results to an xlsx file...')

        #output_filename = self.school_file_name[:-4] + '_' + datetime.date.today().strftime('%m%d%Y') + '.xlsx'

        if self.no_fiber:

            with pd.ExcelWriter(os.path.join(self.data_path, 'output', self.school_file_folder, output_filename)) as writer:

                self.mst_path.to_excel(writer, sheet_name='mst_path', index=False)
                self.mst_nodes.to_excel(writer, sheet_name = 'mst_path_nodes', index = False)
                self.mst_edges.to_excel(writer, sheet_name='mst_path_edges', index = False)
                #pd.DataFrame(self.stats, index = range(1)).to_excel(writer, sheet_name = 'runtime_stats', index = False)

        else:
            
            with pd.ExcelWriter(os.path.join(self.data_path, 'output', self.school_file_folder, output_filename)) as writer:

                self.closest_nodes_x.to_excel(writer, sheet_name='closest_nodes', index=True)
                self.fiber_path_nodes.to_excel(writer, sheet_name = 'fiber_path_nodes', index = False)
                self.fiber_path_edges.to_excel(writer, sheet_name='fiber_path_edges', index = False)
                #pd.DataFrame(self.stats, index = range(1)).to_excel(writer, sheet_name = 'runtime_stats', index = False)
        
        print('Excel file is generated!')
    


    def plot_fiber_path(self):
            print('Visualising fiber path...')
            school_idx = list(set(self.school_data.index) - set(self.fiber_idx))
            xmin, ymin, xmax, ymax = self.school_data.total_bounds

            try:
                config = eval(open(os.path.join(self.path, 'assets', 'hex_config.py')).read())
                config['config']['mapState']['latitude'] = (ymin + ymax)/2
                config['config']['mapState']['longitude'] = (xmin + xmax)/2
                print('Config file is read!')
            except:
                config = {
                    'version': 'v1',
                    'config': {
                        'mapState': {
                            'latitude': (ymin + ymax)/2,
                            'longitude': (xmin + xmax)/2,
                            'zoom': 10
                        },
                    }
                }

            map_ = KeplerGl(height=600, config=config, show_docs=False, read_only=True,
                            data={
                                'fiber nodes': self.school_data.loc[self.fiber_idx, ['geometry']].to_json(),
                                #'connected schools': fiber_path_nodes.loc[connected_school_idx, ['tags', 'id', 'geometry']].to_json(),
                                'schools': self.school_data.loc[school_idx, ['geometry']].to_json(),
                                'splitters': self.nodes.loc[self.nodes['fiber_path_splitter'], ['id', 'geometry']].to_json(),
                                'fiber path': self.edges.loc[self.edges['fiber_path'], 'geometry'].to_json(),
                                #'school merge edges': edges.loc[edges.osm_type == 'school_intersect', ['u', 'v', 'geometry']].to_json(),
                                }
                                )

            map_.save_to_html(config=config, file_name=os.path.join(self.data_path, 'output', self.school_file_folder, self.school_file_name[:-4] + '.html',), read_only=False)
            print('Visual saved as HTML file to the output folder!')

