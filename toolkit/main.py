from network_graph import NetworkGraph
from entities import School
from fiber_path import FiberPath

school = School('data/schools.csv')
#school.prepare_data()

road_graph = NetworkGraph(osm_dataset = 'bwa', locations = school)
#road_graph.prepare_data()