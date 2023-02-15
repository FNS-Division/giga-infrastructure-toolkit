
class NetworkGraph:
    
    def __init__(self, osm_data=None, locations=None):
        self.graph = None
        if osm_data is not None:
            # create the graph using road data
            self.graph = self.prepare_data(osm_data)
        elif locations is not None:
            # create the graph using line of sight edges
            self.graph = self.create_los_graph(locations)

    def prepare_data(self, data):
        # code to create the graph using road data
        pass

    def create_los_graph(self, locations):
        # code to create the graph using line of sight edges
        pass