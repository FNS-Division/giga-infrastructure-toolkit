
class NetworkGraph:
    
    def __init__(self, locations, use_road_data = True):
        self.graph = None

        if use_road_data:
            self.graph = self.prepare_data()
        else:
            # create the graph using line of sight edges
            self.graph = self.create_los_graph(locations)

    def prepare_data(self, data):
        # code to create the graph using road data
        pass

    def create_los_graph(self, locations):
        # code to create the graph using line of sight edges
        pass