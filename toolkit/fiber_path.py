from network_graph import NetworkGraph

class FiberPath:

    def __init__(self, network_graph, entity):
        self.road_graph = network_graph
        self.locations = entity.data
    
    def compute_fiber_path(self, entity_id):
        # access the entity data and find the nearest fiber node
        entity = self.entity_data.loc[entity_id]
        nearest_fiber_node = ...
        return nearest_fiber_node