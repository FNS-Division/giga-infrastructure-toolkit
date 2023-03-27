from abc import ABC, abstractmethod
import pandas as pd
import os
from toolkit.utils import data_read, df_to_gdf

class Entity(ABC):

    def __init__(self, entity_type, filename, id_column, path = os.getcwd()):
        self.entity_type = entity_type
        #self.filepath = path + 'data/' + filename
        self.id_column = id_column
        self.path = path
        self.data = None
    
    def get_type(self):
        return self.entity_type
    
    def get_data(self):
        return self.data

    def _check_id_column(self):
        if self.id_column not in self.data.columns:
            raise ValueError(f"The id column '{self.id_column}' does not exist in the dataset")
    
    def _check_duplicates(self):
        if self.data[self.id_column].duplicated().any():
            raise ValueError(f"Duplicate id found in {self.id_column} column")
    
    @abstractmethod
    def prepare_data(self):
        raise NotImplementedError("Subclass must implement abstract method")

    
    def set_data(self):

        filepath = os.path.join(self.path, 'data', self.filename)

        assert os.path.exists(filepath), f'{self.entity_type} data file does not exist in the data path. Please make sure you specify the correct filename for the {self.entity_type} data.'

        df = data_read(filepath)
        self.data = df_to_gdf(df, rename = True)

        self._check_id_column()
        self._check_duplicates()

        self.data.set_index(self.id_column, inplace=True)
        
        self.prepare_data(self.data)
    


class POI(Entity):
    
    def __init__(self, poi_type = 'poi', filename = conf.poi , id_column = 'poi_id'):
        super().__init__(entity_type = poi_type, filename = filename, id_column=id_column)
    
    def prepare_data(self, df):
        # Code to read data from the file and prepare the data specifically for a point of interest entity
        self.data = df



class CellTower(Entity):
    
    id_column = 'ict_id'

    def __init__(self, filename = conf.celltower_filename, id_column = 'ict_id', entity_type = 'CellTower'):
        super().__init__(entity_type= entity_type, filename = filename, id_column= id_column)
        #self.operator = operator
    
    def prepare_data(self):
        # Code to read data from the file and prepare the data specifically for a cell tower entity
        df = pd.read_csv(self.filename)
        #df = preprocess_cell_tower_data(df)
        self.data = self.df_to_gdf(df)


class FiberNode(Entity):

    id_column = 'fiber_id'

    def __init__(self, filename, operator):
        Entity.__init__(self, "Fiber Node", filename)
        self.operator = operator
    
    def prepare_data(self):
        # Code to read data from the file and prepare the data specifically for a fiber node entity
        df = pd.read_csv(self.filename)
        #df = preprocess_fiber_node_data(df)
        self.data = self.df_to_gdf(df)
