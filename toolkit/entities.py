from abc import ABC, abstractmethod
import pandas as pd
#import geopandas as gp
from utils import data_read, df_to_gdf
from config import data_path

class Entity(ABC):

    def __init__(self, entity_type, filename, id_column):
        self.entity_type = entity_type
        self.filepath = data_path + '/' + filename
        self.id_column = id_column
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
        df = data_read(self.filepath)
        self.data = df_to_gdf(df, rename = True)

        self._check_id_column()
        self._check_duplicates()

        self.data.set_index(self.id_column, inplace=True)
        
        self.prepare_data(self.data)
    


class School(Entity):
    
    def __init__(self, filename):
        super().__init__('School', filename, id_column='giga_school_id')
    
    def prepare_data(self, df):
        # Code to read data from the file and prepare the data specifically for a school entity
        self.data = df



class CellTower(Entity):
    
    id_column = 'tower_id'

    def __init__(self, filename, operator):
        Entity.__init__(self, "Cell Tower", filename)
        self.operator = operator
    
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
