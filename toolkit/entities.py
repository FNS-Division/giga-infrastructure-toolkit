from abc import ABC, abstractmethod
import pandas as pd
from toolkit import data_loader
from config import data
import os

class Entity(ABC):

    _shared_state = dict()

    def __init__(self, path = os.path.dirname(os.path.dirname(os.path.abspath())), country_code = None, entity_type = 'Unidentified', filename=None, id_column=None, crs = 'epsg:4326'):
        
        if not isinstance(entity_type, str):
            raise TypeError("entity_type should be a string")

        self.__dict__ = self._shared_state
        self.path = path
        self.data_dir = os.path.join(path, 'data')
        self.country_code = country_code
        self.entity_type = entity_type
        self.filename = filename
        self.id_column = id_column

        self.data = None
        #filepath = self.get_filepath()

        #self._check_file_exists(filepath)

    
    def get_type(self):
        return self.entity_type
    
    def get_data(self):
        return self.data
    
    def get_filepath(self):
        
        if self.filename is not None:
            return os.path.join(self.path, 'data', self.country_code.upper(), self.filename)
        else:
            return None
        
    def _check_country_folder_exists(self):
        if self.country_code is not None:
             assert os.path.exists(os.path.join(self.path, 'data', self.country_code.upper())), ValueError(f'The country folder {self.country_code.upper()} does not exist in the data folder!')
    
    def _check_file_exists(self, filepath):
        
        if filepath is not None:
            assert os.path.exists(filepath), ValueError(f'The file {self.filename} does not exist in the {self.country_code.upper()} folder!')


    def _validate_id_column(self):
        try:
            self.data[self.id_column]
        except (KeyError, TypeError):
            raise ValueError(f"{self.id_column} is not a valid column name")


    def _check_duplicates(self):
        if self.data[self.id_column].duplicated().any():
            raise ValueError(f"Duplicate id found in {self.id_column} column")
    
    @abstractmethod
    def prepare_data(self, data, crs):
        raise NotImplementedError("Subclass must implement abstract method")


class POI(Entity):

    def __init__(self, **kwargs):
        super().__init__(self)
        self._shared_state.update(kwargs)
    



class CellTower(Entity):
    
    id_column = 'ict_id'

    def __init__(self, filename, id_column = 'ict_id', entity_type = 'CellTower'):
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
