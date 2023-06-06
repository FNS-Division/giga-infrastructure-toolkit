import numpy as np
import pandas as pd
import geopandas as gp
import gdal
from typing import Optional
import os

class DataLoader:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        

    def read_file(self, file_name):
        """
        Read a file from the data directory.

        Args:
            file_name (str): The name of the file to read.

        Returns:
            pd.DataFrame or gp.GeoDataFrame: The data read from the file.
        """
        file_path = os.path.join(self.data_dir, file_name)
        suffix = os.path.splitext(file_name)[1]

        assert os.path.exists(file_path), f'{file_name} cannot be found in the data directory!'

        if suffix == '.csv':
            return pd.read_csv(file_path)
        elif suffix == '.xlsx' or suffix == '.xls':
            return pd.read_excel(file_path, engine = 'openpyxl')
        elif suffix == '.shp' or suffix == '.zip':
            return gp.read_file(file_path)
        elif suffix == '.parquet':
            return pd.read_parquet(file_path)
        elif suffix == '.geoparquet':
            return gp.read_parquet(file_path)
        elif suffix == '.json':
            return pd.read_json(file_path)
        elif suffix == '.geojson':
            return gp.read_file(file_path)
        elif suffix == '.gpkg':
            return gp.read_file(file_path)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")
    
    def save_data(self, data, file_name):
        """
        Saves the input data to a file in the data directory.

        Parameters:
            data (pandas.DataFrame): The data to save.
            file_name (str): The name of the file to save the data to. The file extension determines the file type.
                Supported file types include '.csv', '.parquet', and '.geoparquet'.

        Raises:
            ValueError: If an unsupported file type is specified.

        Returns:
            None
        """
        file_path = os.path.join(self.data_dir, file_name)
        suffix = os.path.splitext(file_name)[1]

        if suffix == '.csv':
            data.to_csv(file_path, index = False)
        elif suffix == '.parquet':
            data.to_parquet(file_path, index = False)
        elif suffix == '.geoparquet':
            if isinstance(data, gp.GeoDataFrame):
                data.to_parquet(file_path, index = False)
            else:
                raise ValueError("Data must be a GeoDataFrame for geoparquet format")
        else:
            raise ValueError(f"Unsupported file type: {suffix}")
    

    def get_georeference_columns(self, data, lat_keywords=['latitude', 'lat', 'y', 'lat_', 'lat(s)'], lon_keywords=['longitude', 'lon', 'long', 'x', 'lon_', 'lon(e)', 'long(e)']):
        """
        Searches for latitude and longitude columns in the input data using a list of keywords.
        
        Args:
        data (pandas.DataFrame): A DataFrame containing the data to search for latitude and longitude columns.
        lat_keywords (list of str): A list of keywords to search for in column names to identify latitude columns.
        lon_keywords (list of str): A list of keywords to search for in column names to identify longitude columns.
        
        Returns:
        A tuple of two strings representing the names of the latitude and longitude columns, respectively.
        
        Raises:
        ValueError: If no unique pair of latitude/longitude columns can be found in the input data.
        """

        # Search for columns that match common names for latitude and longitude
        lat_cols = [col for col in data.columns if any(keyword in col.lower() for keyword in lat_keywords)]
        lon_cols = [col for col in data.columns if any(keyword in col.lower() for keyword in lon_keywords)]

        # Check if exactly one latitude and longitude column is found
        if len(lat_cols) == 1 and len(lon_cols) == 1:
            return lat_cols[0], lon_cols[0]
        elif len(lat_cols) == 0 and len(lon_cols) == 0:
            raise ValueError("No latitude or longitude columns found.")
        elif len(lat_cols) == 0:
            raise ValueError("No latitude columns found.")
        elif len(lon_cols) == 0:
            raise ValueError("No longitude columns found.")
        else:
            raise ValueError("Could not find a unique pair of latitude/longitude columns.")
    
    def rename_georeference_columns(
            self, 
            data: pd.DataFrame, 
            lat_col: str = 'lat', 
            lon_col: str = 'lon', 
            old_lat_col: Optional[str] = None, 
            old_lon_col: Optional[str] = None
        ) -> pd.DataFrame:

        """
        Renames the georeference columns of a given data frame to the specified column names.
        
        Parameters:
            data (pd.DataFrame): The data frame containing the georeference columns to be renamed.
            lat_col (str): The new name for the latitude column. Default is 'lat'.
            lon_col (str): The new name for the longitude column. Default is 'lon'.
            old_lat_col (Optional[str]): The old name for the latitude column. If not specified, 
                                        the method will attempt to find the old name using `get_georeference_columns()`.
            old_lon_col (Optional[str]): The old name for the longitude column. If not specified, 
                                        the method will attempt to find the old name using `get_georeference_columns()`.
        
        Returns:
            pd.DataFrame: The input data frame with the georeference columns renamed.
        
        Raises:
            ValueError: If the specified old column names do not exist in the data frame.
        """
        # If the old column names are not specified, attempt to find them
        if old_lat_col is None or old_lon_col is None:
            old_lat_col, old_lon_col = self.get_georeference_columns(data)

        # Check if the old column names exist in the data frame
        if old_lat_col not in data.columns or old_lon_col not in data.columns:
            raise ValueError(f"One or both of the specified old column names '{old_lat_col}', '{old_lon_col}' do not exist in the data frame.")

        # If the old column names don't match the new ones, rename them
        if old_lat_col != lat_col:
            data = data.rename(columns={old_lat_col: lat_col})
        if old_lon_col != lon_col:
            data = data.rename(columns={old_lon_col: lon_col})
        
        return data

    def to_geodataframe(self, data, lat_col=None, lon_col=None):

        """
        Converts a DataFrame containing latitude and longitude columns into a GeoDataFrame.

        Parameters
        ----------
        data : pandas.DataFrame
            The DataFrame containing the latitude and longitude columns.
        lat_col : str, optional
            The name of the column containing latitude values. If not provided, the method will try to find it.
        lon_col : str, optional
            The name of the column containing longitude values. If not provided, the method will try to find it.

        Returns
        -------
        geopandas.GeoDataFrame
            A GeoDataFrame with the same columns as the input DataFrame and a new geometry column containing
            the Point objects created from the latitude and longitude columns.
        """
        
        # If the latitude and longitude columns are not already specified, try to find them
        if lat_col is None or lon_col is None:
            lat_col, lon_col = self.get_georeference_columns(data)
        
        # Extract the latitude and longitude values from the input data
        latitudes = data[lat_col]
        longitudes = data[lon_col]
        
        # Create a new GeoDataFrame with the input data and geometry column
        geometry = gp.points_from_xy(longitudes, latitudes)
        gdf = gp.GeoDataFrame(data, geometry=geometry, crs="EPSG:4326")
        
        return gdf
    
    def process_tif(self, file_name: str, band_no: int = 1, drop_nodata: bool = True, return_res: bool = False) -> pd.DataFrame:
        """
        Processes a .tif file and returns a pandas DataFrame containing the longitude, latitude,
        and pixel values of the file.

        Parameters:
            file_name (str): The name of the .tif file.
            band_no (int): The band number of the .tif file to be read. Default is 1.
            drop_nodata (bool): Whether to drop the pixels with no data value. Default is True.
            return_res (bool): Whether to return the xsize of the .tif file along with the DataFrame.
                            Default is False.

        Returns:
            pandas.DataFrame: A DataFrame containing the longitude, latitude, and pixel values of the .tif file.
        """
        file_path = os.path.join(self.data_dir, file_name)
        
        # Open the tif file with gdal
        tif = gdal.Open(file_path)
        
        # Check if the band exists in the tif file
        if band_no < 1 or band_no > tif.RasterCount:
            raise ValueError(f"Invalid band number {band_no} for file {file_name}.")
        
        # Get the specified band
        band = tif.GetRasterBand(band_no)

        # Read band values as array
        band_values = band.ReadAsArray()

        # Get the no data value of the band
        nodata_value = band.GetNoDataValue()

        # Get the geotransform parameters of the tif file
        offX, xsize, line1, offY, line2, ysize = tif.GetGeoTransform()

        # Get the number of columns and rows in the tif file
        cols = tif.RasterXSize
        rows = tif.RasterYSize

        # Create one-dimensional arrays for x and y
        x = np.linspace(offX + xsize/2, offX + xsize/2 + (cols-1)*xsize, cols)
        y = np.linspace(offY + ysize/2, offY + ysize/2 + (rows-1)*ysize, rows)

        # Create the mesh based on these arrays
        X, Y = np.meshgrid(x, y)

        # Extract the pixel values, longitude, and latitude arrays from the tif file
        if drop_nodata:
            nodata_mask = band_values != nodata_value
            pixel_values = np.extract(nodata_mask, band_values)
            longitude = np.extract(nodata_mask, X)
            latitude = np.extract(nodata_mask, Y)
        else:
            pixel_values = band_values.flatten()
            longitude = X.reshape((np.prod(X.shape),))
            latitude = Y.reshape((np.prod(Y.shape),))
        
        # Create a dictionary with the extracted arrays
        tif_dict = {'longitude': longitude, 'latitude': latitude, 'pixel_value': pixel_values}

        # Convert the dictionary to a pandas DataFrame
        df = pd.DataFrame(tif_dict)

        if return_res:
            return df, xsize
        else:
            return df