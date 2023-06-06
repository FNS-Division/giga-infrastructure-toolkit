from enum import Enum
from typing import Set

class CellTechnology(str, Enum):
    """Valid level of education"""

    TWO_G = "2G"
    THREE_G = "3G"
    FOUR_G = "4G"
    LTE = "LTE"
    FIVE_G = '5G'


class CellTower:
    """Definition for a single cell tower"""
    def __init__(self, tower_id: str, lat: float, lon: float, height: float, technology: Set[CellTechnology]):


        self.tower_id = tower_id
        self.lat = lat
        self.lon = lon
        self.height = height
        self.technology = technology


