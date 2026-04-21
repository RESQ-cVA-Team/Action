from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel


class ChartType(str, Enum):
    LINE = "LINE"
    BAR = "BAR"
    BOX = "BOX"
    HISTOGRAM = "HISTOGRAM"
    SCATTER = "SCATTER"
    PIE = "PIE"
    RADAR = "RADAR"
    WATERFALL = "WATERFALL"
    AREA = "AREA"


class ChartPoint(BaseModel):
    """Single data point for charts"""

    x: Union[str, int, float]
    y: Union[int, float]
    label: Optional[str] = None


class ChartSeries(BaseModel):
    """Data series for multi-line/multi-series charts"""

    name: str
    data: List[ChartPoint]
    color: Optional[str] = None
    style: Optional[Dict[str, Any]] = None


class ChartAxis(BaseModel):
    """Chart axis configuration"""

    label: str

    class AxisType(str, Enum):
        LINEAR = "linear"
        LOGARITHMIC = "logarithmic"
        CATEGORY = "category"
        TIME = "time"

    type: AxisType = AxisType.LINEAR
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    unit: Optional[str] = None


class ChartMetadata(BaseModel):
    """Common chart metadata"""

    title: str
    x_axis: Optional[ChartAxis] = None
    y_axis: Optional[ChartAxis] = None
    legend: bool = True
    width: Optional[int] = None
    height: Optional[int] = None
