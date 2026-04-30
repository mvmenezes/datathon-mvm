import pandera.pandas as pa
from pandera.pandas import DataFrameSchema, Column
from src.features.data import  recover_data_from_raw
from src.features.feature_engineering import feature_engineering
import pandas as pd
FEATURE_SCHEMA = DataFrameSchema({
    "Date": Column(pa.DateTime, coerce=True),
    "Open": Column(float, pa.Check.gt(0)),
    "High": Column(float, pa.Check.gt(0)),
    "Low": Column(float, pa.Check.gt(0)),
    "Close": Column(float, pa.Check.gt(0)),
    "Volume": Column(float, pa.Check.greater_than_or_equal_to(0)),
    "Dividends": Column(float, pa.Check.greater_than_or_equal_to(0)),
    "Stock Splits": Column(float, pa.Check.greater_than_or_equal_to(0)),
    "Dolar": Column(float, pa.Check.gt(0)),
    "short_mm": Column(float, pa.Check.gt(0)),
    "medium_mm": Column(float, pa.Check.gt(0)),
    "large_mm": Column(float, pa.Check.gt(0)),
})

STOCK = "TESTE.SA"
data = [
    ["2020-04-22 00:00:00-03:00",4.551616818102211,4.810068130493164,4.5315148546122614,4.810068130493164,114967100.0,0.0,0.0,5.456900119781494],
    ["2020-04-23 00:00:00-03:00",4.939295245064612,5.005343826165999,4.772737644004972,4.8675031661987305,94759000.0,0.0,0.0,5.533100128173828],
    ["2020-04-24 00:00:00-03:00",4.792838431065632,4.821555366721284,4.3879309528729165,4.580333709716797,161188800.0,0.0,0.0,5.5725998878479],
    ["2020-04-27 00:00:00-03:00",4.634895408698559,4.746891399355011,4.5315149366437355,4.7239179611206055,85944900.0,0.0,0.0,5.652599811553955],
    ["2020-04-28 00:00:00-03:00",4.893347311462528,4.953652381896972,4.775607823401515,4.953652381896973,91613600.0,0.0,0.0,5.497399806976318],
    ["2020-04-29 00:00:00-03:00",5.111595092477571,5.306869607747369,5.034060074251048,5.2264628410339355,94528400.0,0.0,0.0,5.334799766540527],
    ["2020-04-30 00:00:00-03:00",5.163284637398457,5.289638811802637,5.082877883647558,5.183386325836182,80263900.0,0.0,0.0,5.486000061035156],
    ["2020-05-04 00:00:00-03:00",5.005343660623354,5.059905792121307,4.933551584131867,4.990984916687012,60268400.0,0.0,0.0,5.4770002365112305],
    ["2020-05-05 00:00:00-03:00",5.140311731845954,5.306869300328226,5.131697034614277,5.151798725128174,75043400.0,0.0,0.0,5.540599822998047],
    ["2020-05-06 00:00:00-03:00",5.131697197864107,5.186258774500405,4.962267875671387,4.962267875671387,67937500.0,0.0,0.0,5.578100204467773],
]

columns = ["Date","Open","High","Low","Close","Volume","Dividends","Stock Splits","Dolar"]

df_teste = pd.DataFrame(data, columns=columns)

def test_schema_contract():
    """Features de saída devem respeitar o contrato de schema."""
    result = feature_engineering(df_teste,STOCK)
    FEATURE_SCHEMA.validate(result)

def test_no_nulls():
    """Nenhuma feature pode ter null após transformação."""
    result = feature_engineering(df_teste, STOCK)
    assert result.isnull().sum().sum() == 0


def test_row_count_preserved():
    """Número de registros deve ser preservado."""
    result = feature_engineering(df_teste,STOCK)
    assert len(result) == len(df_teste)