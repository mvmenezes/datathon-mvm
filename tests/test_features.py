import pandera as pa
from pandera import Column, DataFrameSchema
from hypothesis import given
from src.features.data import recover_data_from_processed, recover_data_from_raw
from src.features.feature_engineering import feature_engineering

STOCK = "TESTE"
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


def test_schema_contract():
    """Features de saída devem respeitar o contrato de schema."""
    data = recover_data_from_raw(STOCK)
    result = feature_engineering(data,STOCK)
    FEATURE_SCHEMA.validate(result)

def test_no_nulls():
    """Nenhuma feature pode ter null após transformação."""
    data = recover_data_from_raw(STOCK)
    result = feature_engineering(data, STOCK)
    assert result.isnull().sum().sum() == 0


def test_row_count_preserved():
    """Número de registros deve ser preservado."""
    data = recover_data_from_raw(STOCK)
    result = feature_engineering(data,STOCK)
    print(len(result) , len(data))
    assert len(result) == len(data)