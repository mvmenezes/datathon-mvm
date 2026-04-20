import pandera as pa
from pandera import Column, DataFrameSchema
from hypothesis import given
from src.features.feature_engineering import feature_engineering

STOCK = "TEST_STOCK"
FEATURE_SCHEMA = DataFrameSchema({
    "Date": Column(pa.DateTime),
    "Open": Column(float, pa.Check.gt(0)),
    "High": Column(float, pa.Check.gt(0)),
    "Low": Column(float, pa.Check.gt(0)),
    "Close": Column(float, pa.Check.gt(0)),
    "Volume": Column(float, pa.Check.gt(0)),
    "Dividends": Column(float, pa.Check.gt(0)),
    "Stock Splits": Column(float, pa.Check.gt(0)),
    "Dolar": Column(float, pa.Check.gt(0)),
    "short_mm": Column(float, pa.Check.gt(0)),
    "medium_mm": Column(float, pa.Check.gt(0)),
    "large_mm": Column(float, pa.Check.gt(0)),
})


@given(FEATURE_SCHEMA.strategy(size=100))
def test_schema_contract(data, stock=STOCK):
    """Features de saída devem respeitar o contrato de schema."""
    result = feature_engineering(data,stock)
    FEATURE_SCHEMA.validate(result)

@given(FEATURE_SCHEMA.strategy(size=100))
def test_no_nulls(data, stock=STOCK):
    """Nenhuma feature pode ter null após transformação."""
    result = feature_engineering(data,stock)
    assert result.isnull().sum().sum() == 0


@given(FEATURE_SCHEMA.strategy(size=100))
def test_row_count_preserved(data, stock=STOCK):
    """Número de registros deve ser preservado."""
    result = feature_engineering(data,stock)
    assert len(result) == len(data)