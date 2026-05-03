import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from src.features.feature_engineering import save_parquet
from src.features.feature_engineering import _create_strategy  
from src.features.feature_engineering import feature_engineering  
from src.features.feature_engineering import recover_data_from_raw
from src.features.data import _download_data, download_data, recover_data_from_processed  
import src.features.data as mod 

# ── helpers ──────────────────────────────────────────────────────────────────

def make_raw_df(n: int = 60) -> pd.DataFrame:
    """Cria um DataFrame mínimo que simula um CSV da pasta data/raw/."""
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    rng = np.random.default_rng(42)
    close = 30 + rng.standard_normal(n).cumsum()
    return pd.DataFrame(
        {
            "Date": dates.strftime("%Y-%m-%d"),
            "Open": close * 0.99,
            "High": close * 1.01,
            "Low": close * 0.98,
            "Close": close,
            "Volume": rng.integers(100_000, 1_000_000, n),
        }
    )


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def raw_df() -> pd.DataFrame:
    return make_raw_df()


@pytest.fixture()
def mock_rag(monkeypatch):
    """Neutraliza as chamadas ao pipeline RAG para não precisar de infra."""
    monkeypatch.setattr(
        "src.pipeline.feature_engineering.stock_df_to_documents",
        MagicMock(return_value=[]),
    )
    monkeypatch.setattr(
        "src.pipeline.feature_engineering.upsert_documents",
        MagicMock(),
    )


# ── _create_strategy ──────────────────────────────────────────────────────────

class TestCreateStrategy:
    """Testes unitários para a função _create_strategy."""

    def _call(self, df: pd.DataFrame) -> pd.DataFrame:
        # Importa aqui para o monkeypatch funcionar corretamente

        return _create_strategy(df.copy())

    def test_colunas_medias_moveis_existem(self, raw_df):
        result = self._call(raw_df)
        for col in ("short_mm", "medium_mm", "large_mm"):
            assert col in result.columns, f"Coluna ausente: {col}"

    def test_colunas_rsi_existem(self, raw_df):
        result = self._call(raw_df)
        assert "RSI" in result.columns
        # Colunas intermediárias devem ter sido removidas
        for col in ("Gain", "Loss", "RS", "Delta"):
            assert col not in result.columns, f"Coluna intermediária não removida: {col}"

    def test_colunas_bollinger_existem(self, raw_df):
        result = self._call(raw_df)
        assert "bb_upper_band" in result.columns
        assert "bb_lower_band" in result.columns
        # Colunas intermediárias removidas
        assert "ma20" not in result.columns
        assert "std20" not in result.columns

    def test_rsi_dentro_do_intervalo(self, raw_df):
        result = self._call(raw_df)
        assert result["RSI"].between(0, 100).all(), "RSI fora do intervalo [0, 100]"

    def test_bollinger_upper_maior_que_lower(self, raw_df):
        result = self._call(raw_df)
        assert (result["bb_upper_band"] >= result["bb_lower_band"]).all()

    def test_short_mm_menor_periodo_que_large_mm(self, raw_df):
        """short_mm reage mais rápido; em tendência de alta deve ser >= large_mm no final."""
        # Série estritamente crescente
        n = 60
        dates = pd.date_range("2024-01-01", periods=n, freq="B")
        df_up = pd.DataFrame(
            {"Date": dates, "Close": np.linspace(10, 100, n)}
        )
        result = self._call(df_up)
        # Na última linha de uma série crescente: short_mm > large_mm
        last = result.iloc[-1]
        assert last["short_mm"] >= last["large_mm"]

    def test_sem_nan_apos_dropna(self, raw_df):
        result = self._call(raw_df)
        assert not result.isnull().any().any(), "Existem NaN após dropna"


# ── feature_engineering ───────────────────────────────────────────────────────

class TestFeatureEngineering:
    """Testes de integração (RAG mockado) para feature_engineering."""

    def _call(self, df, stock="TEST4.SA"):

        return feature_engineering(df.copy(), stock)

    def test_retorna_dataframe(self, raw_df, mock_rag):
        result = self._call(raw_df)
        assert isinstance(result, pd.DataFrame)

    def test_date_convertida_para_datetime(self, raw_df, mock_rag):
        result = self._call(raw_df)
        assert pd.api.types.is_datetime64_any_dtype(result["Date"])

    def test_colunas_de_estrategia_presentes(self, raw_df, mock_rag):
        result = self._call(raw_df)
        esperadas = ["short_mm", "medium_mm", "large_mm", "RSI", "bb_upper_band", "bb_lower_band"]
        for col in esperadas:
            assert col in result.columns

    def test_rag_chamado_com_stock_correto(self, raw_df, monkeypatch):
        mock_to_docs = MagicMock(return_value=[])
        mock_upsert = MagicMock()
        monkeypatch.setattr("src.pipeline.feature_engineering.stock_df_to_documents", mock_to_docs)
        monkeypatch.setattr("src.pipeline.feature_engineering.upsert_documents", mock_upsert)

        stock = "VALE3.SA"
        self._call(raw_df, stock=stock)

        mock_to_docs.assert_called_once()
        _, call_stock = mock_to_docs.call_args.args
        assert call_stock == stock

        mock_upsert.assert_called_once()

    def test_levanta_value_error_para_df_invalido(self, mock_rag):
        df_invalido = pd.DataFrame({"Date": ["2024-01-01"], "Close": [None]})
        with pytest.raises(ValueError, match="Não foi possivel recuperar"):
            self._call(df_invalido)


# ── recover_data_from_raw ─────────────────────────────────────────────────────

class TestRecoverDataFromRaw:
    def _call(self, stock):

        return recover_data_from_raw(stock)

    def test_leitura_bem_sucedida(self, tmp_path, monkeypatch):
        stock = "PETR4.SA"
        csv_path = tmp_path / f"{stock}.csv"
        make_raw_df().to_csv(csv_path, index=False)

        # Redireciona o caminho de leitura
        monkeypatch.chdir(tmp_path)
        import os; os.makedirs("data/raw", exist_ok=True)
        make_raw_df().to_csv(f"data/raw/{stock}.csv", index=False)

        df = self._call(stock)
        assert isinstance(df, pd.DataFrame)
        assert "Close" in df.columns

    def test_arquivo_inexistente_levanta_value_error(self):
        with pytest.raises(ValueError, match="Não foi possível recuperar"):
            self._call("ACAO_INEXISTENTE.SA")


# ── save_parquet ──────────────────────────────────────────────────────────────

class TestSaveParquet:
    def test_salva_arquivo_parquet(self, tmp_path, monkeypatch, raw_df):

        monkeypatch.chdir(tmp_path)
        import os; os.makedirs("data/processed", exist_ok=True)

        stock = "ITUB4.SA"
        save_parquet(raw_df, stock)

        saved = pd.read_parquet(f"data/processed/{stock}.parquet")
        assert len(saved) == len(raw_df)
        assert list(saved.columns) == list(raw_df.columns)




import pytest
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock


# ── helpers ───────────────────────────────────────────────────────────────────

def make_ohlcv_df(n: int = 30, ticker_col: str = "Close") -> pd.DataFrame:
    """Cria um DataFrame no formato retornado pelo yfinance."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    close = 30 + rng.standard_normal(n).cumsum()
    return pd.DataFrame(
        {
            "Date": dates,
            "Open": close * 0.99,
            "High": close * 1.01,
            "Low": close * 0.98,
            ticker_col: close,
            "Volume": rng.integers(100_000, 1_000_000, n),
        }
    )


def make_params_yaml(path: Path, stocks: list[str] | None = None) -> None:
    content = {"stocks": stocks or []}
    path.write_text(yaml.dump(content))


# ── _download_data ─────────────────────────────────────────────────────────────

class TestPrivateDownloadData:
    """Testa o helper interno _download_data (yfinance mockado)."""

    def _call(self, ticker="PETR4.SA", per="6y"):

        return _download_data(ticker, per)

    def test_retorna_dataframe_com_dados(self):
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = make_ohlcv_df().set_index("Date")

        with patch("src.pipeline.data_ingestion.yf.Ticker", return_value=mock_ticker):
            df = self._call("PETR4.SA")

        assert isinstance(df, pd.DataFrame)
        assert not df.empty

    def test_levanta_value_error_se_df_vazio(self):
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame()

        with patch("src.pipeline.data_ingestion.yf.Ticker", return_value=mock_ticker):
            with pytest.raises(ValueError, match="PETR4.SA"):
                self._call("PETR4.SA")

    def test_history_chamado_com_periodo_correto(self):
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = make_ohlcv_df().set_index("Date")

        with patch("src.pipeline.data_ingestion.yf.Ticker", return_value=mock_ticker):
            self._call("PETR4.SA", "1y")

        mock_ticker.history.assert_called_once_with(period="1y")

    def test_df_tem_coluna_date_apos_reset_index(self):
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = make_ohlcv_df().set_index("Date")

        with patch("src.pipeline.data_ingestion.yf.Ticker", return_value=mock_ticker):
            df = self._call()

        assert "Date" in df.columns


# ── download_data ─────────────────────────────────────────────────────────────

class TestDownloadData:
    """Testa a função pública download_data (merge stock + dólar)."""

    def _call(self, stock="PETR4.SA", periodo="6y"):

        return download_data(stock, periodo)

    def _patch_download(self, stock_df, dolar_df):
        """Retorna side_effect para _download_data de acordo com o ticker."""

        def side_effect(ticker, per="6y"):
            if ticker == "USDBRL=X":
                return dolar_df
            return stock_df

        return side_effect

    def test_retorna_dataframe_com_coluna_dolar(self):
        stock_df = make_ohlcv_df()
        dolar_df = make_ohlcv_df(ticker_col="Close")

        with patch(
            "src.pipeline.data_ingestion._download_data",
            side_effect=self._patch_download(stock_df, dolar_df),
        ):
            df = self._call()

        assert "Dolar" in df.columns

    def test_sem_coluna_close_duplicada_do_dolar(self):
        """A coluna Close do dólar deve aparecer apenas como 'Dolar'."""
        stock_df = make_ohlcv_df()
        dolar_df = make_ohlcv_df(ticker_col="Close")

        with patch(
            "src.pipeline.data_ingestion._download_data",
            side_effect=self._patch_download(stock_df, dolar_df),
        ):
            df = self._call()

        # Não pode haver coluna chamada "Close" proveniente do dólar
        # (o merge usa apenas df_dolar["Dolar"])
        assert df.columns.tolist().count("Close") == 1

    def test_valores_arredondados_2_casas(self):
        stock_df = make_ohlcv_df()
        dolar_df = make_ohlcv_df(ticker_col="Close")

        with patch(
            "src.pipeline.data_ingestion._download_data",
            side_effect=self._patch_download(stock_df, dolar_df),
        ):
            df = self._call()

        numericas = df.select_dtypes(include="number")
        # Verifica que não há mais de 2 casas decimais
        for col in numericas.columns:
            rounded = numericas[col].round(2)
            pd.testing.assert_series_equal(numericas[col], rounded, check_names=False)

    def test_levanta_value_error_se_download_falha(self):
        with patch(
            "src.pipeline.data_ingestion._download_data",
            side_effect=ValueError("erro"),
        ):
            with pytest.raises(ValueError, match="PETR4.SA"):
                self._call("PETR4.SA")

    def test_numero_de_linhas_igual_ao_stock(self):
        stock_df = make_ohlcv_df(n=20)
        dolar_df = make_ohlcv_df(n=20, ticker_col="Close")

        with patch(
            "src.pipeline.data_ingestion._download_data",
            side_effect=self._patch_download(stock_df, dolar_df),
        ):
            df = self._call()

        assert len(df) == 20


# ── add_stock ─────────────────────────────────────────────────────────────────

class TestAddStock:
    def _call(self, stock, params_path):
        # Injeta o caminho do arquivo via monkeypatch no módulo

        original = mod.PARAMS_PATH
        mod.PARAMS_PATH = params_path
        try:
            mod.add_stock(stock)
        finally:
            mod.PARAMS_PATH = original

    def test_adiciona_novo_stock(self, tmp_path):
        params_path = tmp_path / "params.yaml"
        make_params_yaml(params_path, stocks=["VALE3.SA"])

        self._call("PETR4.SA", params_path)

        with open(params_path) as f:
            params = yaml.safe_load(f)

        assert "PETR4.SA" in params["stocks"]

    def test_nao_duplica_stock_existente(self, tmp_path):
        params_path = tmp_path / "params.yaml"
        make_params_yaml(params_path, stocks=["VALE3.SA"])

        self._call("VALE3.SA", params_path)
        self._call("VALE3.SA", params_path)

        with open(params_path) as f:
            params = yaml.safe_load(f)

        assert params["stocks"].count("VALE3.SA") == 1

    def test_preserva_stocks_anteriores(self, tmp_path):
        params_path = tmp_path / "params.yaml"
        make_params_yaml(params_path, stocks=["VALE3.SA", "ITUB4.SA"])

        self._call("PETR4.SA", params_path)

        with open(params_path) as f:
            params = yaml.safe_load(f)

        assert "VALE3.SA" in params["stocks"]
        assert "ITUB4.SA" in params["stocks"]


# ── save_data_raw ─────────────────────────────────────────────────────────────

class TestSaveDataRaw:
    def _call(self, df, stock, monkeypatch, tmp_path):

        # Cria params.yaml temporário
        params_path = tmp_path / "params.yaml"
        make_params_yaml(params_path)
        monkeypatch.setattr(mod, "PARAMS_PATH", params_path)

        # Cria diretório data/raw dentro do tmp
        raw_dir = tmp_path / "data" / "raw"
        raw_dir.mkdir(parents=True)
        monkeypatch.chdir(tmp_path)

        mod.save_data_raw(df, stock)

    def test_salva_csv_no_caminho_correto(self, tmp_path, monkeypatch):
        df = make_ohlcv_df()
        self._call(df, "PETR4.SA", monkeypatch, tmp_path)

        saved_path = tmp_path / "data" / "raw" / "PETR4.SA.csv"
        assert saved_path.exists()

    def test_csv_tem_mesmas_colunas(self, tmp_path, monkeypatch):
        df = make_ohlcv_df()
        self._call(df, "PETR4.SA", monkeypatch, tmp_path)

        loaded = pd.read_csv(tmp_path / "data" / "raw" / "PETR4.SA.csv")
        assert set(loaded.columns) == set(df.columns)

    def test_add_stock_chamado(self, tmp_path, monkeypatch):

        params_path = tmp_path / "params.yaml"
        make_params_yaml(params_path)
        monkeypatch.setattr(mod, "PARAMS_PATH", params_path)

        raw_dir = tmp_path / "data" / "raw"
        raw_dir.mkdir(parents=True)
        monkeypatch.chdir(tmp_path)

        df = make_ohlcv_df()
        mod.save_data_raw(df, "BBAS3.SA")

        with open(params_path) as f:
            params = yaml.safe_load(f)

        assert "BBAS3.SA" in params["stocks"]


# ── recover_data_from_raw ─────────────────────────────────────────────────────

class TestRecoverDataFromRaw:
    def _call(self, stock):

        return recover_data_from_raw(stock)

    def test_leitura_bem_sucedida(self, tmp_path, monkeypatch):
        raw_dir = tmp_path / "data" / "raw"
        raw_dir.mkdir(parents=True)
        make_ohlcv_df().to_csv(raw_dir / "PETR4.SA.csv", index=False)
        monkeypatch.chdir(tmp_path)

        df = self._call("PETR4.SA")
        assert isinstance(df, pd.DataFrame)
        assert "Close" in df.columns

    def test_arquivo_inexistente_levanta_value_error(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError, match="ACAO_FAKE.SA"):
            self._call("ACAO_FAKE.SA")


# ── recover_data_from_processed ───────────────────────────────────────────────

class TestRecoverDataFromProcessed:
    def _call(self, stock):

        return recover_data_from_processed(stock)

    def test_leitura_parquet_bem_sucedida(self, tmp_path, monkeypatch):
        proc_dir = tmp_path / "data" / "processed"
        proc_dir.mkdir(parents=True)
        make_ohlcv_df().to_parquet(proc_dir / "PETR4.SA.parquet", index=False)
        monkeypatch.chdir(tmp_path)

        df = self._call("PETR4.SA")
        assert isinstance(df, pd.DataFrame)
        assert "Close" in df.columns

    def test_arquivo_inexistente_levanta_value_error(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError, match="ACAO_FAKE.SA"):
            self._call("ACAO_FAKE.SA")