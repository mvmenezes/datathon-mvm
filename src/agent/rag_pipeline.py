# src/agent/rag_pipeline.py
"""Pipeline RAG para análise de ações — Datathon Fase 05.

Componentes:
    - EMBEDDING  : OpenAIEmbeddings — converte texto em vetores
    - RETRIEVER  : Chroma — busca chunks por similaridade
    - GENERATOR  : ChatOpenAI — gera resposta com base no contexto

Uso típico:
    >>> vector_store = build_vector_store(documents)
    >>> retriever, chain = build_rag_chain(vector_store)
    >>> answer, contexts = query_rag_with_context(retriever, chain, "Pergunta")
"""
import json
import logging
import os
from typing import Any
import chromadb
from langchain_chroma import Chroma
import hashlib
import pandas as pd
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()
logger = logging.getLogger(__name__)


EMBEDDING = OpenAIEmbeddings(
        model="text-embedding-3-small",  # Mais barato, suficiente para o Datathon
    )
# ---------------------------------------------------------------------------
# EMBEDDING — Etapa 1: converter dados em Documents indexáveis
# ---------------------------------------------------------------------------

def stock_df_to_documents(df: pd.DataFrame, ticker: str) -> list[Document]:
    """Converte DataFrame de preços históricos em Documents para indexação.

    Cada linha do DataFrame vira um Document com contexto textual descritivo.
    LLMs entendem texto — não números brutos — por isso a conversão é crítica.

    Args:
        df: DataFrame com colunas: date, open, high, low, close, volume.
        ticker: Símbolo da ação (ex: PETR4, VALE3).

    Returns:
        Lista de Documents prontos para indexação no vector store.

    Example:
        >>> df = pd.read_csv("data/processed/stock_data.csv")
        >>> docs = stock_df_to_documents(df, ticker="PETR4")
    """
    required_cols = {"Date","Close","Volume","Dolar","short_mm","medium_mm","large_mm", "RSI", "bb_upper_band", "bb_lower_band"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame faltando colunas: {missing}")

    documents = []
    for _, row in df.iterrows():
        daily_change = (row["Close"] - row["Open"]) / row["Open"] * 100

        fomated_date = pd.to_datetime(row["Date"]).strftime("%d/%m/%Y")
        content = f"""
        No dia {fomated_date}, a ação {ticker} apresentou:
A abertura foi de R$ {row['Open']:.2f} e o fechamento foi de R$ {row['Close']:.2f} (close).
A máxima atingiu R$ {row['High']:.2f} e a mínima foi R$ {row['Low']:.2f}.
O volume negociado foi de {int(row['Volume']):,} ações.
A variação diária foi de {daily_change:.2f}%.
A amplitude do dia foi de R$ {row['High'] - row['Low']:.2f}.
O indicador de RSI foi de {row["RSI"]}
O indicador de Bandas de Bolliner ficou entre {row["bb_lower_band"]:.2f} e {row["bb_upper_band"]:.2f}
O dolar foi de R$ {row["Dolar"]:.2f}
""".strip()

        documents.append(Document(
            page_content=content,
            metadata={
                "ticker": ticker,
                "date": str(row["Date"]),
                "source": "historical_prices",
                "daily_change_pct": round(daily_change, 4),
            },
        ))

    logger.info(
        "Convertidos %d registros de %s em Documents",
        len(documents),
        ticker,
    )
    return documents

def stock_news_to_documents(content: str, ticker: str, date: str) -> list[Document]:


    documents = []


    documents.append(Document(
            page_content=content,
            metadata={
                "ação": ticker,
                "data": date
            },
        ))

    logger.info(
        f"Notícia sobre {ticker} convertida em Document | data: {date}")
    return documents


def news_to_documents(news_list: list[dict], ticker: str) -> list[Document]:
    """Converte lista de notícias em Documents para indexação.

    Args:
        news_list: Lista de dicts com chaves: date, headline, summary.
        ticker: Símbolo da ação associada.

    Returns:
        Lista de Documents de notícias.

    Example:
        >>> news = [{"date": "2024-01-10", "headline": "...", "summary": "..."}]
        >>> docs = news_to_documents(news, ticker="PETR4")
    """
    documents = []
    for item in news_list:
        content = f"""
Notícia sobre: {ticker}
Data: {item.get('date', 'N/A')}
Título: {item.get('headline', '')}
Resumo: {item.get('summary', '')}
""".strip()

        documents.append(Document(
            page_content=content,
            metadata={
                "ticker": ticker,
                "date": str(item.get("date", "")),
                "source": "news",
            },
        ))

    logger.info("Convertidas %d notícias de %s em Documents", len(documents), ticker)
    return documents


# ---------------------------------------------------------------------------
# EMBEDDING — Etapa 2: indexar Documents no vector store
# ---------------------------------------------------------------------------

def build_vector_store(
    collection_name: str = "langchain",
    persist_dir: str = "./data/chroma_db"
) -> Chroma:
    """Indexa Documents no Chroma com embeddings da OpenAI.

    Divide documentos em chunks, gera embeddings e persiste em disco.
    Executar apenas uma vez — depois carregar com load_vector_store().

    Args:
        documents: Lista de Documents convertidos (preços + notícias).
        persist_dir: Diretório para persistir o índice Chroma.
        chunk_size: Tamanho máximo de cada chunk em caracteres.
        chunk_overlap: Sobreposição entre chunks (evita perda de contexto).

    Returns:
        Vector store Chroma pronto para queries.

    Example:
        >>> docs = stock_df_to_documents(df, "PETR4")
        >>> vector_store = build_vector_store(docs)
    """


    vector_store = Chroma(
    persist_directory=persist_dir,
    embedding_function=EMBEDDING,
    collection_name=collection_name)


    logger.info("Vector store criado com sucesso em %s", persist_dir)
    return vector_store


def load_vector_store(persist_dir: str = "./data/chroma_db", collection_name: str = "langchain") -> Chroma:
    """Carrega vector store já existente em disco.

    Use após a primeira indexação — evita reindexar desnecessariamente.

    Args:
        persist_dir: Diretório onde o índice Chroma foi persistido.

    Returns:
        Vector store Chroma carregado.

    Example:
        >>> vector_store = load_vector_store()
    """

    vector_store = Chroma(
        persist_directory=persist_dir,
        embedding_function=EMBEDDING,
        collection_name=collection_name,
    )

    logger.info("Vector store carregado de %s", persist_dir)
    return vector_store


def upsert_documents(
    new_documents: list[Document], collection_name: str = "langchain",
) -> None:
    """Adiciona novos Documents ao vector store de forma incremental.

    Args:
        vector_store: Vector store Chroma existente.
        new_documents: Novos Documents a indexar.

    Example:
        >>> upsert_documents(vector_store, novos_docs)
    """

    index_documents_safe(new_documents, collection_name )
    logger.info("Upsert: %d novos Documents adicionados ao vector store", len(new_documents))


# ---------------------------------------------------------------------------
# RETRIEVER + GENERATOR — Etapa 3: construir a chain RAG completa
# ---------------------------------------------------------------------------

def build_rag_chain(
    vector_store: Chroma,
    k: int = 4,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0
) -> tuple[Any, Any]:
    """Constrói a chain RAG completa: retriever + prompt + LLM.

    Args:
        vector_store: Vector store Chroma já indexado.
        k: Número de chunks a recuperar por query.
        model_name: Modelo OpenAI a usar como generator.
        temperature: Temperatura do LLM (0.0 = determinístico e auditável).

    Returns:
        Tupla (retriever, rag_chain) para uso em queries e avaliação RAGAS.

    Example:
        >>> retriever, chain = build_rag_chain(vector_store)
    """
    # RETRIEVER — busca por similaridade semântica
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )

    # Prompt otimizado para análise de ações com dados históricos
    prompt = ChatPromptTemplate.from_template("""
        Você é um analista financeiro especializado em renda variável brasileira.
        Responda com base EXCLUSIVAMENTE nos dados fornecidos abaixo.
        Se os dados não forem suficientes para responder com precisão, diga claramente.
        Nunca invente valores, datas ou tendências que não estejam no contexto.
        Responda de forma objetiva, sem explicações longas.
        Dados históricos recuperados:
        {context}

        Pergunta do analista: {question}
""")

    # GENERATOR — LLM que lê contexto e gera resposta
    llm = ChatOpenAI(
        model=model_name,
        temperature=temperature,
    )

    def format_docs(docs: list[Document]) -> str:
        """Formata lista de Documents em texto único para o prompt."""
        return "\n\n---\n\n".join(doc.page_content for doc in docs)

    # Chain completa com LCEL (LangChain Expression Language)
    rag_chain = (
        RunnableParallel({
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        })
        | prompt
        | llm
        | StrOutputParser()
    )

    logger.info(
        "Chain RAG construída: k=%d, model=%s, temperature=%.1f",
        k,
        model_name,
        temperature,
    )
    return retriever, rag_chain


# ---------------------------------------------------------------------------
# Interface principal — query com contextos para RAGAS
# ---------------------------------------------------------------------------

def query_rag_with_context(
    retriever: Any,
    rag_chain: Any,
    question: str,
    collection_name: str = "langchain",
) -> tuple[str, list[str]]:
    """Executa query no pipeline RAG e retorna resposta + contextos usados.

    Retornar os contextos é obrigatório para calcular context_precision
    e context_recall no RAGAS (Etapa 3 do Datathon).

    Args:
        retriever: Retriever do vector store.
        rag_chain: Chain RAG completa.
        question: Pergunta em linguagem natural.

    Returns:
        Tupla (resposta_str, lista_de_contextos_str).

    Example:
        >>> answer, contexts = query_rag_with_context(retriever, chain, "Pergunta")
        >>> print(answer)
        >>> print(f"Contextos usados: {len(contexts)}")
    """
    # Recupera chunks separadamente para capturar os contextos
    if collection_name != "langchain":
        docs_retrieved = retriever.invoke(question)
    else:
        docs_retrieved = search_all_collections(question, persist_dir="./data/chroma_db", k=4)

    contexts = [doc.page_content for doc in docs_retrieved]

    # Gera a resposta com a chain completa
    answer = rag_chain.invoke(question)

    logger.info(
        "Query processada | contextos=%d | pergunta='%s...'",
        len(contexts),
        question[:60],
    )
    return answer, contexts


# ---------------------------------------------------------------------------
# Utilitário — carregar golden set para RAGAS
# ---------------------------------------------------------------------------

def load_golden_set(golden_set_path: str) -> list[dict]:
    """Carrega o golden set para avaliação RAGAS.

    Args:
        golden_set_path: Caminho para JSON com lista de pares
                         {"query": ..., "expected_answer": ...}.

    Returns:
        Lista de dicts com os pares de avaliação.

    Example:
        >>> golden_set = load_golden_set("data/golden_set/golden_set.json")
    """
    if not os.path.exists(golden_set_path):
        raise FileNotFoundError(
            f"Golden set não encontrado: {golden_set_path}\n"
            "Crie pelo menos 20 pares query/expected_answer."
        )

    with open(golden_set_path, encoding="utf-8") as f:
        golden_set = json.load(f)

    if len(golden_set) < 20:
        logger.warning(
            "Golden set com apenas %d pares. Datathon exige >= 20.",
            len(golden_set),
        )

    logger.info("Golden set carregado: %d pares", len(golden_set))
    return golden_set

def get_or_create_vector_store(collection_name: str = "langchain",
    persist_dir: str = "./data/chroma_db",
) -> Chroma:
    """Carrega o store se já existe, cria do zero se não existe.

    Nunca chame build_vector_store manualmente — use esta função.

    Args:
        documents: Docs para indexar (usado apenas na criação).
        persist_dir: Diretório do índice Chroma.

    Returns:
        Vector store pronto para uso.
    """
    
    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        # Store já existe — só carrega
        logger.info("Vector store encontrado em %s — carregando.", persist_dir)
        return load_vector_store(persist_dir,collection_name)

    else:
        # Primeira vez — cria do zero
        logger.info("Vector store não encontrado — criando em %s.", persist_dir)
        return build_vector_store(persist_dir=persist_dir, collection_name=collection_name)
    
def run_pipeline(input: str) -> str:
    """Executa o Pipeline RAG com a entrada do usuário.

    Args:
        input: Pergunta ou comando do usuário.

    Returns:
        Resposta gerada pelo pipeline.

    Example:
        >>> answer = run_agent("Qual é a previsão para PETR4 na próxima semana?")
        >>> print(answer)
    """

    vector_store = get_or_create_vector_store()
    retriever, rag_chain = build_rag_chain(vector_store)
    answer, _ = query_rag_with_context(retriever, rag_chain, input)
    return answer


def run_pipeline_llm_judge(input: str):
    """Executa o Pipeline RAG com a entrada do usuário.

    Args:
        input: Pergunta ou comando do usuário.

    Returns:
        Resposta gerada pelo pipeline.

    Example:
        >>> answer = run_agent("Qual é a previsão para PETR4 na próxima semana?")
        >>> print(answer)
    """

    vector_store = get_or_create_vector_store()
    retriever, rag_chain = build_rag_chain(vector_store)
    answer, contexts = query_rag_with_context(retriever, rag_chain, input)
    return   answer, contexts   



def search_all_collections(
    query: str,
    persist_dir: str,
    k: int = 4,
) -> list[Document]:
    """Busca em todas as collections e retorna os melhores resultados."""

    all_results = []

    client = chromadb.PersistentClient(path=persist_dir)
    collections = client.list_collections()
    for col in collections:
        print(f"Buscando na collection: {col.name}")
        vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=EMBEDDING,
            collection_name=col.name,
        )
        results = vectorstore.similarity_search_with_score(query, k=k)
        print(f"  Resultados encontrados: {len(results)}")
        print(results)
        for doc, score in results:
            doc.metadata["collection"] = col.name  # rastreia a origem
            doc.metadata["score"] = score
            all_results.append((doc, score))

    # ordena por score (menor distância = mais relevante no ChromaDB)
    all_results.sort(key=lambda x: x[1])

    # retorna os k melhores do total
    top_docs = [doc for doc, _ in all_results[:k]]
    return top_docs





#-------------------------------------------------------------------------------------------------




def get_chunk_id(chunk) -> str:
    """Gera id único baseado no conteúdo + fonte + página."""
    content = f"{chunk.metadata.get('source', '')}-{chunk.metadata.get('page', 0)}-{chunk.page_content}"
    return hashlib.md5(content.encode(), usedforsecurity=False).hexdigest()

def index_documents_safe(
    documents: list,
    collection_name: str,
    chunk_size: int = 512,
    chunk_overlap: int = 0,
) -> Chroma:
    """Indexa documentos sem duplicar — idempotente."""

    vectorstore = get_or_create_vector_store(collection_name=collection_name)

    # ids já existentes no índice
    existing = set(vectorstore.get()["ids"])
    print(f"Chunks já indexados: {len(existing)}")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(documents)

    logger.info(
        "Indexando %d chunks no Chroma (%s)...",
        len(chunks),
        collection_name,
    )
    # filtra só os novos
    new_chunks = []
    new_ids    = []
    for chunk in chunks:
        chunk_id = get_chunk_id(chunk)
        if chunk_id not in existing:
            new_chunks.append(chunk)
            new_ids.append(chunk_id)

    if not new_chunks:
        print("Nenhum chunk novo — índice já atualizado.")
        return vectorstore

    print(f"Indexando {len(new_chunks)} chunks novos...")
    vectorstore.add_documents(documents=new_chunks, ids=new_ids)
    print(f"Total agora: {vectorstore._collection.count()} chunks")

    return vectorstore


def debug_rag_pipeline(retriever, rag_chain, question: str, expected_answer: str):
    """Diagnóstico em camadas — identifica onde o RAG está falhando."""

    print(f"\n{'='*60}")
    print(f"QUERY: {question}")
    print(f"EXPECTED: {expected_answer}")
    print(f"{'='*60}")

    # Camada 1: o retriever encontrou algo?
    docs = search_all_collections(question, persist_dir="./data/chroma_db",  k=4)
    print(f"\n[RETRIEVER] {len(docs)} chunks retornados:")
    for i, doc in enumerate(docs):
        print(f"  [{i+1}] score={getattr(doc, 'score', 'N/A')} | {doc.page_content[:120]}...")

    if not docs:
        print("  !! PROBLEMA: retriever retornou 0 documentos")
        print("  -> Verifique se os documentos foram indexados corretamente")
        return

    # Camada 2: o conteúdo esperado está em algum chunk?
    keywords = expected_answer.lower().split()[:5]  # primeiras 5 palavras do ground truth
    found_in_chunks = any(
        any(kw in doc.page_content.lower() for kw in keywords)
        for doc in docs
    )
    print(f"\n[CONTEÚDO] Keywords do ground truth encontradas nos chunks: {found_in_chunks}")
    if not found_in_chunks:
        print("  !! PROBLEMA: retriever retornou chunks, mas nenhum contém a informação esperada")
        print("  -> Problema de chunking ou de embedding — veja Passo 2")

    # Camada 3: o gerador usou o contexto?
    answer = rag_chain.invoke(question)
    print(f"\n[GERADOR] Resposta produzida:\n  {answer}")
    print("\n[DIAGNÓSTICO FINAL]")
    print(f"  Retriever achou contexto relevante : {found_in_chunks}")
    print(f"  Resposta gerada                    : {'OK' if answer else 'VAZIA'}")

# ---------------------------------------------------------------------------
# Exemplo de uso completo (executar diretamente para testar)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)


    vector_store = get_or_create_vector_store(collection_name="PETR4.SA")

    # 4. Constrói chain (RETRIEVER + GENERATOR)
    retriever, chain = build_rag_chain(vector_store,k=4)
    debug_rag_pipeline(retriever, chain, "Qual foi o preço de fechamento de PETR4.SA no dia 10/06/2025?", 
                       "O fechamento de PETR4 no dia 10/06/2025 foi de R$28,11.")
