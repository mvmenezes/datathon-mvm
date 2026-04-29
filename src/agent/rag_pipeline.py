import feedparser
import requests
from bs4 import BeautifulSoup
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging

logger = logging.getLogger(__name__)

# Feeds RSS financeiros
RSS_FEEDS = {
    "infomoney": "https://www.infomoney.com.br/feed/",
    "valor": "https://valor.globo.com/rss/financas/",
    "investing_br": "https://br.investing.com/rss/news.rss",
}

def extrair_texto_noticia(url: str) -> str | None:
    """Extrai texto completo de uma notícia via scraping."""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Remove elementos desnecessários
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        
        paragrafos = soup.find_all("p")
        texto = " ".join([p.get_text() for p in paragrafos])
        
        return texto if len(texto) > 200 else None
        
    except Exception as e:
        logger.warning("Falha ao extrair %s: %s", url, e)
        return None


def coletar_noticias(ticker: str, n_noticias: int = 20) -> list[Document]:
    """Coleta notícias relevantes para o ticker via RSS."""
    
    documentos = []
    
    for fonte, url_feed in RSS_FEEDS.items():
        feed = feedparser.parse(url_feed)
        
        for entry in feed.entries[:10]:
            titulo = entry.get("title", "")
            
            # Filtra por relevância ao ticker
            if ticker.replace(".SA","").upper() not in titulo.upper():
                continue
            
            url_noticia = entry.get("link", "")
            texto = extrair_texto_noticia(url_noticia)
            
            if not texto:
                continue
            
            doc = Document(
                page_content=texto,
                metadata={
                    "titulo": titulo,
                    "fonte": fonte,
                    "url": url_noticia,
                    "data": entry.get("published", ""),
                    "ticker": ticker,
                }
            )
            documentos.append(doc)
            
            if len(documentos) >= n_noticias:
                break
    
    logger.info(
        "Coletadas %d notícias para %s",
        len(documentos), ticker
    )
    return documentos


def construir_vectorstore(ticker: str) -> FAISS:
    """Coleta notícias, indexa e retorna vector store."""
    
    # Coleta documentos
    docs = coletar_noticias(ticker)
    
    if not docs:
        raise ValueError(f"Nenhuma notícia encontrada para {ticker}")
    
    # Chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=64,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(docs)
    
    # Indexação
    vectorstore = FAISS.from_documents(
        chunks,
        OpenAIEmbeddings()
    )
    
    logger.info(
        "Vector store construído: %d chunks para %s",
        len(chunks), ticker
    )
    return vectorstore