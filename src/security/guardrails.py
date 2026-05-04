# src/security/guardrails.py
"""Guardrails de segurança para input e output do agente.

Referência: OWASP Top 10 for LLM Applications (2025)
            https://owasp.org/www-project-top-10-for-large-language-model-applications/
"""
import logging
import re
from presidio_analyzer import PatternRecognizer, Pattern
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
logger = logging.getLogger(__name__)


class InputGuardrail:
    """Valida e sanitiza input do usuário antes de enviar ao LLM."""

    # Padrões comuns de prompt injection
    INJECTION_PATTERNS_PT = [
    r"ignore\s+(todas\s+as\s+)?instruções\s+(anteriores|acima)",
    r"ignore\s+(isso|tudo)\s+e\s+",
    r"esqueça\s+(tudo|todas\s+as\s+instruções|o\s+que\s+foi\s+dito)",
    r"você\s+(agora\s+)?é\s+um(a)?\s+",
    r"aja\s+como\s+um(a)?\s+",
    r"finja\s+ser\s+um(a)?\s+",
    r"se\s+comporte\s+como\s+",
    r"sistema:\s*",
    r"assistente:\s*",
    r"usuário:\s*",
    r"<\|im_start\|>",
    r"\[INST\]",
    r"execute\s+(isso|o\s+código|o\s+comando)",
    r"rode\s+(isso|o\s+código|o\s+comando)",
    r"retorne\s+(dados\s+sensíveis|informações\s+confidenciais)",
    r"mostre\s+(senhas|tokens|chaves|dados\s+privados)",
]

    def __init__(self, allowed_topics: list[str] | None = None):
        self.allowed_topics = allowed_topics or []
        self._compiled_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.INJECTION_PATTERNS_PT
        ]

    def validate(self, user_input: str) -> tuple[bool, str]:
        """Valida input do usuário.

        Args:
            user_input: Texto do usuário.

        Returns:
            Tupla (is_valid, reason).
        """
        # Check 1: Prompt injection detection
        for pattern in self._compiled_patterns:
            if pattern.search(user_input):
                logger.warning("Prompt injection detectado: %s", user_input[:100])
                return False, "Input bloqueado: padrão suspeito detectado."

        # Check 2: Tamanho máximo (evitar context stuffing)
        if len(user_input) > 4096:
            return False, "Input bloqueado: excede tamanho máximo (4096 chars)."

        return True, "OK"


class OutputGuardrail:
    """Valida e sanitiza output do LLM antes de retornar ao usuário."""

    cpf_pattern = Pattern(
        name="CPF",
        regex=r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b",
        score=0.85,
    )

    cpf_recognizer = PatternRecognizer(
        supported_entity="BR_CPF",
        patterns=[cpf_pattern],
    )

    cnpj_patterns = [
        Pattern(
            name="CNPJ (formatado)",
            regex=r"\b\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}\b",
            score=0.85,
        ),
        Pattern(
            name="CNPJ (sem máscara)",
            regex=r"\b\d{14}\b",
            score=0.6,
        ),
    ]

    cnpj_recognizer = PatternRecognizer(
        supported_entity="BR_CNPJ",
        patterns=cnpj_patterns,
    )


    def __init__(self, language: str = "pt"):
        configuration = {
        "nlp_engine_name": "spacy",
        "models": [
            {"lang_code": "pt", "model_name": "pt_core_news_lg"},
        ],
    }

        provider = NlpEngineProvider(nlp_configuration=configuration)
        nlp_engine = provider.create_engine()
        self.analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["pt"])
        self.anonymizer = AnonymizerEngine()
        self.language = language

    def sanitize(self, llm_output: str) -> str:
        """Remove PII do output do LLM.

        Args:
            llm_output: Texto gerado pelo LLM.

        Returns:
            Texto sanitizado.
        """
        

        results  = self.analyzer.analyze(
            text=llm_output,
            language=self.language,
            entities=["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "BR_CPF", "BR_CNPJ"],
        ) 

        if results:
            logger.warning("PII detectado no output: %d entidades", len(results))
            anonymized = self.anonymizer.anonymize(
                text=llm_output,
                analyzer_results=results, # type: ignore
            ) 
            return anonymized.text

        return llm_output