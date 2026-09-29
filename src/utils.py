"""
Funções auxiliares para o projeto de otimização de prompts.
"""

import os
import yaml
import json
from typing import Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def load_yaml(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Carrega arquivo YAML.

    Args:
        file_path: Caminho do arquivo YAML

    Returns:
        Dicionário com conteúdo do YAML ou None se erro
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return data
    except FileNotFoundError:
        print(f"ERROR Arquivo nao encontrado: {file_path}")
        return None
    except yaml.YAMLError as e:
        print(f"ERROR Erro ao parsear YAML: {e}")
        return None
    except Exception as e:
        print(f"ERROR Erro ao carregar arquivo: {e}")
        return None


def save_yaml(data: Dict[str, Any], file_path: str) -> bool:
    """
    Salva dados em arquivo YAML.

    Args:
        data: Dados para salvar
        file_path: Caminho do arquivo de saída

    Returns:
        True se sucesso, False caso contrário
    """
    try:
        output_file = Path(file_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, allow_unicode=True, sort_keys=False, indent=2)

        return True
    except Exception as e:
        print(f"ERROR Erro ao salvar arquivo: {e}")
        return False


def check_env_vars(required_vars: list) -> bool:
    """
    Verifica se variáveis de ambiente obrigatórias estão configuradas.

    Args:
        required_vars: Lista de variáveis obrigatórias

    Returns:
        True se todas configuradas, False caso contrário
    """
    missing_vars = []

    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)

    if missing_vars:
        print("ERROR Variaveis de ambiente faltando:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nConfigure-as no arquivo .env antes de continuar.")
        return False

    return True


def format_score(score: float, threshold: float = 0.8) -> str:
    """
    Formata score com indicador visual de aprovação.

    Args:
        score: Score entre 0.0 e 1.0
        threshold: Limite mínimo para aprovação

    Returns:
        String formatada com score e símbolo
    """
    symbol = "OK" if score >= threshold else "FAIL"
    return f"{score:.2f} {symbol}"


def print_section_header(title: str, char: str = "=", width: int = 50):
    """
    Imprime cabeçalho de seção formatado.

    Args:
        title: Título da seção
        char: Caractere para a linha
        width: Largura da linha
    """
    print("\n" + char * width)
    print(title)
    print(char * width + "\n")


def validate_prompt_structure(prompt_data: Dict[str, Any]) -> tuple[bool, list]:
    """
    Valida estrutura básica de um prompt.

    Args:
        prompt_data: Dados do prompt

    Returns:
        (is_valid, errors) - Tupla com status e lista de erros
    """
    errors = []

    required_fields = ['description', 'system_prompt', 'version']
    for field in required_fields:
        if field not in prompt_data:
            errors.append(f"Campo obrigatório faltando: {field}")

    system_prompt = prompt_data.get('system_prompt', '').strip()
    if not system_prompt:
        errors.append("system_prompt está vazio")

    if 'TODO' in system_prompt:
        errors.append("system_prompt ainda contém TODOs")

    techniques = prompt_data.get('techniques_applied', [])
    if len(techniques) < 2:
        errors.append(f"Mínimo de 2 técnicas requeridas, encontradas: {len(techniques)}")

    return (len(errors) == 0, errors)


def extract_json_from_response(response_text: str) -> Optional[Dict[str, Any]]:
    """
    Extrai JSON de uma resposta de LLM que pode conter texto adicional.

    Args:
        response_text: Texto da resposta do LLM

    Returns:
        Dicionário extraído ou None se não encontrar JSON válido
    """
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        start = response_text.find('{')
        end = response_text.rfind('}') + 1

        if start != -1 and end > start:
            try:
                json_str = response_text[start:end]
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass

    return None


def get_llm(model: Optional[str] = None, temperature: float = 0.0):
    """
    Retorna uma instância de LLM configurada baseada no provider.

    Args:
        model: Nome do modelo (opcional, usa LLM_MODEL do .env por padrão)
        temperature: Temperatura para geração (padrão: 0.0 para determinístico)

    Returns:
        Instância de ChatOpenAI ou ChatGoogleGenerativeAI

    Raises:
        ValueError: Se provider não for suportado ou API key não configurada
    """
    provider = os.getenv('LLM_PROVIDER', 'openai').lower()
    model_name = model or os.getenv('LLM_MODEL', '')

    if not model_name:
        raise ValueError(
            "Nenhum modelo definido.\n"
            "Preencha LLM_MODEL (e EVAL_MODEL) no .env com um modelo disponível\n"
            "na documentação oficial do provider escolhido:\n"
            "  Google -> https://ai.google.dev/gemini-api/docs/models\n"
            "  OpenAI -> https://platform.openai.com/docs/models"
        )

    if provider == 'openai':
        from langchain_openai import ChatOpenAI

        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY não configurada no .env\n"
                "Obtenha uma chave em: https://platform.openai.com/api-keys"
            )

        # Sanity check: Google Gemini model IDs (e.g., 'gemini-*') are not
        # valid on the OpenAI provider and will cause runtime model-not-found
        # errors. Detect accidental mismatches early and provide guidance.
        mn = (model_name or "").lower()
        if "gemini" in mn:
            raise ValueError(
                f"Modelo '{model_name}' parece ser um modelo Google Gemini, mas 'LLM_PROVIDER' está configurado como 'openai'.\n"
                "Opções para resolver:\n"
                "  - Use um modelo OpenAI (ex.: 'gpt-3.5-turbo') definindo 'LLM_MODEL' apropriadamente.\n"
                "  - Ou altere 'LLM_PROVIDER=google' e configure 'GOOGLE_API_KEY' se pretende usar Gemini."
            )

        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=api_key
        )

    elif provider == 'google':
        from langchain_google_genai import ChatGoogleGenerativeAI

        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY não configurada no .env\n"
                "Obtenha uma chave em: https://aistudio.google.com/app/apikey"
            )

        # Sanity check: common OpenAI-style model IDs are not supported by
        # the Google GenAI API (they cause hard 404s at runtime). Detect
        # accidental mismatches and raise a clear error before making API calls.
        mn = (model_name or "").lower()
        if mn.startswith("gpt-") or "gpt-" in mn:
            raise ValueError(
                f"Modelo '{model_name}' parece ser um modelo OpenAI, mas 'LLM_PROVIDER' está configurado como 'google'.\n"
                "Opções para resolver:\n"
                "  - Use um modelo compatível com Google definindo 'LLM_MODEL' para um modelo GenAI (ex.: 'gemini-*') e configure 'GOOGLE_API_KEY'.\n"
                "  - Ou altere 'LLM_PROVIDER=openai' e configure 'OPENAI_API_KEY' no .env se pretende usar modelos OpenAI."
            )

        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            google_api_key=api_key
        )

    else:
        raise ValueError(
            f"Provider '{provider}' não suportado.\n"
            f"Use 'openai' ou 'google' na variável LLM_PROVIDER do .env"
        )


def get_eval_llm(temperature: float = 0.0):
    """
    Retorna LLM configurado especificamente para avaliação (usa EVAL_MODEL).

    Args:
        temperature: Temperatura para geração

    Returns:
        Instância de LLM configurada para avaliação
    """
    eval_model = os.getenv('EVAL_MODEL', '')
    return get_llm(model=eval_model, temperature=temperature)
