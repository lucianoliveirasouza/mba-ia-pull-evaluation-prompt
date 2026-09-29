"""
Script para fazer pull de prompts do LangSmith Prompt Hub.

Este script:
1. Carrega as variáveis de ambiente do arquivo .env
2. Valida as credenciais necessárias
3. Conecta ao LangSmith
4. Faz pull do prompt público:
      leonanluppi/bug_to_user_story_v1
5. Extrai as mensagens do ChatPromptTemplate
6. Converte o prompt para o formato YAML esperado pelo projeto
7. Salva em:
      prompts/bug_to_user_story_v1.yml

Observação:
O parâmetro dangerously_pull_public_prompt=True é obrigatório para
fazer pull de um prompt público identificado como "owner/nome".
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langsmith import Client

from utils import save_yaml, check_env_vars, print_section_header


# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

load_dotenv()

PROMPT_NAME = "leonanluppi/bug_to_user_story_v1"

OUTPUT_FILE = (
    Path(__file__).resolve().parent.parent
    / "prompts"
    / "bug_to_user_story_v1.yml"
)


# ---------------------------------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------------------------------

def extract_message_template(message):
    """
    Extrai o template de uma mensagem do ChatPromptTemplate.

    Normalmente as mensagens do LangChain possuem uma estrutura semelhante a:

        message.prompt.template

    Porém, algumas versões do LangChain podem apresentar pequenas
    diferenças na estrutura. Por isso fazemos algumas tentativas.

    Args:
        message: objeto de mensagem/template retornado pelo LangChain.

    Returns:
        str: conteúdo do template.

    Raises:
        ValueError: caso não seja possível extrair o template.
    """

    # Estrutura esperada pelo exercício:
    # message.prompt.template
    if hasattr(message, "prompt") and hasattr(message.prompt, "template"):
        return message.prompt.template

    # Algumas versões podem expor "template" diretamente.
    if hasattr(message, "template"):
        return message.template

    raise ValueError(
        f"Não foi possível extrair o template da mensagem. "
        f"Tipo recebido: {type(message)}"
    )


def convert_prompt_to_yaml(prompt):
    """
    Converte um ChatPromptTemplate retornado pelo LangSmith
    para o formato YAML utilizado pelo exercício.

    Args:
        prompt: ChatPromptTemplate retornado pelo client.pull_prompt()

    Returns:
        dict: estrutura pronta para save_yaml()
    """

    if not hasattr(prompt, "messages"):
        raise ValueError(
            "O objeto retornado pelo LangSmith não possui o atributo "
            "'messages'. O prompt retornado não parece ser um "
            "ChatPromptTemplate."
        )

    messages = prompt.messages

    if not messages:
        raise ValueError(
            "O prompt retornado pelo LangSmith não possui mensagens."
        )

    system_prompt = ""
    user_prompt = ""

    for message in messages:
        template = extract_message_template(message)

        # Identifica o tipo da mensagem.
        message_type = getattr(message, "type", "").lower()

        if message_type == "system":
            if system_prompt:
                system_prompt += "\n\n"

            system_prompt += template

        elif message_type in ("human", "user"):
            if user_prompt:
                user_prompt += "\n\n"

            user_prompt += template

        else:
            # Caso o LangChain não exponha o tipo como esperado,
            # tentamos identificar pela classe.
            class_name = message.__class__.__name__.lower()

            if "system" in class_name:
                if system_prompt:
                    system_prompt += "\n\n"

                system_prompt += template

            elif "human" in class_name:
                if user_prompt:
                    user_prompt += "\n\n"

                user_prompt += template

    # Alguns prompts podem possuir somente uma mensagem.
    # Caso não tenhamos identificado system/user pelo type,
    # usamos a primeira mensagem como system_prompt.
    if not system_prompt and not user_prompt:
        system_prompt = extract_message_template(messages[0])

    elif not system_prompt:
        system_prompt = user_prompt
        user_prompt = ""

    return {
        "bug_to_user_story_v1": {
            "description": "Prompt para converter relatos de bugs em User Stories",
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "version": "v1",
            "created_at": "2025-01-15",
            "tags": [
                "bug-analysis",
                "user-story",
                "product-management",
            ],
        }
    }


# ---------------------------------------------------------------------------
# Pull do LangSmith
# ---------------------------------------------------------------------------

def pull_prompts_from_langsmith():
    """
    Faz o pull do prompt v1 do LangSmith Prompt Hub
    e salva o resultado localmente.

    Returns:
        bool: True se o processo foi concluído com sucesso.
    """

    print_section_header("PULL DO PROMPT DO LANGSMITH")

    # -----------------------------------------------------------------------
    # 1. Validar variáveis de ambiente
    # -----------------------------------------------------------------------

    required_vars = [
        "LANGSMITH_API_KEY",
    ]

    if not check_env_vars(required_vars):
        return False

    # -----------------------------------------------------------------------
    # 2. Criar cliente LangSmith
    # -----------------------------------------------------------------------

    print("Conectando ao LangSmith...")

    try:
        client = Client()
        print("CLIENT: Cliente LangSmith criado com sucesso.")

    except Exception as exc:
        print("ERROR: Não foi possível criar o cliente LangSmith.")
        print(f"   Erro: {exc}")
        return False

    # -----------------------------------------------------------------------
    # 3. Fazer pull do prompt
    # -----------------------------------------------------------------------

    print()
    print(f"Fazendo pull do prompt:")
    print(f"   {PROMPT_NAME}")
    print()

    try:
        prompt = client.pull_prompt(
            PROMPT_NAME,
            dangerously_pull_public_prompt=True,
        )

        print("CLIENT: Prompt carregado com sucesso.")
        print(f"  Tipo retornado: {type(prompt).__name__}")

    except Exception as exc:
        print()
        print("=" * 70)
        print("ERROR ERRO AO FAZER PULL DO PROMPT")
        print("=" * 70)
        print()
        print(f"Prompt: {PROMPT_NAME}")
        print(f"Erro: {exc}")
        print()
        print("Verifique:")
        print("1. LANGSMITH_API_KEY está correta.")
        print("2. Sua conta possui acesso ao LangSmith.")
        print("3. Existe conexão com a Internet.")
        print("4. O prompt público existe:")
        print(f"   {PROMPT_NAME}")
        print()
        print(
            "5. O parâmetro "
            "dangerously_pull_public_prompt=True "
            "está sendo utilizado."
        )
        print()
        return False

    # -----------------------------------------------------------------------
    # 4. Converter o prompt para o formato do projeto
    # -----------------------------------------------------------------------

    print("Convertendo prompt para formato YAML...")

    try:
        yaml_data = convert_prompt_to_yaml(prompt)

    except Exception as exc:
        print()
        print("=" * 70)
        print("ERROR ERRO AO CONVERTER O PROMPT")
        print("=" * 70)
        print()
        print(f"Erro: {exc}")
        print()
        print(
            "Dica: o LangSmith retornou um objeto diferente do "
            "esperado pelo script."
        )
        print()
        return False

    # -----------------------------------------------------------------------
    # 5. Salvar YAML
    # -----------------------------------------------------------------------

    print(f"Salvando prompt em:")
    print(f"   {OUTPUT_FILE}")

    try:
        success = save_yaml(
            yaml_data,
            str(OUTPUT_FILE),
        )

    except Exception as exc:
        print()
        print("ERROR Erro ao salvar o arquivo YAML.")
        print(f"   {exc}")
        return False

    if not success:
        print()
        print("ERROR O arquivo não pôde ser salvo.")
        return False

    # -----------------------------------------------------------------------
    # 6. Resultado
    # -----------------------------------------------------------------------

    print()
    print("=" * 70)
    print("PULL CONCLUIDO COM SUCESSO")
    print("=" * 70)
    print()
    print(f"Prompt remoto:")
    print(f"  {PROMPT_NAME}")
    print()
    print(f"Arquivo local:")
    print(f"  {OUTPUT_FILE}")
    print()

    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """
    Função principal do script.

    Returns:
        int:
            0 = sucesso
            1 = erro
    """

    try:
        success = pull_prompts_from_langsmith()

        return 0 if success else 1

    except KeyboardInterrupt:
        print()
        print("WARNING Operacao interrompida pelo usuario.")
        return 1

    except Exception as exc:
        print()
        print("=" * 70)
        print("ERROR ERRO NAO TRATADO")
        print("=" * 70)
        print()
        print(f"{type(exc).__name__}: {exc}")
        print()

        return 1


if __name__ == "__main__":
    sys.exit(main())