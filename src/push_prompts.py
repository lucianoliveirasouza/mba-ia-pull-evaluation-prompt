"""Push prompt utility for LangSmith with multiple-SDK-signature support.

This script attempts to publish `prompts/bug_to_user_story_v2.yml` to the
LangSmith Prompt Hub by trying several known SDK method signatures. If all
attempts fail it prints clear manual steps to publish via the LangSmith UI.
"""

import os
import yaml
import inspect
from dotenv import load_dotenv

load_dotenv()


def load_prompt_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    data = yaml.safe_load(raw)
    return data, raw


def try_publish_via_sdk(raw: str, name: str, data: dict) -> bool:
    """Try multiple SDK call signatures to publish the prompt.

    Returns True if any call succeeded.
    """
    tried = []
    try:
        from langsmith import Client
    except Exception as e:
        print("LangSmith SDK não disponível:", e)
        return False

    from langchain_core.prompts import ChatPromptTemplate
    # Import concrete message templates for reliable construction
    try:
        from langchain_core.prompts.chat import (
            SystemMessagePromptTemplate,
            HumanMessagePromptTemplate,
        )
    except Exception:
        # Fallback names if package layout differs
        from langchain_core.prompts import (
            SystemMessagePromptTemplate,
            HumanMessagePromptTemplate,
        )

    client = Client()

    # Build a ChatPromptTemplate from YAML fields when possible
    system_prompt = data.get("system_prompt")
    user_prompt = data.get("user_prompt")

    messages = []
    if system_prompt:
        messages.append(("system", system_prompt))
    if user_prompt:
        messages.append(("user", user_prompt))

    # Build ChatPromptTemplate using explicit message templates
    try:
        block = []
        for role, text in messages:
            if role == "system":
                block.append(SystemMessagePromptTemplate.from_template(text))
            else:
                # treat 'user' and others as human messages
                block.append(HumanMessagePromptTemplate.from_template(text))
        chat_prompt = ChatPromptTemplate.from_messages(block) if block else None
    except Exception as e:
        print("WARNING: could not build ChatPromptTemplate:", e)
        chat_prompt = None

    # Preferred: Client.push_prompt(prompt_identifier, object=..., is_public=True, description=...)
    try:
        if chat_prompt is not None and hasattr(client, "push_prompt"):
            res = client.push_prompt(name, object=chat_prompt, is_public=True, description=data.get("metadata", {}).get("description"))
            print("client.push_prompt returned:", res)
            return True
    except Exception as e:
        tried.append(("push_prompt(object)", [], list(["object","is_public"]), str(e)))

    # Fallback: create_prompt then push a commit
    try:
        if hasattr(client, "create_prompt"):
            res = client.create_prompt(name, description=data.get("metadata", {}).get("description"), is_public=True)
            print("client.create_prompt returned:", getattr(res, "id", str(res)))
            return True
    except Exception as e:
        tried.append(("create_prompt", [name], [], str(e)))

    # Last resort: try lower-level push with raw YAML as object
    try:
        if hasattr(client, "push_prompt"):
            res = client.push_prompt(name, object=raw, is_public=True, description=data.get("metadata", {}).get("description"))
            print("client.push_prompt(raw) returned:", res)
            return True
    except Exception as e:
        tried.append(("push_prompt(raw)", [], ["object"], str(e)))

    print("Todas as tentativas via SDK falharam. Detalhes:")
    for t in tried:
        print(f" - func={t[0]} args={t[1]} kwargs={t[2]} error={t[3]}")

    return False


def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    prompt_path = os.path.join(repo_root, "prompts", "bug_to_user_story_v2.yml")

    if not os.path.exists(prompt_path):
        print("Arquivo de prompt não encontrado:", prompt_path)
        return 1

    data, raw = load_prompt_yaml(prompt_path)
    handle = os.getenv("USERNAME_LANGSMITH_HUB", "your_handle")
    name = f"{handle}/bug_to_user_story_v2"

    print(f"Preparado para publicar: {name}")
    print("Resumo metadata:", data.get("metadata", {}))

    success = try_publish_via_sdk(raw, name, data)

    if success:
        print("Prompt publicado com sucesso via SDK.")
        return 0

    print("\nPush automático não funcionou. Para publicar manualmente siga: \n")
    print("1) Abra https://smith.langchain.com e faça login na sua conta")
    print("2) Vá em Prompts → Create Prompt")
    print("3) Copie o conteúdo do arquivo 'prompts/bug_to_user_story_v2.yml' e cole no editor")
    print(f"4) Nomeie como: {name} e marque como Public (Make Public) se desejado")
    print("5) Salve/Publish")

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
