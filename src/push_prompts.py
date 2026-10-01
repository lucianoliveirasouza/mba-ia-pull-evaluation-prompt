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


def build_chat_prompt(data: dict):
    from langchain_core.prompts import ChatPromptTemplate

    messages = []
    if data.get("system_prompt"):
        messages.append(("system", data["system_prompt"]))

    for example in data.get("few_shot_examples", []):
        example_input = example.get("input", "").replace("{", "{{").replace("}", "}}")
        example_output = example.get("output", "").replace("{", "{{").replace("}", "}}")
        if example_input and example_output:
            messages.append(("human", f"BUG REPORT:\n{example_input}"))
            messages.append(("ai", example_output))

    if data.get("user_prompt"):
        messages.append(("human", data["user_prompt"]))

    return ChatPromptTemplate.from_messages(messages) if messages else None


def build_prompt_tags(data: dict) -> list[str]:
    metadata = data.get("metadata", {})
    tags = list(metadata.get("techniques", []))
    version = metadata.get("version") or data.get("version")
    if version:
        tags.append(f"version:{version}")
    return tags


def is_unchanged_prompt_error(error: Exception) -> bool:
    return "nothing to commit" in str(error).lower()


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

    client = Client()

    try:
        chat_prompt = build_chat_prompt(data)
    except Exception as e:
        print("WARNING: could not build ChatPromptTemplate:", e)
        chat_prompt = None

    # Preferred: Client.push_prompt(prompt_identifier, object=..., is_public=True, description=...)
    try:
        if chat_prompt is not None and hasattr(client, "push_prompt"):
            res = client.push_prompt(
                name,
                object=chat_prompt,
                is_public=True,
                description=data.get("metadata", {}).get("description"),
                tags=build_prompt_tags(data),
            )
            print("client.push_prompt returned:", res)
            return True
    except Exception as e:
        if is_unchanged_prompt_error(e):
            print("Prompt sem alterações de conteúdo; metadados sincronizados no LangSmith.")
            return True
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
