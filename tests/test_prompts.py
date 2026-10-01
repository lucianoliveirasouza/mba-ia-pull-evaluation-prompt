"""
Testes automatizados para validação de prompts.
"""
import json
import pytest
import yaml
import sys
from pathlib import Path

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils import validate_prompt_structure
from push_prompts import build_chat_prompt, build_prompt_tags, is_unchanged_prompt_error

def load_prompts(file_path: str):
    """Carrega prompts do arquivo YAML."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

class TestPrompts:
    def test_prompt_has_system_prompt(self):
        """Verifica se o campo 'system_prompt' existe e não está vazio."""
        prompts_file = Path(__file__).parent.parent / "prompts" / "bug_to_user_story_v2.yml"
        data = load_prompts(str(prompts_file))
        assert data.get("system_prompt") and str(data.get("system_prompt")).strip(), "system_prompt ausente ou vazio"

    def test_prompt_has_role_definition(self):
        """Verifica se o prompt define uma persona (ex: "Você é um Product Manager")."""
        prompts_file = Path(__file__).parent.parent / "prompts" / "bug_to_user_story_v2.yml"
        data = load_prompts(str(prompts_file))
        sp = data.get("system_prompt", "")
        assert "Product Manager" in sp, "role 'Product Manager' não encontrado no system_prompt"

    def test_prompt_uses_dataset_input_key(self):
        root = Path(__file__).parent.parent
        data = load_prompts(str(root / "prompts" / "bug_to_user_story_v2.yml"))
        dataset_path = root / "datasets" / "bug_to_user_story.jsonl"
        examples = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines() if line.strip()]

        assert "{bug_report}" in data.get("user_prompt", "")
        assert all(set(example["inputs"]) == {"bug_report"} for example in examples)

    def test_push_prompt_includes_few_shot_messages(self):
        prompts_file = Path(__file__).parent.parent / "prompts" / "bug_to_user_story_v2.yml"
        data = load_prompts(str(prompts_file))
        rendered = build_chat_prompt(data).invoke({"bug_report": "Relato de teste."})

        assert len(rendered.messages) == 2 + 2 * len(data["few_shot_examples"])
        assert rendered.messages[1].type == "human"
        assert data["few_shot_examples"][0]["input"] in rendered.messages[1].content
        assert rendered.messages[2].type == "ai"
        assert data["few_shot_examples"][0]["output"] in rendered.messages[2].content
        assert rendered.messages[-1].content.strip() == "BUG REPORT:\nRelato de teste."

    def test_push_prompt_tags_include_techniques_and_version(self):
        prompts_file = Path(__file__).parent.parent / "prompts" / "bug_to_user_story_v2.yml"
        data = load_prompts(str(prompts_file))

        assert build_prompt_tags(data) == [
            "role_prompting",
            "few_shot",
            "skeleton_of_thought",
            "version:v2",
        ]

    def test_push_accepts_unchanged_content_after_metadata_update(self):
        error = Exception("Nothing to commit: prompt has not changed since latest commit")
        assert is_unchanged_prompt_error(error)
        assert not is_unchanged_prompt_error(Exception("permission denied"))

    def test_prompt_mentions_format(self):
        """Verifica se o prompt exige formato Markdown ou User Story padrão."""
        prompts_file = Path(__file__).parent.parent / "prompts" / "bug_to_user_story_v2.yml"
        data = load_prompts(str(prompts_file))
        sp = (data.get("system_prompt") or "") + "\n" + (data.get("user_prompt") or "")
        assert "User Story" in sp, "Formato 'User Story' não mencionado"
        assert ("Critérios de Aceitação" in sp) or ("Dado" in sp and "Quando" in sp and "Então" in sp), "Critérios BDD não detectados"

    def test_prompt_has_few_shot_examples(self):
        """Verifica se o prompt contém exemplos de entrada/saída (técnica Few-shot)."""
        prompts_file = Path(__file__).parent.parent / "prompts" / "bug_to_user_story_v2.yml"
        data = load_prompts(str(prompts_file))
        fs = data.get("few_shot_examples")
        assert isinstance(fs, list) and len(fs) >= 3, "Few-shot com menos de 3 exemplos"

    def test_prompt_no_todos(self):
        """Garante que você não esqueceu nenhum `[TODO]` no texto."""
        prompts_file = Path(__file__).parent.parent / "prompts" / "bug_to_user_story_v2.yml"
        text = open(prompts_file, 'r', encoding='utf-8').read()
        forbidden = ["TODO", "[TODO]", "FIXME", "TBD"]
        for token in forbidden:
            assert token not in text, f"Encontrado token proibido: {token}"

    def test_minimum_techniques(self):
        """Verifica (através dos metadados do yaml) se pelo menos 2 técnicas foram listadas."""
        prompts_file = Path(__file__).parent.parent / "prompts" / "bug_to_user_story_v2.yml"
        data = load_prompts(str(prompts_file))
        tech = data.get("metadata", {}).get("techniques", [])
        assert isinstance(tech, list) and "role_prompting" in tech and "few_shot" in tech, "metadata.techniques deve conter ao menos role_prompting e few_shot"

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])