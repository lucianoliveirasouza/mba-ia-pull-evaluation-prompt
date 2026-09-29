"""
Script COMPLETO para avaliar prompts otimizados.

Este script:
1. Carrega o dataset de avaliação do arquivo .jsonl (datasets/bug_to_user_story.jsonl)
2. Cria (ou reutiliza) o dataset de avaliação no LangSmith
3. Puxa o prompt otimizado do LangSmith Prompt Hub (fonte única de verdade)
4. Roda o prompt contra o dataset como um EXPERIMENTO do LangSmith
5. Calcula 5 métricas por exemplo (Helpfulness, Correctness, F1-Score, Clarity, Precision)
6. Publica cada nota como feedback no experimento, visível no dashboard do LangSmith
7. Exibe o resumo no terminal e imprime o link direto do experimento

Suporta múltiplos providers de LLM: OpenAI e Google Gemini.
Os modelos não são fixados aqui: defina LLM_MODEL e EVAL_MODEL no .env,
consultando a documentação oficial do provider escolhido.

Configure o provider no arquivo .env através da variável LLM_PROVIDER.
"""

import os
import sys
import json
from typing import List, Dict, Any
from pathlib import Path
from dotenv import load_dotenv
from langsmith import Client
from langchain_core.prompts import ChatPromptTemplate
from utils import check_env_vars, format_score, print_section_header, get_llm as get_configured_llm, get_eval_llm
from metrics import evaluate_f1_score, evaluate_clarity, evaluate_precision

load_dotenv()

# Nota mínima exigida em CADA métrica e também na média geral
APPROVAL_THRESHOLD = 0.8

# Ordem em que as métricas aparecem no relatório e no LangSmith
METRIC_KEYS = ["helpfulness", "correctness", "f1_score", "clarity", "precision"]


def get_llm():
    return get_configured_llm(temperature=0)


def normalize_inputs_for_prompt(prompt_template, inputs: dict) -> dict:
    """Normalize example input keys to match the prompt template's expected variables.

    Heuristics:
    - Use exact matches first.
    - Try common variants (e.g., 'bug_report' -> 'bug', '{var}_text', '{var}_report').
    - If single expected variable and the example has other keys, map the first value to it (fallback).
    """
    if inputs is None:
        return {}

    # Try to get expected variable names from the prompt template
    expected = None
    try:
        expected = getattr(prompt_template, "input_variables", None)
    except Exception:
        expected = None

    if not expected:
        try:
            expected = getattr(prompt_template, "input_keys", None)
        except Exception:
            expected = None

    expected = list(expected) if expected else []

    mapped: dict = {}
    received_keys = list(inputs.keys())

    def norm(s: str) -> str:
        return ''.join([c for c in s.lower() if c.isalnum()])

    for var in expected:
        if var in inputs:
            mapped[var] = inputs[var]
            continue

        # Try heuristics
        candidates = [f"{var}_report", f"{var}_text", f"{var}Report", f"{var}Text"]
        found = False
        for k in received_keys:
            if k in candidates or norm(k) == norm(var) or norm(k).endswith(norm(var)) or norm(var).endswith(norm(k)):
                mapped[var] = inputs[k]
                found = True
                break

        if not found:
            # special-case common source key
            if var == 'bug' and 'bug_report' in inputs:
                mapped[var] = inputs['bug_report']
                found = True

    # Fallback: if expected contains a single var and nothing mapped, map first available value
    if expected and len(expected) == 1 and expected[0] not in mapped and received_keys:
        mapped[expected[0]] = inputs[received_keys[0]]
        print(f"WARNING: mapeando '{received_keys[0]}' -> '{expected[0]}' por heuristica")

    # If prompt expects nothing, pass original inputs
    if not expected:
        return inputs

    return mapped


def load_dataset_from_jsonl(jsonl_path: str) -> List[Dict[str, Any]]:
    examples = []

    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:  # Ignorar linhas vazias
                    example = json.loads(line)
                    examples.append(example)

        return examples

    except FileNotFoundError:
        print(f"ERROR Arquivo nao encontrado: {jsonl_path}")
        print("\nCertifique-se de que o arquivo datasets/bug_to_user_story.jsonl existe.")
        return []
    except json.JSONDecodeError as e:
        print(f"ERROR Erro ao parsear JSONL: {e}")
        return []
    except Exception as e:
        print(f"ERROR Erro ao carregar dataset: {e}")
        return []


def create_evaluation_dataset(client: Client, dataset_name: str, jsonl_path: str) -> str:
    print(f"Criando dataset de avaliação: {dataset_name}...")

    examples = load_dataset_from_jsonl(jsonl_path)

    if not examples:
        print("ERROR Nenhum exemplo carregado do arquivo .jsonl")
        return dataset_name

    print(f"   OK Carregados {len(examples)} exemplos do arquivo {jsonl_path}")

    try:
        for ds in client.list_datasets(dataset_name=dataset_name):
            if ds.name == dataset_name:
                print(f"   OK Dataset '{dataset_name}' ja existe, usando existente")
                print(f"   OK {ds.url}")
                return dataset_name

        dataset = client.create_dataset(dataset_name=dataset_name)

        for example in examples:
            client.create_example(
                dataset_id=dataset.id,
                inputs=example["inputs"],
                outputs=example["outputs"]
            )

        print(f"   OK Dataset criado com {len(examples)} exemplos")
        print(f"   OK {dataset.url}")
        return dataset_name

    except Exception as e:
        print(f"   WARN Erro ao criar dataset: {e}")
        return dataset_name


def pull_prompt_from_langsmith(client: Client, prompt_name: str) -> ChatPromptTemplate:
    try:
        print(f"   Puxando prompt do LangSmith Hub: {prompt_name}")

        # dangerously_pull_public_prompt=True é obrigatório sempre que o identificador
        # tem dono explícito ("owner/nome"). O LangSmith bloqueia esse pull por padrão
        # porque um prompt do Hub é um objeto LangChain serializado, que pode vir de
        # terceiros. Como aqui o prompt é o seu (ou o prompt semente do desafio),
        # o risco é conhecido e aceito.
        prompt = client.pull_prompt(prompt_name, dangerously_pull_public_prompt=True)

        print(f"   OK Prompt carregado com sucesso")
        return prompt

    except Exception as e:
        error_msg = str(e).lower()

        print(f"\n{'=' * 70}")
        print(f"ERROR: Não foi possível carregar o prompt '{prompt_name}'")
        print(f"{'=' * 70}\n")

        if "not found" in error_msg or "404" in error_msg:
            print("WARN O prompt não foi encontrado no LangSmith Hub.\n")
            print("AÇÕES NECESSÁRIAS:")
            print("1. Verifique se você já fez push do prompt otimizado:")
            print(f"   python src/push_prompts.py")
            print()
            print("2. Confirme se o prompt foi publicado com sucesso em:")
            print(f"   https://smith.langchain.com/prompts")
            print()
            print(f"3. Confira se USERNAME_LANGSMITH_HUB no .env é o seu handle do Hub")
            print()
            print("4. Se você alterou o prompt no YAML, refaça o push:")
            print(f"   python src/push_prompts.py")
        else:
            print(f"Erro técnico: {e}\n")
            print("Verifique:")
            print("- LANGSMITH_API_KEY está configurada corretamente no .env")
            print("- Você tem acesso ao workspace do LangSmith")
            print("- Sua conexão com a internet está funcionando")

        print(f"\n{'=' * 70}\n")
        raise


def build_target(prompt_template: ChatPromptTemplate, llm: Any):
    """
    Monta a função que o LangSmith executa para cada exemplo do dataset.

    Recebe os inputs do exemplo (ex: {"bug_report": "..."}) e devolve a saída
    do modelo num dicionário. Cada chamada vira um run dentro do experimento.
    """
    chain = prompt_template | llm

    def target(inputs: dict) -> dict:
        mapped_inputs = normalize_inputs_for_prompt(prompt_template, inputs or {})
        response = chain.invoke(mapped_inputs)
        # Support different response shapes
        content = getattr(response, 'content', None)
        if content is None:
            # Some LLM wrappers return object with 'generations' or 'text'
            content = str(response)
        return {"answer": content}

    return target


def evaluate_all_metrics(inputs: dict, outputs: dict, reference_outputs: dict) -> Dict[str, Any]:
    """
    Avaliador do LangSmith: roda os três juízes uma única vez por exemplo e
    devolve as 5 métricas de uma vez.

    O LangSmith grava cada item da lista "results" como um feedback separado,
    então as 5 notas aparecem por exemplo no dashboard e são agregadas
    automaticamente no experimento.
    """
    question = (inputs or {}).get("bug_report", "")
    answer = (outputs or {}).get("answer", "") or ""
    from pathlib import Path
    from typing import List, Dict, Any
    from dotenv import load_dotenv
    from utils import get_llm, get_eval_llm
    import re

    load_dotenv()

    def call_llm_for_generation(prompt_template: str, bug_text: str) -> str:
        """Generate a user story from the prompt template and bug text using configured LLM.

        Returns the raw LLM text output.
        """
        # Use utils.get_llm to obtain a configured model

        try:
            llm = get_llm()
        except Exception as e:
            raise RuntimeError(f"Erro ao configurar LLM: {e}")

        # Prepare a simple prompt by replacing {bug} placeholder
        prompt_text = prompt_template.replace("{bug}", bug_text)

        # Use LangChain chat-style interface if available
        try:
            # Chat model call
            response = llm.generate([{"role": "user", "content": prompt_text}])
            # response.generations -> list[list[Generation]]
            generations = getattr(response, "generations", None)
            if generations:
                return generations[0][0].text
        except Exception:
            # Fallback to simple call
            try:
                return llm(prompt_text)
            except Exception as e:
                raise RuntimeError(f"Erro ao chamar LLM: {e}")

    def evaluate_with_llm(eval_prompt: str, reference: str, candidate: str) -> Dict[str, float]:
        """Call evaluation LLM (EVAL_MODEL) to score helpfulness and correctness.

        Returns a dict with keys: helpfulness, correctness.
        """

        try:
            eval_llm = get_eval_llm()
        except Exception as e:
            raise RuntimeError(f"Erro ao configurar LLM de avaliação: {e}")

        # Compose evaluation prompt
        prompt = eval_prompt.replace("{reference}", reference).replace("{candidate}", candidate)

        try:
            response = eval_llm.generate([{"role": "user", "content": prompt}])
            generations = getattr(response, "generations", None)
            text = generations[0][0].text if generations else str(response)
        except Exception:
            try:
                text = eval_llm(prompt)
            except Exception as e:
                raise RuntimeError(f"Erro ao chamar LLM de avaliação: {e}")

        # Extrair números simples do texto (procurar por padrões como 0.85 ou 85%)
        nums = re.findall(r"\d+\.?\d*%?", text)
        scores = {"helpfulness": 0.0, "correctness": 0.0}
        # heurística: procurar dois primeiros números
        parsed = []
        for n in nums:
            try:
                if n.endswith("%"):
                    parsed.append(float(n[:-1]) / 100.0)
                else:
                    parsed.append(float(n))
            except Exception:
                continue
        if parsed:
            scores["helpfulness"] = parsed[0] if parsed[0] <= 1.0 else parsed[0] / 100.0
        if len(parsed) > 1:
            scores["correctness"] = parsed[1] if parsed[1] <= 1.0 else parsed[1] / 100.0

        return scores

    reference = (reference_outputs or {}).get("reference", "")

    if not answer:
        return {
            "results": [
                {"key": key, "score": 0.0, "comment": "Resposta vazia"}
                for key in METRIC_KEYS
            ]
        }

    f1 = evaluate_f1_score(question, answer, reference)
    clarity = evaluate_clarity(question, answer, reference)
    precision = evaluate_precision(question, answer, reference)

    scores = {
        "f1_score": f1["score"],
        "clarity": clarity["score"],
        "precision": precision["score"],
    }
    # Métricas derivadas das métricas base
    scores["helpfulness"] = round((scores["clarity"] + scores["precision"]) / 2, 4)
    scores["correctness"] = round((scores["f1_score"] + scores["precision"]) / 2, 4)

    comments = {
        "f1_score": f1.get("reasoning", ""),
        "clarity": clarity.get("reasoning", ""),
        "precision": precision.get("reasoning", ""),
        "helpfulness": "Média entre Clarity e Precision",
        "correctness": "Média entre F1-Score e Precision",
    }

    return {
        "results": [
            {"key": key, "score": scores[key], "comment": comments[key]}
            for key in METRIC_KEYS
        ]
    }


def run_experiment(client: Client, prompt_name: str, dataset_name: str):
    """
    Executa o prompt contra o dataset como um experimento do LangSmith.

    Devolve (scores_medios, url_do_experimento).
    """
    print(f"\nRUNNING Avaliando: {prompt_name}")

    prompt_template = pull_prompt_from_langsmith(client, prompt_name)
    llm = get_llm()

    print("   Rodando experimento no LangSmith...")

    results = client.evaluate(
        build_target(prompt_template, llm),
        data=dataset_name,
        evaluators=[evaluate_all_metrics],
        experiment_prefix=prompt_name.replace("/", "-"),
        # Sequencial de propósito: os planos gratuitos de LLM costumam ter limite
        # baixo de requisições por minuto. Se o seu limite permitir, suba esse valor.
        max_concurrency=1,
        metadata={
            "prompt": prompt_name,
            "llm_model": os.getenv("LLM_MODEL", ""),
            "eval_model": os.getenv("EVAL_MODEL", ""),
            "provider": os.getenv("LLM_PROVIDER", ""),
        },
    )

    collected = {key: [] for key in METRIC_KEYS}

    try:
        for i, row in enumerate(results, 1):
            per_example = {}

            for result in row["evaluation_results"]["results"]:
                if result.key in collected and result.score is not None:
                    collected[result.key].append(result.score)
                    per_example[result.key] = result.score

            print(
                f"      [{i}] "
                f"F1:{per_example.get('f1_score', 0.0):.2f} "
                f"Clarity:{per_example.get('clarity', 0.0):.2f} "
                f"Precision:{per_example.get('precision', 0.0):.2f}"
            )
    except KeyboardInterrupt:
        print("\nWARN: Execução interrompida pelo usuário. Calculando métricas parciais...")
    except RuntimeError as e:
        # Captura RuntimeError de 'cannot schedule new futures after shutdown' e outros erros
        print(f"\nWARN: Erro durante a iteração dos resultados: {e}")
        print("WARN: Calculando métricas parciais a partir dos exemplos processados até agora...")
    except Exception as e:
        print(f"\nERROR: Falha inesperada ao processar resultados: {e}")
        print("WARN: Calculando métricas parciais a partir dos exemplos processados até agora...")

    scores = {
        key: round(sum(values) / len(values), 4) if values else 0.0
        for key, values in collected.items()
    }

    # Tentar recuperar a URL do experimento quando disponível
    experiment_url = None
    try:
        experiment_url = results.url
    except Exception:
        experiment_url = None

    return scores, experiment_url


def display_results(prompt_name: str, scores: Dict[str, float]) -> bool:
    print("\n" + "=" * 50)
    print(f"Prompt: {prompt_name}")
    print("=" * 50)

    print("\nMétricas Derivadas:")
    print(f"  - Helpfulness: {format_score(scores['helpfulness'], threshold=APPROVAL_THRESHOLD)}")
    print(f"  - Correctness: {format_score(scores['correctness'], threshold=APPROVAL_THRESHOLD)}")

    print("\nMétricas Base:")
    print(f"  - F1-Score: {format_score(scores['f1_score'], threshold=APPROVAL_THRESHOLD)}")
    print(f"  - Clarity: {format_score(scores['clarity'], threshold=APPROVAL_THRESHOLD)}")
    print(f"  - Precision: {format_score(scores['precision'], threshold=APPROVAL_THRESHOLD)}")

    average_score = sum(scores.values()) / len(scores)

    print("\n" + "-" * 50)
    print(f"MEDIA GERAL: {average_score:.4f}")
    print("-" * 50)

    all_above_threshold = all(score >= APPROVAL_THRESHOLD for score in scores.values())
    passed = all_above_threshold and average_score >= APPROVAL_THRESHOLD

    if passed:
        print(f"\nSTATUS: APROVADO - Todas as métricas >= {APPROVAL_THRESHOLD}")
    else:
        print(f"\nSTATUS: REPROVADO")

    return passed



def main() -> int:
    # Preparar client LangSmith e dataset
    if not check_env_vars(["LANGSMITH_API_KEY", "USERNAME_LANGSMITH_HUB", "LLM_PROVIDER", "LLM_MODEL", "EVAL_MODEL"]):
        print("Variáveis de ambiente obrigatórias faltando. Verifique o .env e tente novamente.")
        return 2

    username = os.getenv("USERNAME_LANGSMITH_HUB")
    dataset_name = os.getenv("LANGSMITH_PROJECT", "prompt-optimization-challenge-dataset")

    # Validate LLM configuration early to avoid running expensive network calls
    try:
        # validate main LLM
        get_llm()
        # validate evaluator LLM
        get_eval_llm(temperature=0)
    except Exception as e:
        print("ERROR: Configuracao de LLM invalida:", e)
        print("Verifique as variaveis de ambiente LLM_PROVIDER, LLM_MODEL, EVAL_MODEL e chaves de API.")
        return 2

    client = Client()

    # Criar/usar dataset no LangSmith
    jsonl_path = str(Path(__file__).parent.parent / "datasets" / "bug_to_user_story.jsonl")
    create_evaluation_dataset(client, dataset_name, jsonl_path)

    print("\n" + "=" * 70)
    print("PROMPTS PARA AVALIAR")
    print("=" * 70)
    print("\nEste script irá puxar prompts do LangSmith Hub.")
    print("Certifique-se de ter feito push dos prompts antes de avaliar:")
    print("  python src/push_prompts.py\n")

    prompts_to_evaluate = [f"{username}/bug_to_user_story_v2"]

    all_passed = True
    results_summary = []

    for prompt_name in prompts_to_evaluate:
        try:
            scores, experiment_url = run_experiment(client, prompt_name, dataset_name)
            passed = display_results(prompt_name, scores)
            all_passed = all_passed and passed

            results_summary.append({
                "prompt": prompt_name,
                "scores": scores,
                "passed": passed,
                "url": experiment_url,
            })

        except Exception as e:
            print(f"\nERROR Falha ao avaliar '{prompt_name}': {e}")
            all_passed = False

            results_summary.append({
                "prompt": prompt_name,
                "scores": {key: 0.0 for key in METRIC_KEYS},
                "passed": False,
                "url": None,
            })

    print("\n" + "=" * 50)
    print("RESUMO FINAL")
    print("=" * 50 + "\n")

    print(f"Prompts avaliados: {len(results_summary)}")
    print(f"Aprovados: {sum(1 for r in results_summary if r['passed'])}")
    print(f"Reprovados: {sum(1 for r in results_summary if not r['passed'])}\n")

    print("Resultados no LangSmith (notas gravadas como feedback no experimento):")
    for result in results_summary:
        if result["url"]:
            print(f"  {result['prompt']}")
            print(f"    {result['url']}")

    if all_passed:
        print(f"\nTodos os prompts atingiram todas as métricas >= {APPROVAL_THRESHOLD}!")
        print("\nProximos passos: documente o processo, capture screenshots e commit/push.")
        return 0
    else:
        print(f"\nALGUNS prompts nao atingiram todas as metricas >= {APPROVAL_THRESHOLD}")
        print("\nProximos passos: refatore os prompts com score baixo, push e reavalie.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
