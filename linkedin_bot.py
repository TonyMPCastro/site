import os
import time
import re
import json
import unicodedata
import random
import argparse
import requests
import warnings
import traceback
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ─────────────────────────────────────────────
# Suporte ao Google Generative AI (Gemini)
# ─────────────────────────────────────────────
HAS_GEMINI = False
try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    pass

# ─────────────────────────────────────────────
# Suporte ao ONNX Runtime (NPU / DirectML)
# ─────────────────────────────────────────────
HAS_ONNX = False
ONNX_PROVIDERS = []
optimum_model = None
optimum_tokenizer = None
ONNX_FAILED = False

try:
    import onnxruntime as ort
    from transformers import AutoTokenizer
    from optimum.onnxruntime import ORTModelForCausalLM
    HAS_ONNX = True
    ONNX_PROVIDERS = ort.get_available_providers()
except ImportError:
    pass

# ─────────────────────────────────────────────
# Dados do Candidato (Carregados de candidate_config.json ou env vars)
# ─────────────────────────────────────────────
DEFAULT_CANDIDATE_INFO = {
    "name": "Seu Nome Completo",
    "email": "seu.email@exemplo.com",
    "phone": "11999999999",
    "city": "Sao Paulo",
    "state": "Sao Paulo",
    "country": "Brasil",
    "country_code": "+55",
    "salary_clt": "8000",
    "salary_pj": "12000",
    "salary_default": "8000",
    "experience_years": "4",
    "english_level": "Intermediario",
    "notice_period": "1",
    "resume_path": "Curriculo.pdf",
    "remote_ok": "Yes",
    "cst_timezone_flexible": "Yes",
    "gcp_experience_years": "0",
    "bigquery_experience_years": "0",
    "healthcare_domain_plus": "No",
    "known_technologies": [
        "TypeScript", "JavaScript", "Node.js", "React", "Python", "SQL"
    ]
}

def load_candidate_info():
    """Carrega dados do candidato a partir de candidate_config.json local ou env vars."""
    info = DEFAULT_CANDIDATE_INFO.copy()
    
    # 1. Tenta carregar do arquivo local candidate_config.json (ignorado no git)
    config_path = os.path.join(os.getcwd(), "candidate_config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                local_data = json.load(f)
                info.update(local_data)
        except Exception as e:
            print(f"Erro ao ler candidate_config.json: {e}")

    # 2. Permite override via variaveis de ambiente
    env_mappings = {
        "CANDIDATE_NAME": "name",
        "CANDIDATE_EMAIL": "email",
        "CANDIDATE_PHONE": "phone",
        "CANDIDATE_CITY": "city",
        "CANDIDATE_STATE": "state",
        "CANDIDATE_SALARY": "salary_default",
        "CANDIDATE_RESUME": "resume_path",
    }
    for env_var, key in env_mappings.items():
        if os.environ.get(env_var):
            info[key] = os.environ.get(env_var)

    # 3. Garante path absoluto para o curriculo
    if info.get("resume_path"):
        info["resume_path"] = os.path.abspath(info["resume_path"])
        
    return info

CANDIDATE_INFO = load_candidate_info()

DEFAULT_SEARCH_TERMS = [
    "Programador php","Desenvolvedor pleno",
    "Desenvolvedor senior",
    
    "TI",
    "Programador Full Stack",
    "Developer",
    "Software Developer",
    "Analista TI"
]

applied_jobs = set()
processed_jobs = set()

# ─────────────────────────────────────────────
# Historico de Respostas (aprende com o tempo)
# ─────────────────────────────────────────────
HISTORY_FILE = os.path.join(os.getcwd(), "answers_history.json")

def load_history():
    """Carrega o historico de respostas salvas em disco."""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
            # Limpeza defensiva: elimina placeholder e respostas catalogadas como ruins.
            for key, value in list(history.items()):
                key_norm = normalize(str(key))
                value_norm = normalize(str(value))
                if is_placeholder_answer(value):
                    print(f"      [📚] Removendo placeholder do historico: {key}={value}")
                    del history[key]
                elif any(k in key_norm for k in ["country code", "codigo do pais", "codigo de pais", "codigo pais"]) and any(k in value_norm for k in ["alban", "albania", "albânia"]):
                    print(f"      [📚] Removendo resposta antiga de codigo do pais no historico: {key}={value}")
                    del history[key]
            return history
        except Exception:
            pass
    return {}

def save_history(history):
    """Salva o historico atualizado em disco."""
    try:
        clean = {}
        for key, value in (history or {}).items():
            if is_placeholder_answer(value):
                continue
            if is_placeholder_answer(key):
                continue
            clean[key] = value
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(clean, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def history_key(question_text, options=None):
    """Gera uma chave normalizada para o historico."""
    q = re.sub(r'\s+', ' ', question_text.strip().lower())
    if options:
        opts_key = "|".join(sorted([o.lower().strip() for o in options[:5]]))
        return f"{q}::{opts_key}"
    return q

def normalize(text):
    """Remove acentos e normaliza texto para comparacao robusta."""
    return unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode('ascii').lower()


def is_placeholder_answer(value):
    """Detecta respostas de placeholder salvadas no historico (select option, etc.)."""
    if value is None:
        return True
    v = normalize(str(value).strip())
    return any(k in v for k in ["selecionar opcao", "select an option", "select", "escolha uma opcao", "choose", "--", "placeholder"])


def is_numeric_field_request(question_text, field_type="text"):
    """Detecta perguntas onde a resposta deve ser numerica."""
    ft = normalize(str(field_type or "").strip())
    if ft in ["number", "numeric", "integer"] or "number" in ft:
        return True
    q = normalize(question_text or "")
    numeric_markers = [
        "de 0 a", "de 1 a", "de 2 a", "de 3 a", "de 4 a", "de 5 a",
        "de 0 ate", "de 1 ate", "de 2 ate", "de 0 a 100", "0 a 100",
        "1 a 5", "1 a 10", "de 1 a 5", "de 0 a 5", "range", "numero", "quantos",
        "quantidade", "anos de experiencia", "experience in years", "years of experience",
        "years", "anos", "salary", "pretensao salarial", "remuneracao", "salario"
    ]
    return any(k in q for k in numeric_markers)


def is_numeric_value(value):
    """Valida se a string devolvida e numeral puro."""
    if value is None:
        return False
    s = str(value).strip()
    if re.fullmatch(r"\d+", s):
        return True
    return False


def numeric_profile_answer(question_text):
    """Escolhe um numero do perfil para uma pergunta numerica."""
    q = normalize(question_text or "")
    if any(k in q for k in ["salary", "salario", "pretensao", "remuneracao"]):
        return str(CANDIDATE_INFO.get("salary_default", "9000"))
    if any(k in q for k in ["years", "anos", "experience"]):
        return str(CANDIDATE_INFO.get("experience_years", "4"))
    if any(k in q for k in ["preaviso", "notice", "start", "quando pode", "disponibilidade"]):
        return str(CANDIDATE_INFO.get("notice_period", "1"))
    if any(k in q for k in ["gcp", "google cloud"]):
        return str(CANDIDATE_INFO.get("gcp_experience_years", "2"))
    if any(k in q for k in ["bigquery"]):
        return str(CANDIDATE_INFO.get("bigquery_experience_years", "1"))
    return "4"

# Chaves bloqueadas (sem acentos para comparacao)
BLOCKED_KEYS = [
    "email", "e-mail", "phone", "telefone", "celular", "mobile", "telephone",
    "nome completo", "full name", "sobrenome", "last name", "first name",
    "primeiro nome", "linkedin url", "endereco de email", "phone number",
    "numero de telefone", "whatsapp", "selecionar opcao"
]

def is_blocked_field(label_txt):
    """Verifica se um campo e de contato/perfil (nao deve ser preenchido)."""
    nl = normalize(label_txt)
    return any(k in nl for k in BLOCKED_KEYS)

# Carrega o historico na inicializacao
answers_history = load_history()
if answers_history:
    print(f"   [HISTORICO] Historico carregado: {len(answers_history)} respostas conhecidas.")

# ─────────────────────────────────────────────
# Motor IA via ONNX / NPU
# ─────────────────────────────────────────────
def query_onnx_local(prompt):
    global optimum_model, optimum_tokenizer, ONNX_FAILED
    if not HAS_ONNX or ONNX_FAILED:
        return None

    model_id = "HuggingFaceTB/SmolLM-135M-Instruct"
    local_model_dir = os.path.join(os.getcwd(), "ai_model")  # Pasta local para cache

    try:
        if optimum_model is None or optimum_tokenizer is None:
            provider = "DmlExecutionProvider" if "DmlExecutionProvider" in ONNX_PROVIDERS else "CPUExecutionProvider"

            if os.path.isdir(local_model_dir) and os.listdir(local_model_dir):
                # Carrega do disco local (instantaneo, sem download)
                print(f"   [NPU] Carregando modelo local de '{local_model_dir}'...")
                optimum_tokenizer = AutoTokenizer.from_pretrained(local_model_dir)
                optimum_model = ORTModelForCausalLM.from_pretrained(local_model_dir, provider=provider)
                print(f"   [NPU] Modelo carregado do disco via {provider}!")
            else:
                # Primeira vez: baixa, converte e salva no disco
                print(f"   [NPU] Primeira execucao: baixando e convertendo modelo (~200MB)...")
                print(f"   [NPU] Isso so acontece UMA VEZ. Aguarde...")
                optimum_tokenizer = AutoTokenizer.from_pretrained(model_id)
                optimum_model = ORTModelForCausalLM.from_pretrained(model_id, export=True, provider=provider)
                # Salva localmente para nao precisar baixar de novo
                os.makedirs(local_model_dir, exist_ok=True)
                optimum_model.save_pretrained(local_model_dir)
                optimum_tokenizer.save_pretrained(local_model_dir)
                print(f"   [NPU] Modelo salvo em '{local_model_dir}'. Proximas execucoes serao instantaneas!")

        inputs = optimum_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
        outputs = optimum_model.generate(**inputs, max_new_tokens=8, do_sample=False)
        response_text = optimum_tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = response_text[len(prompt):].strip()
        if answer:
            first = answer.split()[0].strip(".,;:")
            bad_first_tokens = ["pergunta", "qual", "qual", "when", "what", "como", "resposta", "responda"]
            if normalize(first) in [normalize(x) for x in bad_first_tokens]:
                print(f"   [NPU] Resposta descartada por texto de pergunta: {first}")
                return None
            if len(first) < 2 or len(first) > 80:
                print(f"   [NPU] Resposta descartada por comprimento suspeito: {first}")
                return None
            print(f"   [NPU] Respondeu: {first}")
            return first
    except Exception as e:
        print(f"   [NPU] Erro: {e}")
        ONNX_FAILED = True
    return None

OLLAMA_OK = True
def query_ollama(prompt):
    global OLLAMA_OK
    if not OLLAMA_OK:
        return None
    try:
        res = requests.post("http://localhost:11434/api/generate",
            json={"model": "phi3", "prompt": prompt, "stream": False}, timeout=0.5)
        if res.status_code == 200:
            return res.json().get("response", "").strip()
    except Exception:
        OLLAMA_OK = False
    return None

# ─────────────────────────────────────────────
# Motor de Decisao Principal
# ─────────────────────────────────────────────
def option_from_profile(question_text, options):
    """Escolhe a melhor opcao do select usando dados do candidato como referencia."""
    if not options:
        return None

    q = normalize(question_text)
    clean_opts = []
    for opt in options:
        if not opt:
            continue
        if is_placeholder_answer(opt):
            continue
        norm_opt = normalize(opt)
        if any(p in norm_opt for p in ["selecione", "selecionar", "select", "escolha", "choose", "--", "select an option"]):
            continue
        clean_opts.append(opt)

    if not clean_opts:
        return None

    # --- Tecnologias conhecidas no perfil do candidato ---
    known = {normalize(t): t for t in CANDIDATE_INFO.get("known_technologies", [])}
    tech_terms = {
        "aws": ["aws", "amazon web services", "amazon"],
        "typescript": ["typescript", "ts"],
        "javascript": ["javascript", "js"],
        "node": ["node", "node.js", "nodejs"],
        "next": ["next", "next.js", "nextjs"],
        "react": ["react"],
        "python": ["python"],
        "sql": ["sql", "database"],
        "html": ["html"],
        "css": ["css"],
        "rest": ["rest", "api", "apis"],
        "n8n": ["n8n", "workflow automation"],
        "ia": ["ia", "ai", "machine learning", "ml"],
        "cloud": ["cloud", "devops"],
        "docker": ["docker", "containers"]
    }

    matched_opts = []
    for opt in clean_opts:
        opt_norm = normalize(opt)
        for tech_key, tech_patterns in tech_terms.items():
            if any(p in opt_norm for p in tech_patterns):
                # se a pergunta cita a tecnologia, esta e a melhor resposta no perfil
                if any(p in q for p in tech_patterns) or tech_key in q:
                    matched_opts.append(opt)
                    break

    if matched_opts:
        return matched_opts[0]

    # Se a pergunta revelar uma tecnologia e a lista de opcoes tiver essa tecnologia,
    # devolve a opcao de conhecimento do perfil.
    if any(k in q for k in ["aws", "amazon", "web services", "typescript", "javascript", "node", "next", "react", "python", "sql", "html", "css", "rest", "api", "n8n", "ai", "ia", "docker", "cloud"]):
        for opt in clean_opts:
            opt_norm = normalize(opt)
            for tech_key, patterns in tech_terms.items():
                if any(p in q for p in patterns) and any(p in opt_norm for p in patterns):
                    return opt
        # fallback se a pergunta for de tecnologia e a resposta for yes/no
        if any(v in q for v in ["know", "experiencia", "familiar", "experiente", "use"]):
            for opt in clean_opts:
                opt_norm = normalize(opt)
                if any(token in opt_norm for token in ["yes", "sim", "aws", "javascript", "typescript", "node", "react", "next", "python", "sql", "html", "css", "api", "n8n", "ai", "cloud", "docker"]):
                    return opt

    # --- Mapeamento direto com o perfil do candidato ---
    if any(k in q for k in ["remote", "remoto", "home office", "working from home", "comfortable working in a remote setting"]):
        pref = "Yes" if str(CANDIDATE_INFO.get("remote_ok", "Yes")).lower() in ["yes", "y", "sim", "true"] else "No"
        for opt in clean_opts:
            if normalize(pref) == normalize(opt) or (pref.lower() in normalize(opt)):
                return opt
        return clean_opts[0]

    if any(k in q for k in ["timezone", "cst", "business hours", "business timezone"]):
        pref = "Yes" if str(CANDIDATE_INFO.get("cst_timezone_flexible", "Yes")).lower() in ["yes", "y", "sim", "true"] else "No"
        for opt in clean_opts:
            if normalize(pref) == normalize(opt) or (pref.lower() in normalize(opt)):
                return opt
        return clean_opts[0]

    if any(k in q for k in ["google cloud", "gcp", "cloud platform"]):
        expected = str(CANDIDATE_INFO.get("gcp_experience_years", "2"))
        for opt in clean_opts:
            if expected in normalize(opt) or normalize(expected) in normalize(opt):
                return opt
        return None

    if any(k in q for k in ["bigquery", "google bigquery"]):
        expected = str(CANDIDATE_INFO.get("bigquery_experience_years", "1"))
        for opt in clean_opts:
            if expected in normalize(opt) or normalize(expected) in normalize(opt):
                return opt
        return None

    if any(k in q for k in ["health care", "healthcare", "saude", "health"]):
        pref = "No" if str(CANDIDATE_INFO.get("healthcare_domain_plus", "No")).lower() in ["no", "n", "nao", "false"] else "Yes"
        for opt in clean_opts:
            if normalize(pref) == normalize(opt) or (pref.lower() in normalize(opt)):
                return opt
        return clean_opts[0]

    # --- Reuso de perfil para campos geograficos e de perfil ---
    if any(k in q for k in ["country", "pais", "country of residence"]):
        for opt in clean_opts:
            if any(t in normalize(opt) for t in ["brasil", "brazil"]):
                return opt
        # Nao inventa pais fora do perfil nem devolve string sem opcao do select.
        return None

    if any(k in q for k in ["country code", "codigo do pais", "codigo de pais", "codigo pais"]):
        for opt in clean_opts:
            if "+55" in opt or "55" in opt or any(t in normalize(opt) for t in ["brasil", "brazil", "55"]):
                return opt
        return None

    # --- Reuso de perfil para outras perguntas com contexto claro ---
    if any(k in q for k in ["experience", "anos", "years"]):
        expected = str(CANDIDATE_INFO.get("experience_years", "4"))
        for opt in clean_opts:
            if expected in normalize(opt) or normalize(expected) in normalize(opt):
                return opt
        return clean_opts[0]

    if any(k in q for k in ["salary", "pretensao", "remuneracao", "compensation"]):
        salary = str(CANDIDATE_INFO.get("salary_default", "9000"))
        for opt in clean_opts:
            if normalize(salary) in normalize(opt) or any(d in normalize(opt) for d in ["9000", "9k", "r$ 9", "9000"]):
                return opt
        for opt in clean_opts:
            if "pj" in normalize(opt) or "clt" in normalize(opt):
                return opt
        return clean_opts[0]

    # --- Recomendacao geral de escolha pelo conjunto ---
    scored = []
    for opt in clean_opts:
        norm_opt = normalize(opt)
        score = 0
        # Preferir opcoes com 'sim' quando a pergunta e de confirmacao e o candidato aceita.
        if any(k in q for k in ["yes", "sim", "work", "available", "flexible"]):
            if "yes" in norm_opt or "sim" in norm_opt:
                score += 2
            if "no" in norm_opt or "nao" in norm_opt:
                score -= 1
        # Se a pergunta pede um numero e opt tem um numero, e a pergunta e relacionada a perfil; score favoravel.
        if any(k in q for k in ["years", "anos", "number", "quantos"]):
            if any(ch.isdigit() for ch in opt):
                score += 2
        # Se o select tem opcoes de localidade/estado, e o candidato e Brasil, priorizar Brazil/BR.
        if any(k in q for k in ["country", "pais", "city", "estado", "state", "cidade"]):
            if any(t in norm_opt for t in ["brasil", "brazil", "sao luis", "maranhao"]):
                score += 3
        scored.append((score, opt))

    if scored:
        best = max(scored, key=lambda x: x[0])[1]
        return best
    return None


def ask_ai(question_text, options=None, field_type="text", api_key=None, use_npu=True):
    global answers_history
    q = question_text.strip().lower()
    numeric_request = is_numeric_field_request(question_text, field_type)

    # --- SEGURANCA: pula campos de contato/perfil ---
    if is_blocked_field(question_text):
        return ""  # Nao processa esses campos

    print(f"      [?] {question_text[:70]}")

    # --- HISTORICO: resposta ja conhecida? ---
    hkey = history_key(question_text, options)
    if hkey in answers_history:
        cached = answers_history[hkey]
        if is_placeholder_answer(cached):
            print(f"      [📚] Historico descartado por placeholder: '{cached}'")
            del answers_history[hkey]
            save_history(answers_history)
        elif numeric_request and not is_numeric_value(cached):
            print(f"      [📚] Historico descartado por nao ser numerico: '{cached}'")
            del answers_history[hkey]
            save_history(answers_history)
        else:
            print(f"      [📚] Historico: '{cached}'")
            return cached

    # --- Resposta numerica do perfil, sem deixar cair em Sim/No ---
    if numeric_request:
        answer = numeric_profile_answer(question_text)
        answers_history[hkey] = answer
        save_history(answers_history)
        print(f"      [🔢] Numero perfil: '{answer}'")
        return answer

    # --- SELECAO DO SELECT COM BASE NO PERFIL DO CANDIDATO ---
    if options:
        profile_match = option_from_profile(question_text, options)
        if is_placeholder_answer(profile_match):
            print(f"      [🎯] Perfil descartado por placeholder: '{profile_match}'")
            profile_match = None
        if profile_match:
            if hkey not in answers_history:
                answers_history[hkey] = profile_match
                save_history(answers_history)
            print(f"      [🎯] Perfil: '{profile_match}'")
            return profile_match

    # --- HEURISTICA CRITICA (sem risco de alucinacao) ---
    # Reforco de seguranca: nunca deixe um modelo generativo inventar codigo do pais ou pais
    # se a pergunta estiver no escopo geografico dele.
    if any(k in q for k in ["country code", "codigo do pais", "codigo de pais", "codigo pais"]):
        return CANDIDATE_INFO.get("country_code", "+55")

    if any(k in q for k in ["pais", "country"]):
        return CANDIDATE_INFO.get("country", "Brasil")

    if any(k in q for k in ["remote", "remoto", "working from home", "home office", "comfortable working in a remote setting"]):
        return "Yes" if str(CANDIDATE_INFO.get("remote_ok", "Yes")).lower() in ["yes", "y", "sim", "true"] else "No"

    if any(k in q for k in ["flexible to work during cst", "timezone business hours", "cst", "business hours", "business timezone"]):
        return "Yes" if str(CANDIDATE_INFO.get("cst_timezone_flexible", "Yes")).lower() in ["yes", "y", "sim", "true"] else "No"

    if any(k in q for k in ["google cloud platform", "gcp", "cloud platform"]):
        return str(CANDIDATE_INFO.get("gcp_experience_years", "0"))

    if any(k in q for k in ["bigquery", "google bigquery"]):
        return str(CANDIDATE_INFO.get("bigquery_experience_years", "0"))

    if any(k in q for k in ["health care domain", "healthcare domain", "saude", "health care", "domain is a plus"]):
        return "Yes" if str(CANDIDATE_INFO.get("healthcare_domain_plus", "No")).lower() in ["yes", "y", "sim", "true"] else "No"

    if any(k in q for k in ["financeiro", "financial", "processos financeiros", "csc", "shared services", "centro de servicos compartilhados"]):
        return "No"

    if any(k in q for k in ["sao paulo", "são paulo", "sp", "sede em sao paulo", "sede em são paulo", "availability in sao paulo"]):
        return "No"

    if any(k in q for k in ["anos de experiencia", "years of experience", "anos exp", "quantos anos", "anos com", "experience in years"]):
        return str(CANDIDATE_INFO.get("experience_years", "4"))

    if any(k in q for k in ["pretensao salarial", "salary expectation", "remuneracao", "expectativa salarial", "salario pretendido", "pretensao"]):
        if options:
            for opt in options:
                if "pj" in opt.lower() or "clt" in opt.lower():
                    return opt
        return CANDIDATE_INFO["salary_default"]

    if any(k in q for k in ["country code", "codigo do pais", "codigo de pais", "codigo pais"]):
        return CANDIDATE_INFO.get("country_code", "+55")

    if any(k in q for k in ["telefone", "celular", "phone", "mobile", "whatsapp"]):
        return CANDIDATE_INFO["phone"]

    if any(k in q for k in ["aviso previo", "notice period", "disponibilidade", "data de inicio", "quando pode comecar"]):
        return CANDIDATE_INFO["notice_period"]

    if any(k in q for k in ["pais", "country"]):
        return CANDIDATE_INFO.get("country", "Brasil")

    if any(k in q for k in ["cidade", "city", "municipio", "localidade"]):
        return CANDIDATE_INFO["city"]

    if any(k in q for k in ["estado ", "state", " uf"]):
        return CANDIDATE_INFO["state"]

    if "ingles" in q or "english" in q or "lingua inglesa" in q or "nivel de ingles" in q:
        if options:
            for kw in ["basico", "basic", "elementary", "a1", "a2"]:
                for opt in options:
                    if kw in opt.lower():
                        return opt
            return options[0]
        return "Basico"

    if any(k in q for k in ["trabalhar remoto", "home office", "remoto", "hibrido", "modalidade de trabalho"]):
        if options:
            for kw in ["remoto", "hibrido", "sim", "yes"]:
                for opt in options:
                    if kw in opt.lower():
                        return opt
            return options[0]
        return "Sim"

    if any(k in q for k in ["regime clt", "regime pj", "tipo de contrato", "contrato"]):
        if options:
            for kw in ["ambos", "clt", "pj"]:
                for opt in options:
                    if kw in opt.lower():
                        return opt
            return options[0]
        return "CLT"

    # --- IA PARA PERGUNTAS ABERTAS ---
    opts_desc = f"\nOpcoes: {options}" if options else ""
    prompt_str = (
        f"Candidato: Dev Fullstack, 4 anos exp, R$9000, ingles Basico, aceita PJ/CLT.\n"
        f"Responda com uma unica palavra/numero para:\nPergunta: {question_text}{opts_desc}\nResposta:"
    )

    if use_npu:
        r = query_onnx_local(prompt_str)
        if r:
            # O NPU/ONNX local ocasionalmente devolve um token de pergunta ou frase aleatoria
            # em vez de uma resposta de dominio. Essa validacao impede que ela seja aceita.
            if numeric_request and not is_numeric_value(r):
                r = None
            if r and any(k in normalize(r) for k in ["pergunta", "qual ta errando", "respoista", "perguntas", "responda", "qual"]):
                r = None
            elif options:
                if not any(normalize(r) == normalize(opt) or normalize(opt) in normalize(r) or normalize(r) in normalize(opt) for opt in options):
                    r = None
            if r:
                return r
        r = query_ollama(prompt_str)
        if r:
            if numeric_request and not is_numeric_value(r):
                r = None
            if r and any(k in normalize(r) for k in ["pergunta", "qual ta errando", "respoista", "perguntas", "responda", "qual"]):
                r = None
            elif options:
                if not any(normalize(r) == normalize(opt) or normalize(opt) in normalize(r) or normalize(r) in normalize(opt) for opt in options):
                    r = None
            if r:
                return r

    if HAS_GEMINI and (api_key or os.environ.get("GEMINI_API_KEY")):
        try:
            genai.configure(api_key=api_key or os.environ.get("GEMINI_API_KEY"))
            r = genai.GenerativeModel("gemini-1.5-flash").generate_content(prompt_str).text.strip()
            if r: return r
        except Exception:
            pass

    # --- FALLBACK ---
    if numeric_request:
        answer = numeric_profile_answer(question_text)
        answers_history[hkey] = answer
        save_history(answers_history)
        return answer

    if options:
        # EMERGENCIA: nunca deixar um select de pais cair na primeira opcao arbitraria
        q_country = normalize(question_text)
        if any(k in q_country for k in ["country", "pais", "country of residence"]):
            answer = CANDIDATE_INFO.get("country", "Brasil")
            answers_history[hkey] = answer
            save_history(answers_history)
            return answer
        if any(k in q_country for k in ["country code", "codigo do pais", "codigo de pais", "codigo pais"]):
            answer = CANDIDATE_INFO.get("country_code", "+55")
            answers_history[hkey] = answer
            save_history(answers_history)
            return answer
        opt_candidates = [opt for opt in options if opt and not is_placeholder_answer(opt)]
        if not opt_candidates:
            return "No"
        for opt in opt_candidates:
            if normalize(opt) in ["yes", "sim", "true", "y"]:
                answer = opt
                answers_history[hkey] = answer
                save_history(answers_history)
                return answer
        for opt in opt_candidates:
            if normalize(opt) in ["no", "nao", "false", "n"]:
                answer = opt
                answers_history[hkey] = answer
                save_history(answers_history)
                return answer
        for opt in opt_candidates:
            if opt and not any(p in opt.lower() for p in ["selecione", "select", "escolha", "choose", "--"]):
                answer = opt
                answers_history[hkey] = answer
                save_history(answers_history)
                return answer
        answer = opt_candidates[-1]
        answers_history[hkey] = answer
        save_history(answers_history)
        return answer

    if field_type == "number":
        answer = "4"
        answers_history[hkey] = answer
        save_history(answers_history)
        return answer

    answer = "Sim"
    answers_history[hkey] = answer
    save_history(answers_history)
    return answer

# ─────────────────────────────────────────────
# Utilitarios
# ─────────────────────────────────────────────
def safe_text(elem, default=""):
    try:
        return elem.inner_text().strip() if elem else default
    except Exception:
        return default

SCREENSHOTS_DIR = os.path.join(os.getcwd(), "screenshots")
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

def take_screenshot(page, label="unknown"):
    """Tira print da tela e tenta analisar via Gemini Vision se disponivel."""
    try:
        fname = os.path.join(SCREENSHOTS_DIR, f"{label}_{int(time.time())}.png")
        page.screenshot(path=fname, full_page=False)
        print(f"      [📸] Screenshot salva: {fname}")

        # Tenta analise via Gemini Vision
        if HAS_GEMINI:
            api_key_env = os.environ.get("GEMINI_API_KEY")
            if api_key_env:
                try:
                    import pathlib
                    genai.configure(api_key=api_key_env)
                    model = genai.GenerativeModel("gemini-1.5-flash")
                    img_data = pathlib.Path(fname).read_bytes()
                    import base64
                    b64 = base64.b64encode(img_data).decode()
                    response = model.generate_content([
                        {
                            "parts": [
                                {"text": "Esta e a tela de um formulario de candidatura de emprego no LinkedIn. "
                                         "Identifique todos os campos visiveis que precisam ser preenchidos "
                                         "e sugira como preenche-los para um desenvolvedor Fullstack brasileiro "
                                         "com 4 anos de experiencia, ingles basico, salario R$9000."},
                                {"inline_data": {"mime_type": "image/png", "data": b64}}
                            ]
                        }
                    ])
                    print(f"      [🤖 Gemini Vision]: {response.text[:300]}")
                except Exception as ve:
                    print(f"      [Vision] Erro: {ve}")
        return fname
    except Exception as e:
        print(f"      [Screenshot] Erro: {e}")
        return None

def wait_modal(page, timeout=10):
    """Aguarda o modal de candidatura abrir com timeout generoso."""
    # Primeiro aguarda um pouco para a animacao iniciar
    time.sleep(0.5)
    selectors = [
        ".jobs-easy-apply-modal",
        ".artdeco-modal[role='dialog']",
        "div[data-test-modal]",
        "div[role='dialog']",
        ".artdeco-modal",
        ".ember-view[role='dialog']",
        "[data-test-modal-container]"
    ]
    deadline = time.time() + timeout
    while time.time() < deadline:
        for sel in selectors:
            try:
                m = page.query_selector(sel)
                if m and m.is_visible():
                    # Confirma que e modal de candidatura (tem botao de avançar/enviar)
                    has_form = page.evaluate("""() => {
                        const modal = document.querySelector(
                            '.jobs-easy-apply-modal, .artdeco-modal[role=\"dialog\"], div[role=\"dialog\"]');
                        if (!modal) return false;
                        const btns = modal.querySelectorAll('button');
                        return btns.length > 0;
                    }""")
                    if has_form:
                        return m
            except Exception:
                pass
        time.sleep(0.4)
    return None

# ─────────────────────────────────────────────
# Recuperacao por validacao de campo
# ─────────────────────────────────────────────
def field_profile_value_for(label_text):
    """Retorna o valor esperado do perfil para um campo geografico ou de perfil."""
    lt = normalize(label_text or "")
    if any(k in lt for k in ["country code", "codigo do pais", "codigo de pais", "codigo pais", "country code"]):
        return CANDIDATE_INFO.get("country_code") or "+55"
    if any(k in lt for k in ["country", "pais", "country of residence"]):
        return CANDIDATE_INFO.get("country") or "Brasil"
    if any(k in lt for k in ["state", "estado", "uf"]):
        return CANDIDATE_INFO.get("state") or "Maranhao"
    if any(k in lt for k in ["city", "cidade", "municipio", "localidade"]):
        return CANDIDATE_INFO.get("city") or "Sao Luis"
    return None


def validate_and_recover_visible_fields(page, modal, api_key=None, use_npu=True):
    """Valida campos visiveis apos o preenchimento e reenvia a melhor opcao usando o perfil/IA."""
    if not modal:
        return

    modal_text = (modal.inner_text() or "").lower()
    if any(k in modal_text for k in ["informacoes de contato", "candidate-se", "contact information", "contact"]):
        # A primeira tela do Easy Apply ja vem com o perfil do LinkedIn preenchido
        # (email/telefone/pais etc). O bot apenas precisa seguir para o proximo passo.
        return

    ctx = modal

    def get_label_for_element(el):
        label_text = ""
        fid = el.get_attribute("id")
        if fid:
            lbl = ctx.query_selector(f"label[for='{fid}']") or page.query_selector(f"label[for='{fid}']")
            if lbl:
                label_text = safe_text(lbl)
        if not label_text:
            label_text = el.get_attribute("aria-label") or el.get_attribute("placeholder") or ""
        return label_text

    def visible_error_text(el):
        try:
            msg = ""
            # Elemento com aria-invalid em visual feedback
            try:
                aria_invalid = el.get_attribute("aria-invalid") or ""
                if str(aria_invalid).lower() == "true":
                    msg += " aria-invalid"
            except Exception:
                pass

            # Mensagens referenciadas por aria-describedby / aria-errormessage
            try:
                refs = el.get_attribute("aria-describedby") or ""
                for rid in str(refs).split():
                    v = ctx.query_selector(f"#{rid}")
                    if v:
                        msg += " " + (v.inner_text() or "")
            except Exception:
                pass
            try:
                msg_id = el.get_attribute("aria-errormessage") or ""
                v = ctx.query_selector(f"#{msg_id}") if msg_id else None
                if v:
                    msg += " " + (v.inner_text() or "")
            except Exception:
                pass

            # Texto visivel no corpo do container e proximidades
            parent = el.evaluate("el => el.closest('div, section, form, fieldset')")
            if parent:
                txt = (parent.inner_text() or "").strip()
                msg += " " + txt
            try:
                siblings = el.evaluate("el => Array.from(el.parentElement ? el.parentElement.children : [])")
                if siblings:
                    for sib in siblings:
                        try:
                            if sib is el:
                                continue
                            txt = (sib.inner_text() or "").strip()
                            if txt:
                                msg += " " + txt
                        except Exception:
                            pass
            except Exception:
                pass
            return normalize(msg)
        except Exception:
            return ""

    # Text/number/email/tel inputs
    for inp in ctx.query_selector_all("input[type='text'], input[type='number'], input[type='email'], input[type='tel'], textarea, input:not([type])"):
        try:
            if not inp.is_visible():
                continue
            if inp.evaluate("el => el.disabled"):
                continue

            label_text = get_label_for_element(inp)
            if is_blocked_field(label_text):
                continue

            error_txt = visible_error_text(inp)
            inp_type = (inp.get_attribute("type") or "text").lower()
            if "resposta valida" in error_txt or "insira" in error_txt or "invalid" in error_txt or "required" in error_txt or "obrig" in error_txt:
                print(f"      [⚠️] Erro visual no input: {label_text}")
                answer = ask_ai(label_text, field_type=inp_type, api_key=api_key, use_npu=use_npu)
                if answer:
                    inp.fill(answer)
                    time.sleep(0.25)
                    print(f"      [🛠️] Campo corrigido via validação: {label_text} -> {answer}")
                continue

            expected = field_profile_value_for(label_text)
            if not expected:
                continue
            try:
                current = (inp.input_value() or "").strip()
            except Exception:
                current = ""
            if not current:
                continue
            current_norm = normalize(current)
            expected_norm = normalize(expected)
            if expected_norm not in current_norm and not any(token in current_norm for token in ["brasil", "brazil", "sao luis", "maranhao"] if token in expected_norm):
                answer = ask_ai(label_text, field_type=(inp.get_attribute("type") or "text").lower(), api_key=api_key, use_npu=use_npu)
                if answer:
                    inp.fill(answer)
                    time.sleep(0.25)
                    print(f"      [🛠️] Campo corrigido com base no perfil: {label_text} -> {answer}")
        except Exception:
            continue

    # Selects
    for sel in ctx.query_selector_all("select"):
        try:
            if not sel.is_visible():
                continue

            label_text = get_label_for_element(sel)
            expected = field_profile_value_for(label_text)
            error_txt = visible_error_text(sel)
            invalid = False
            try:
                invalid = (sel.get_attribute("aria-invalid") or "").lower() == "true"
            except Exception:
                invalid = False
            if error_txt and any(k in error_txt for k in ["resposta valida", "insira", "invalid", "required", "obrig", "campo"]):
                invalid = True

            if invalid:
                print(f"      [⚠️] Select com erro visivel: {label_text}")
                opts = [opt.inner_text().strip() for opt in sel.query_selector_all("option") if opt.inner_text().strip()]
                if opts:
                    chosen = ask_ai(label_text, options=opts, field_type="select", api_key=api_key, use_npu=use_npu)
                    if chosen:
                        opt_vals = []
                        for opt in sel.query_selector_all("option"):
                            t = opt.inner_text().strip()
                            v = opt.get_attribute("value") or ""
                            if t:
                                opt_vals.append((t, v))
                        for opt_text, val in opt_vals:
                            if normalize(chosen) in normalize(opt_text) or normalize(opt_text) in normalize(chosen):
                                try:
                                    sel.select_option(value=val) if val else sel.select_option(index=0)
                                    print(f"      [🛠️] Select corrigido via validação: {label_text} -> {chosen}")
                                    break
                                except Exception:
                                    pass
            if not expected:
                continue
            selected_text = sel.evaluate("el => el.options[el.selectedIndex]?.text || ''").strip()
            if selected_text:
                selected_norm = normalize(selected_text)
                expected_norm = normalize(expected)
                if expected_norm not in selected_norm:
                    opts = [opt.inner_text().strip() for opt in sel.query_selector_all("option") if opt.inner_text().strip()]
                    chosen = ask_ai(label_text, options=opts, field_type="select", api_key=api_key, use_npu=use_npu)
                    if chosen:
                        opt_vals = []
                        for opt in sel.query_selector_all("option"):
                            t = opt.inner_text().strip()
                            v = opt.get_attribute("value") or ""
                            if t:
                                opt_vals.append((t, v))
                        for opt_text, val in opt_vals:
                            if normalize(chosen) in normalize(opt_text) or normalize(opt_text) in normalize(chosen):
                                try:
                                    sel.select_option(value=val) if val else sel.select_option(index=0)
                                    print(f"      [🛠️] Select corrigido com base no perfil: {label_text} -> {chosen}")
                                    break
                                except Exception:
                                    pass
        except Exception:
            continue

# ─────────────────────────────────────────────
# Preenchimento de Formulario
# ─────────────────────────────────────────────
def fill_form(page, api_key=None, use_npu=True):
    modal = wait_modal(page, timeout=3)
    if not modal:
        return

    # Tela ja preenchida pelo perfil do LinkedIn: nao tentar re-preencher
    # o conteudo de contato e sim apenas seguir para o proximo passo.
    modal_text = (modal.inner_text() or "").lower()
    if any(k in modal_text for k in ["informacoes de contato", "candidate-se", "avancar"]):
        # Se o modal ja expuser o estado de perfil/resumo preenchido e o botao de avancar,
        # devolve sem tocar nos campos; o caller continua e chama click_next_or_submit().
        pass

    ctx = modal

    # Marca via JS os campos de perfil para NÃO preencher
    BLOCKED_FIELD_KEYWORDS = [
        "email", "phone", "telephone",
        "celular", "mobile", "nome completo", "full name", "sobrenome",
        "last name", "primeiro nome", "first name", "linkedin url"
    ]

    def is_blocked_field(label_txt):
        lt = label_txt.lower()
        return any(k in lt for k in BLOCKED_FIELD_KEYWORDS)

    # Texto / Numero
    SKIP_LABELS = ["email", "e-mail", "telefone", "phone", "celular", "mobile",
                   "nome", "name", "sobrenome", "surname", "linkedin.com", "perfil"]
    for inp in ctx.query_selector_all("input[type='text'], input[type='number'], input[type='email'], input[type='tel'], textarea, input:not([type])"):
        try:
            if not inp.is_visible():
                continue
            # Pula campos de busca
            ph = (inp.get_attribute("placeholder") or "").lower()
            if any(k in ph for k in ["search", "pesquisa", "buscar"]):
                continue
            # Pula pelo tipo HTML
            inp_type = (inp.get_attribute("type") or "").lower()
            if inp_type in ["email", "tel"]:
                continue

            # Verifica o label antes de qualquer coisa
            label_text = ""
            fid = inp.get_attribute("id")
            if fid:
                lbl = ctx.query_selector(f"label[for='{fid}']") or page.query_selector(f"label[for='{fid}']")
                if lbl:
                    label_text = safe_text(lbl)
            if not label_text:
                label_text = inp.get_attribute("aria-label") or inp.get_attribute("placeholder") or ""

            # Pula campos de perfil/contato pelo label
            if is_blocked_field(label_text):
                continue

            # Pula campo que ja tem valor preenchido
            curr = ""
            try:
                curr = inp.input_value()
            except Exception:
                pass
            if curr and curr.strip() not in ["", "0"]:
                continue

            if not label_text.strip():
                continue  # sem label nao processa

            f_type = inp.get_attribute("type") or "text"
            answer = ask_ai(label_text, field_type=f_type, api_key=api_key, use_npu=use_npu)
            if answer:  # nao preenche se ask_ai retornou vazio (campo bloqueado)
                inp.fill(answer)
            time.sleep(0.15)
        except Exception:
            continue

    # Radio Buttons em fieldsets
    for fs in ctx.query_selector_all("fieldset"):
        try:
            legend = fs.query_selector("legend, span.fb-form-element__label")
            q_title = safe_text(legend, "Pergunta")
            radios = fs.query_selector_all("input[type='radio']")
            if any(r.is_checked() for r in radios if r.is_visible()):
                continue
            opts = []
            r_map = {}
            for r in radios:
                rid = r.get_attribute("id") or ""
                lbl = fs.query_selector(f"label[for='{rid}']")
                t = safe_text(lbl) if lbl else ""
                if t:
                    opts.append(t)
                    r_map[t] = lbl
            if not opts:
                opts = [safe_text(l) for l in fs.query_selector_all("label") if safe_text(l)]
            chosen = ask_ai(q_title, options=opts, field_type="radio", api_key=api_key, use_npu=use_npu)
            clicked = False
            for t, lbl_elem in r_map.items():
                if chosen.lower() in t.lower() or t.lower() in chosen.lower():
                    if lbl_elem:
                        lbl_elem.click()
                        clicked = True
                        break
            if not clicked:
                labels = fs.query_selector_all("label")
                for lbl in labels:
                    if safe_text(lbl).lower() in ["sim", "yes"]:
                        lbl.click()
                        clicked = True
                        break
                if not clicked and labels:
                    labels[0].click()
            time.sleep(0.15)
        except Exception:
            continue

    # Radio Buttons fora de fieldsets (alguns forms de LinkedIn usam div/labels direto)
    for rad in ctx.query_selector_all("input[type='radio']"):
        try:
            if not rad.is_visible():
                continue
            if rad.is_checked():
                continue
            if rad.evaluate("el => Boolean(el.closest('fieldset'))"):
                continue
            rid = rad.get_attribute("id") or ""
            lbl = ctx.query_selector(f"label[for='{rid}']")
            container = rad.evaluate("el => el.closest('div, section, form')")
            if container:
                labels = container.query_selector_all("label")
                opts = [safe_text(l) for l in labels if safe_text(l)]
                if not opts:
                    # sem label para inferir opcoes, semana sem quantidade
                    continue
                q_text = safe_text(container).splitlines()[0] if safe_text(container) else "Pergunta"
                if "".join(opts).strip() == "":
                    continue
                chosen = ask_ai(q_text, options=opts, field_type="radio", api_key=api_key, use_npu=use_npu)
                for opt, lbl_elem in zip(opts, labels):
                    if chosen.lower() in opt.lower() or opt.lower() in chosen.lower():
                        try:
                            lbl_elem.click()
                        except Exception:
                            pass
                        break
            elif lbl:
                label_text = safe_text(lbl)
                answer = ask_ai(label_text, options=[label_text], field_type="radio", api_key=api_key, use_npu=use_npu)
                if answer:
                    try:
                        lbl.click()
                    except Exception:
                        pass
        except Exception:
            continue

    # Checkboxes de confirmacao / follow / confirm / agreement
    for chk in ctx.query_selector_all("input[type='checkbox']"):
        try:
            if not chk.is_visible():
                continue
            if chk.is_checked():
                continue
            # Evita mexer em controles de follow / companhia, se vier com texto de seguir
            chk_label = chk.evaluate("el => el.closest('label')")
            label_text = safe_text(chk_label) if chk_label else ""
            if any(k in normalize(label_text) for k in ["seguir", "follow", "salvar", "save"]):
                continue
            q_title = label_text or "Confirmacao"
            chosen = ask_ai(q_title, options=["Yes", "No"], field_type="checkbox", api_key=api_key, use_npu=use_npu)
            if "yes" in normalize(chosen) or "sim" in normalize(chosen):
                chk.check()
        except Exception:
            continue

    # Dropdowns
    for sel in ctx.query_selector_all("select"):
        try:
            # 1. PRIMEIRO verifica se ja tem valor selecionado - se sim, NAO TOCA
            selected_text = sel.evaluate("el => el.options[el.selectedIndex]?.text || ''").strip()
            # Lista ampla de placeholders (inclui portugues com/sem acento)
            placeholder_vals = ["selecione", "select", "escolha", "choose", "-- select",
                                 "selecionar", "selecionar opcao", "selecionar op", "",
                                 "please select", "select an option"]
            normalized_selected = normalize(selected_text)
            if selected_text and not any(p in normalized_selected for p in [normalize(x) for x in placeholder_vals]):
                continue  # Ja preenchido com valor real, NAO TOCA!

            # 2. Descobre o label
            label_text = ""
            fid = sel.get_attribute("id")
            if fid:
                lbl = ctx.query_selector(f"label[for='{fid}']") or page.query_selector(f"label[for='{fid}']")
                if lbl:
                    label_text = safe_text(lbl)
            if not label_text:
                label_text = sel.get_attribute("aria-label") or ""

            # 3. Pula campos de perfil/contato pelo label
            if any(k in label_text.lower() for k in SKIP_LABELS):
                continue

            # 4. Se label vazio nao tem como decidir, pula
            if not label_text.strip():
                continue

            opts_elems = sel.query_selector_all("option")
            opt_list, opt_vals = [], []
            for opt in opts_elems:
                t = opt.inner_text().strip()
                v = opt.get_attribute("value") or ""
                if t and not any(p in t.lower() for p in ["selecione", "select", "escolha", "choose", "--"]):
                    opt_list.append(t)
                    opt_vals.append(v)
            if not opt_list:
                continue
            chosen = ask_ai(label_text, options=opt_list, field_type="select", api_key=api_key, use_npu=use_npu)
            if not chosen or not str(chosen).strip():
                continue
            matched = False
            for i, ot in enumerate(opt_list):
                if str(chosen).lower() in ot.lower() or ot.lower() in str(chosen).lower():
                    try:
                        sel.select_option(value=opt_vals[i]) if opt_vals[i] else sel.select_option(index=i)
                        matched = True
                        break
                    except Exception:
                        pass
            if not matched:
                try:
                    # fallback seguro para selects que ja estao vazios e precisam
                    # responder de modo consistente com a API de respostas
                    for i, ot in enumerate(opt_list):
                        if normalize(str(chosen)) in normalize(ot) or normalize(ot) in normalize(str(chosen)):
                            sel.select_option(value=opt_vals[i]) if opt_vals[i] else sel.select_option(index=i)
                            matched = True
                            break
                    if not matched:
                        sel.select_option(index=0)
                except Exception:
                    pass
            time.sleep(0.15)
        except Exception:
            continue

    # Corrige campos que ja tenham sido preenchidos com valor errado
    validate_and_recover_visible_fields(page, ctx, api_key=api_key, use_npu=use_npu)

    # Curriculo
    for f_inp in ctx.query_selector_all("input[type='file']"):
        if os.path.exists(CANDIDATE_INFO["resume_path"]):
            try:
                f_inp.set_input_files(CANDIDATE_INFO["resume_path"])
                time.sleep(0.3)
            except Exception:
                pass

# ─────────────────────────────────────────────
# Botao Easy Apply
# ─────────────────────────────────────────────
def click_easy_apply(page):
    """Clica no botao de Candidatura Simplificada usando Playwright nativo + fallback JS."""

    # Primeiro evita o botao de candidatura externa que vem como role=link
    # com label apontando para o site da empresa (nao e Easy Apply). 
    external = False
    try:
        external = page.evaluate("""() => {
            const btns = Array.from(document.querySelectorAll('button, [role="button"], [role="link"]'));
            const found = btns.find(b => {
                const text = (b.innerText || b.getAttribute('aria-label') || b.getAttribute('title') || '').trim().toLowerCase();
                const role = (b.getAttribute('role') || '').toLowerCase();
                const isExternalLabel = text.includes('site da empresa') || text.includes('site da') || text.includes('no site da empresa');
                const hasExternalSvg = Boolean(b.querySelector('svg use[href*="link-external-small"]'));
                const hasApplyClass = b.classList && b.classList.contains('jobs-apply-button');
                return (role === 'link' || hasExternalSvg) && (isExternalLabel || (hasApplyClass && text.includes('candidatar-se')));
            });
            return Boolean(found);
        }""")
    except Exception:
        external = False

    if external:
        print("      Vaga externa: botao de candidatura fora do Easy Apply detectado. Pulando.")
        return False

    # 1. Playwright nativo por seletores CSS
    selectors = [
        "button.jobs-apply-button--top-card",
        "button.jobs-apply-button",
        "button[aria-label*='Candidatura Simplificada']",
        "button[aria-label*='Easy Apply']",
        ".jobs-s-apply button",
        "[data-control-name='jobdetails_topcard_inapply']",
    ]
    for sel in selectors:
        try:
            btn = page.query_selector(sel)
            if btn and btn.is_visible():
                btn.scroll_into_view_if_needed()
                time.sleep(0.3)
                btn.click()
                return True
        except Exception:
            continue

    # 2. get_by_role (mais confiavel que seletor CSS)
    for name in ["Candidatura Simplificada", "Easy Apply"]:
        try:
            btn = page.get_by_role("button", name=name).first
            if btn.is_visible():
                btn.click()
                return True
        except Exception:
            continue

    # 3. JS fallback varrendo todo o DOM
    try:
        clicked = page.evaluate("""() => {
            const btns = Array.from(document.querySelectorAll('button, [role="button"]'));
            const btn = btns.find(b => {
                const t = (b.innerText || b.getAttribute('aria-label') || b.getAttribute('title') || '').trim().toLowerCase();
                return (t.includes('candidatura simplificada') || t === 'easy apply') && !b.disabled;
            });
            if (btn) { btn.scrollIntoView({block:'center'}); btn.click(); return true; }
            return false;
        }""")
        return bool(clicked)
    except Exception:
        pass

    return False



def click_next_or_submit(page):
    # 1. Primeiro: exercita o modal diretamente com Playwright para que a escolha
    # seja feita com base no texto visivel e no aria-label do botao.
    modal = wait_modal(page, timeout=1)
    if modal:
        for btn in modal.query_selector_all("button"):
            try:
                if not btn.is_visible():
                    continue
                t = normalize((btn.inner_text() or "") + " " + (btn.get_attribute("aria-label") or "") + " " + (btn.get_attribute("title") or ""))
                if not t:
                    continue
                # ignora botoes de voltar/fechar/voltar ao estado anterior no mesmo modal
                if any(k in t for k in ["voltar", "back", "return", "cancelar", "cancel", "dismiss", "fechar"]):
                    continue
                if any(k in t for k in ["submit", "submeter", "enviar", "send", "application"]):
                    btn.click()
                    return "submitted"
                if any(k in t for k in ["revisar", "review", "review application", "avaliar", "analisar"]):
                    btn.click()
                    return "next"
                if any(k in t for k in ["concluir", "conclude", "finish", "finalizar", "done", "completar", "complete"]):
                    btn.click()
                    return "next"
                if any(k in t for k in ["avancar", "advance", "next", "continue", "continuar", "proximo", "proceed", "go ahead", "go on"]):
                    btn.click()
                    return "next"
            except Exception:
                continue

    # 2. Se o primeiro passo nao detectar visualmente, tenta o role mapping antigo
    labels = [
        "Enviar candidatura", "Submit application", "Submit", "Send application",
        "Revisar", "Review", "Review application", "Avaliar", "Analisar",
        "Avançar", "Avancar", "Advance", "Next", "Continue", "Continuar",
        "Próximo", "Proximo", "Proceed", "Go on", "Go ahead", "Concluir",
        "Conclude", "Finalizar", "Finish", "Done", "Completar"
    ]
    for label in labels:
        try:
            btn = page.get_by_role("button", name=label).first
            if btn and btn.is_visible():
                btn.click()
                if any(k in normalize(label) for k in ["submit", "submeter", "enviar", "send", "application"]):
                    return "submitted"
                return "next"
        except Exception:
            pass

    # 3. fallback CSS e portal/js para o caso em que o conteudo e dynamic.
    selectors = [
        "button[aria-label*='Enviar candidatura']",
        "button[aria-label*='Submit application']",
        "button[aria-label*='Submit']",
        "button[aria-label*='Send application']",
        "button[aria-label*='Revisar']",
        "button[aria-label*='Review']",
        "button[aria-label*='Avançar']",
        "button[aria-label*='Avancar']",
        "button[aria-label*='Advance']",
        "button[aria-label*='Next']",
        "button[aria-label*='Continue']",
        "button[aria-label*='Continuar']",
        "button[aria-label*='Próximo']",
        "button[aria-label*='Proximo']",
        "button[aria-label*='Proceed']",
        "button[aria-label*='Go']",
        "button[aria-label*='Finalizar']",
        "button[aria-label*='Finish']",
        "button[aria-label*='Concluir']",
        "button[aria-label*='Conclude']",
        "button.jobs-apply-button",
    ]
    for sel in selectors:
        try:
            btn = page.locator(sel).first
            if btn and btn.is_visible():
                btn.click()
                return "next"
        except Exception:
            continue

    try:
        return page.evaluate("""() => {
            const inModal = el => el.closest('[role="dialog"], .artdeco-modal, .jobs-easy-apply-modal');
            const btns = Array.from(document.querySelectorAll('button'));
            const normalize = s => (s || '').toLowerCase().normalize('NFD').replace(/[\u0300-\u036f]/g, '');
            const label = b => normalize(b.getAttribute('aria-label') || b.innerText || b.getAttribute('title') || '');
            const submit = btns.find(b => {
                const t = label(b);
                if (!inModal(b) || b.disabled) return false;
                return t.includes('enviar candidatura') || t.includes('submit application') || t.includes('submit') || t.includes('send application') || t.includes('enviar') || (t.includes('application') && t.includes('send'));
            });
            if (submit) { submit.click(); return 'submitted'; }

            const review = btns.find(b => {
                const t = label(b);
                if (!inModal(b) || b.disabled) return false;
                return t.includes('revisar') || t.includes('review') || t.includes('review application') || t.includes('avaliar') || t.includes('analisar');
            });
            if (review) { review.click(); return 'next'; }

            const finish = btns.find(b => {
                const t = label(b);
                if (!inModal(b) || b.disabled) return false;
                return t.includes('concluir') || t.includes('conclude') || t.includes('finish') || t.includes('finalizar') || t.includes('done') || t.includes('complete') || t.includes('completar');
            });
            if (finish) { finish.click(); return 'next'; }

            const nxt = btns.find(b => {
                const t = label(b);
                if (!inModal(b) || b.disabled) return false;
                return t.includes('avancar') || t.includes('advance') || t.includes('next') || t.includes('continue') || t.includes('continuar') || t.includes('proximo') || t.includes('proceed') || t.includes('go ahead') || t.includes('go on');
            });
            if (nxt) { nxt.click(); return 'next'; }
            return 'none';
        }""")
    except Exception:
        return "none"

def unfollow_company(page):
    try:
        page.evaluate("""() => {
            const lbl = Array.from(document.querySelectorAll('label')).find(l =>
                (l.innerText || '').toLowerCase().includes('seguir') || (l.innerText || '').toLowerCase().includes('follow'));
            if (!lbl) return;
            const id = lbl.getAttribute('for');
            if (id) {
                const chk = document.getElementById(id);
                if (chk && chk.checked) lbl.click();
            } else {
                const chk = lbl.querySelector('input[type="checkbox"]');
                if (chk && chk.checked) chk.click();
            }
        }""")
    except Exception:
        pass

def close_modal(page):
    """Fecha o modal e descarta a candidatura se aparecer o dialog de confirmacao."""
    try:
        # Fecha o X do modal principal
        page.evaluate("""() => {
            const btn = document.querySelector(
                'button[aria-label*="Fechar"], button[aria-label*="Dismiss"], button[aria-label*="Close"], .artdeco-modal__dismiss');
            if (btn) btn.click();
        }""")
        time.sleep(1)
    except Exception:
        pass

    # Trata o dialog de "Salvar candidatura?" / "Descartar candidatura"
    try:
        page.evaluate("""() => {
            // Procura botao de Descartar (nao queremos salvar rascunho)
            const btns = Array.from(document.querySelectorAll('button'));
            const discard = btns.find(b => {
                const t = (b.innerText || b.getAttribute('aria-label') || '').toLowerCase();
                return t.includes('descartar') || t.includes('discard') || t.includes('delete') || t.includes('excluir');
            });
            if (discard) { discard.click(); return; }
            // Se nao houver descartar, fecha o segundo modal que aparecer
            const dismiss = document.querySelector('.artdeco-modal__dismiss, button[aria-label*="Fechar"]');
            if (dismiss) dismiss.click();
        }""")
        time.sleep(0.5)
    except Exception:
        pass

# ─────────────────────────────────────────────
# Esperar Login
# ─────────────────────────────────────────────
def wait_for_login(page):
    print("\n" + "="*52)
    print("FACA LOGIN NO LINKEDIN NA JANELA QUE ABRIU")
    print("="*52)
    print("O robo aguarda. Assim que logar, comeca automaticamente!\n")
    while True:
        try:
            url = page.url.lower()
            if not any(k in url for k in ["login", "signup", "checkpoint", "uas", "authwall"]):
                if page.query_selector(".global-nav, #global-nav") or "feed" in url or "jobs" in url:
                    print("LOGIN DETECTADO! Iniciando automacao...\n")
                    return
        except Exception:
            pass
        time.sleep(1.5)

# ─────────────────────────────────────────────
# Processar uma vaga
# ─────────────────────────────────────────────
def get_job_cards(page):
    """Detecta os cards de vagas via JavaScript - robusto a mudancas de CSS do LinkedIn."""
    try:
        count = page.evaluate("""() => {
            // Tenta varios seletores em ordem de prioridade
            const selectors = [
                '.jobs-search-results__list-item',
                'ul.scaffold-layout__list-container > li',
                'li[data-occluded-item-id]',
                '.job-card-container',
                '[data-view-name="job-card"]',
                '.scaffold-layout__list > ul > li',
                '.jobs-search-results-list > ul > li'
            ];
            for (const sel of selectors) {
                const items = document.querySelectorAll(sel);
                if (items.length > 0) return items.length;
            }
            return 0;
        }""")
        return count
    except Exception:
        return 0

def click_job_card(page, index):
    """Clica no card de vaga pelo indice via JavaScript."""
    try:
        return page.evaluate(f"""() => {{
            const selectors = [
                '.jobs-search-results__list-item',
                'ul.scaffold-layout__list-container > li',
                'li[data-occluded-item-id]',
                '.job-card-container',
                '[data-view-name="job-card"]',
                '.scaffold-layout__list > ul > li',
                '.jobs-search-results-list > ul > li'
            ];
            for (const sel of selectors) {{
                const items = document.querySelectorAll(sel);
                if (items.length > {index}) {{
                    items[{index}].scrollIntoView({{behavior: 'smooth', block: 'center'}});
                    items[{index}].click();
                    return true;
                }}
            }}
            return false;
        }}""")
    except Exception:
        return False

def process_job(page, index, api_key, use_npu):
    try:
        clicked = click_job_card(page, index)
        if not clicked:
            return False
        time.sleep(2)

        title_elem = page.query_selector(
            "h1.job-details-jobs-unified-top-card__job-title, "
            "h2.job-details-jobs-unified-top-card__job-title, "
            ".t-24.t-bold, h1.t-24"
        )
        company_elem = page.query_selector(
            ".job-details-jobs-unified-top-card__company-name a, "
            ".jobs-unified-top-card__company-name a"
        )

        job_title = safe_text(title_elem, "Vaga")
        company = safe_text(company_elem, "Empresa")
        job_key = f"{job_title.lower()}|{company.lower()}"

        if job_key in applied_jobs or job_key in processed_jobs:
            print(f"   [{index+1}] Ja analisada/candidatada: {job_title}")
            return False

        processed_jobs.add(job_key)

        print(f"\n   [{index+1}] {job_title} | {company}")

        # Sempre que o bot se deparar com vaga que nao e Easy Apply, nao sai da pagina de vagas.
        # Ele apenas registra o skip e segue para a proxima listed job.
        if not click_easy_apply(page):
            print("      Vaga externa ou sem candidatura simplificada. Pulando.")
            processed_jobs.add(job_key)
            return False

        # Aguarda 1.5s para animacao do modal iniciar
        time.sleep(1.5)
        modal = wait_modal(page, timeout=10)
        if not modal:
            # Tenta clicar de novo (pode ter tido falha silenciosa)
            click_easy_apply(page)
            time.sleep(2)
            modal = wait_modal(page, timeout=5)
        
        if not modal:
            print("      Modal nao abriu (vaga pode ser externa). Pulando.")
            processed_jobs.add(job_key)
            close_modal(page)
            return False

        print("      Preenchendo formulario...")
        submitted = False
        stalled = 0

        for step in range(12):
            time.sleep(1.2)
            fill_form(page, api_key=api_key, use_npu=use_npu)
            time.sleep(0.5)
            unfollow_company(page)

            action = click_next_or_submit(page)

            if action == "submitted":
                time.sleep(1.5)
                applied_jobs.add(job_key)
                processed_jobs.add(job_key)
                print(f"      CANDIDATURA ENVIADA! ({job_title})")
                close_modal(page)
                return True
            elif action == "next":
                stalled = 0
            else:
                stalled += 1
                if stalled >= 2:
                    print("      Formulario travado. Tirando screenshot para analise...")
                    processed_jobs.add(job_key)
                    take_screenshot(page, label=f"stuck_{job_title[:20].replace(' ', '_')}")
                    close_modal(page)
                    break
                m = wait_modal(page, timeout=1)
                if not m:
                    processed_jobs.add(job_key)
                    break

        close_modal(page)
        return False

    except Exception as ex:
        print(f"      Erro na vaga {index+1}: {ex}")
        close_modal(page)
        return False

def build_jobs_search_url(term, include_remote=True, include_easy_apply=True):
    """Construtor centralizado com filtro de candidatura simplificada (Easy Apply).

    Mantem a estrutura do exemplo de filtro enviado pelo usuario:
    https://www.linkedin.com/jobs/search/?currentJobId=4448796521&distance=25.0&f_AL=true&f_TPR=r604800&f_WT=2&geoId=106057199&keywords=Desenvolvedor&origin=JOB_SEARCH_PAGE_JOB_FILTER&sortBy=R
    """
    params = [
        "f_AL=true" if include_easy_apply else "f_AL=false",
        "f_WT=2" if include_remote else "f_WT=2",
        "f_TPR=r604800",
        "geoId=106057199",
        "distance=25.0",
        "origin=JOB_SEARCH_PAGE_JOB_FILTER",
        "sortBy=R",
        f"keywords={requests.utils.quote(term)}",
    ]
    return "https://www.linkedin.com/jobs/search/" + "?" + "&".join(params)

# ─────────────────────────────────────────────
# Principal
# ─────────────────────────────────────────────
def click_next_jobs_page(page):
    """Clica na pagina seguinte da busca de vagas, usando o sistema de paginaçao do LinkedIn."""
    try:
        for name in ["Next", "Next page", "Próxima", "Próxima página", "Próximo"]:
            try:
                btn = page.get_by_role("button", name=name).first
                if btn and btn.is_visible():
                    btn.click()
                    time.sleep(2)
                    return True
            except Exception:
                pass
    except Exception:
        pass

    try:
        clicked = page.evaluate("""() => {
            const btns = Array.from(document.querySelectorAll('button, a'));
            const btn = btns.find(b => {
                const t = (b.innerText || b.getAttribute('aria-label') || b.getAttribute('title') || '').trim().toLowerCase();
                return (t.includes('proxima') || t.includes('next') || t.includes('next page')) && !b.disabled;
            });
            if (btn) { btn.click(); return true; }
            return false;
        }""")
        time.sleep(2)
        return bool(clicked)
    except Exception:
        return False


def run_bot(headless=False, cdp_port=None, api_key=None, use_npu=True):
    user_data_dir = os.path.join(os.getcwd(), "linkedin_session")
    os.makedirs(user_data_dir, exist_ok=True)

    print("="*52)
    print("BOT LINKEDIN - CANDIDATURAS AUTOMATICAS")
    print("="*52)
    print(f"Candidato : {CANDIDATE_INFO['name']}")
    print(f"Curriculo : {CANDIDATE_INFO['resume_path']}")
    print(f"NPU/ONNX  : {'ATIVADO (' + ', '.join(ONNX_PROVIDERS) + ')' if HAS_ONNX else 'DESATIVADO'}")
    print(f"Gemini API: {'ATIVADO' if HAS_GEMINI and (api_key or os.environ.get('GEMINI_API_KEY')) else 'DESATIVADO'}")
    print("="*52 + "\n")

    with sync_playwright() as p:
        browser_context = None

        if cdp_port:
            try:
                bi = p.chromium.connect_over_cdp(f"http://localhost:{cdp_port}")
                browser_context = bi.contexts[0] if bi.contexts else None
            except Exception:
                pass

        if not browser_context:
            browser_context = p.chromium.launch_persistent_context(
                user_data_dir=user_data_dir,
                headless=headless,
                viewport={"width": 1366, "height": 768},
                args=["--disable-blink-features=AutomationControlled", "--no-sandbox"],
                locale="pt-BR",
                timezone_id="America/Sao_Paulo"
            )

        page = browser_context.pages[0] if browser_context.pages else browser_context.new_page()
        page.goto("https://www.linkedin.com/", wait_until="domcontentloaded")
        time.sleep(2)

        url = page.url.lower()
        if any(k in url for k in ["login", "signup", "authwall"]):
            wait_for_login(page)
        else:
            print("Sessao ativa. Iniciando...\n")

        total = 0

        for term in DEFAULT_SEARCH_TERMS:
            print(f"\n{'='*52}\nPesquisando: '{term}'\n{'='*52}")
            search_url = build_jobs_search_url(term, include_remote=True, include_easy_apply=True)
            try:
                page.goto(search_url, wait_until="domcontentloaded")
                time.sleep(3)
                page.evaluate("window.scrollTo(0, 400)")
                time.sleep(1)
            except Exception as e:
                print(f"Erro ao navegar: {e}")
                continue

            # Aguarda os cards carregarem
            time.sleep(2)
            for page_loop in range(6):
                count = get_job_cards(page)
                print(f"   {count} vagas encontradas")

                if count == 0:
                    # Tenta sem filtro de remoto para garantir
                    search_url2 = build_jobs_search_url(term, include_remote=False, include_easy_apply=True)
                    page.goto(search_url2, wait_until="domcontentloaded")
                    time.sleep(3)
                    page.evaluate("window.scrollTo(0, 400)")
                    time.sleep(1)
                    count = get_job_cards(page)
                    print(f"   {count} vagas (sem filtro remoto)")
                    if count == 0:
                        break

                for i in range(min(count, 20)):
                    if process_job(page, i, api_key=api_key, use_npu=use_npu):
                        total += 1
                    time.sleep(0.5)

                # Paginacao da busca: rola o conjunto seguinte dentro da mesma pesquisa
                # usando os filtros do LinkedIn e nao indo para outros sites
                if not click_next_jobs_page(page):
                    break

                time.sleep(2)

        print(f"\n{'='*52}")
        print(f"AUTOMACAO CONCLUIDA! Candidaturas enviadas: {total}")
        print(f"{'='*52}")

        try:
            browser_context.close()
        except Exception:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bot LinkedIn com NPU")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--cdp", type=int, default=None)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--no-npu", action="store_true")
    args = parser.parse_args()
    run_bot(headless=args.headless, cdp_port=args.cdp, api_key=args.api_key, use_npu=not args.no_npu)
