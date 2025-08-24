import requests
from fastapi import FastAPI, Form, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import PyPDF2
import os

from dotenv import load_dotenv

# -----------------------------
# NLP - Pré-processamento
# -----------------------------
import nltk
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
import string

# Baixar recursos do NLTK (apenas na primeira execução)
nltk.download("stopwords", quiet=True)
nltk.download("rslp", quiet=True)

stop_words = set(stopwords.words("portuguese"))
stemmer = RSLPStemmer()

def preprocess_text(text: str) -> str:
    """
    Realiza pré-processamento do texto:
    - Converte para minúsculas
    - Remove pontuação
    - Remove stopwords
    - Aplica stemming
    """
    # Minúsculas
    text = text.lower()

    # Remove pontuação
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Tokenização simples (split por espaço)
    tokens = text.split()

    # Remove stopwords e aplica stemming
    processed_tokens = [
        stemmer.stem(token) for token in tokens if token not in stop_words
    ]

    return " ".join(processed_tokens)

# -----------------------------
# Configurações da API DeepSeek
# -----------------------------
load_dotenv()
DS_API_KEY = os.getenv("DS_API_KEY")
DS_MODEL_ID = "deepseek/deepseek-r1-0528:free"
DS_API_URL = "https://openrouter.ai/api/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {DS_API_KEY}",
    "Content-Type": "application/json"
}

# -----------------------------
# Função para gerar respostas automáticas
# -----------------------------
def generate_email_reply(text: str, categoria: str) -> str:
    payload = {
        "model": DS_MODEL_ID,
        "messages": [
            {"role": "system", "content": "Você é um assistente que responde emails de forma produtiva e objetiva."},
            {"role": "user", "content": (
                f"Email recebido:\n{text}\n\n"
                f"Classificação do email: {categoria}\n\n"
                f"Baseado na classificação, escreva uma resposta apropriada, clara e objetiva."
            )}
        ],
        "temperature": 0.2,
        "max_output_tokens": 300
    }
    try:
        response = requests.post(DS_API_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        if "choices" in data and len(data["choices"]) > 0:
            answer = data["choices"][0].get("message", {}).get("content", "")
            if answer:
                return answer
        return "Não foi possível gerar uma resposta."
    except requests.exceptions.HTTPError as e:
        return f"Erro HTTP: {e}"
    except requests.exceptions.RequestException as e:
        return f"Erro na requisição: {e}"
    except ValueError:
        return "Erro ao decodificar resposta da API (resposta não é JSON)."

# -----------------------------
# Função para extrair texto do arquivo
# -----------------------------
def extract_text_from_file(file: UploadFile) -> str:
    content = ""
    file.file.seek(0)
    if file.filename.endswith(".txt"):
        content = file.file.read().decode("utf-8")
    elif file.filename.endswith(".pdf"):
        reader = PyPDF2.PdfReader(file.file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                content += page_text + "\n"
    else:
        raise ValueError("Formato de arquivo não suportado. Use .txt ou .pdf")
    return content

# -----------------------------
# Função para classificar o email
# -----------------------------
def classify_email(text: str) -> str:
    """
    Classificação baseada em palavras-chave simples.
    Agora usa texto já pré-processado.
    """
    keywords_produtivas = ["projet", "reuni", "tarefa", "praz", "entreg", "solicit", "ajud", "erro"]
    for kw in keywords_produtivas:
        if kw in text:
            return "Produtivo"
    return "Improdutivo"

# -----------------------------
# Inicialização do FastAPI
# -----------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Endpoint principal
# -----------------------------
@app.post("/classificar-email")
async def classificar_email_endpoint(
    content: str = Form(None),
    file: UploadFile = File(None)
):
    try:
        if file:
            raw_text = extract_text_from_file(file)
        elif content:
            raw_text = content
        else:
            return JSONResponse(status_code=400, content={"erro": "Nenhum conteúdo ou arquivo enviado."})

        # Pré-processar texto
        processed_text = preprocess_text(raw_text)

        # Classificar email com base no texto pré-processado
        categoria = classify_email(processed_text)

        # Gerar resposta automática (usando texto original para contexto melhor)
        resposta_sugerida = generate_email_reply(raw_text, categoria)

        result = {
            "categoria": categoria,
            "resposta_sugerida": resposta_sugerida,
            "texto_original": raw_text,
            "texto_processado": processed_text
        }

        return JSONResponse(content=result)

    except Exception as e:
        return JSONResponse(status_code=500, content={"erro": str(e)})
