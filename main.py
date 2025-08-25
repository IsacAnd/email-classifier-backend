import requests
from fastapi import FastAPI, Form, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import PyPDF2
import os

from dotenv import load_dotenv

import nltk
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
import string

nltk.download("stopwords", quiet=True)
nltk.download("rslp", quiet=True)

stop_words = set(stopwords.words("portuguese"))
stemmer = RSLPStemmer()

def preprocess_text(text: str) -> str:
    text = text.lower()

    text = text.translate(str.maketrans("", "", string.punctuation))

    tokens = text.split()

    processed_tokens = [
        stemmer.stem(token) for token in tokens if token not in stop_words
    ]

    return " ".join(processed_tokens)

load_dotenv()
DS_API_KEY = os.getenv("DS_API_KEY")
DS_MODEL_ID = "deepseek/deepseek-r1-0528:free"
DS_API_URL = "https://openrouter.ai/api/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {DS_API_KEY}",
    "Content-Type": "application/json"
}

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

keywords_produtivas_raw = [
    "projeto", "reunião", "tarefa", "prazo", "entrega", "solicitação", "ajuda", "erro"
]

# Stem das palavras-chave para combinar com texto pré-processado
keywords_produtivas = [stemmer.stem(kw.lower()) for kw in keywords_produtivas_raw]

def classify_email(text: str) -> str:

    for kw in keywords_produtivas:
        if kw in text:
            return "Produtivo"
    return "Improdutivo"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

        processed_text = preprocess_text(raw_text)

        categoria = classify_email(processed_text)

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
