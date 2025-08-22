import requests
from fastapi import FastAPI, Form, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import PyPDF2
import os

from dotenv import load_dotenv

# Carrega variáveis do arquivo .env
load_dotenv()

# -----------------------------
# Configurações da API DeepSeek
# -----------------------------
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
    """
    Gera uma resposta automática considerando o email original e sua classificação.
    Compatível com a API DeepSeek via OpenRouter.
    """
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
        # A resposta do OpenRouter vem em choices[0].message.content
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
    keywords_produtivas = ["projeto", "reunião", "tarefa", "prazo", "entrega", "solicitação", "ajuda", "erro"]
    for kw in keywords_produtivas:
        if kw in text.lower():
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

        # Classificar email
        categoria = classify_email(raw_text)

        # Gerar resposta automática com base na classificação
        resposta_sugerida = generate_email_reply(raw_text, categoria)

        result = {
            "categoria": categoria,
            "resposta_sugerida": resposta_sugerida,
            "texto_original": raw_text
        }

        return JSONResponse(content=result)

    except Exception as e:
        return JSONResponse(status_code=500, content={"erro": str(e)})
