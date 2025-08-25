# 🖥️ Backend - Classificador de E-mails

Este é o **backend** da aplicação de classificação de e-mails.  
Ele é responsável por processar os textos enviados pela interface (frontend) e retornar:

- A **categoria do e-mail** (produtivo ou improdutivo)
- Uma **resposta sugerida** para o usuário

---

## 🚀 Tecnologias utilizadas

- **Python**
- **FastAPI** (framework web)
- **Transformers (HuggingFace)**

---

## ⚙️ Como executar localmente

### 🔹 Pré-requisitos

- [Python](https://www.python.org/) **3.10 ou superior**
- [pip](https://pip.pypa.io/) (gerenciador de pacotes)

---

### 🔹 Passo a passo

1. **Clonar o repositório (se ainda não fez isso)**

```bash
   git clone https://github.com/IsacAnd/email-classifier-backend
   cd ./email-classifier-backend
```

2. **Criar e ativar o ambiente virtual**

```bash
   python -m venv venv
   venv\Scripts\activate
```

3. **Instalar dependências**

```bash
   pip install -r requirements.txt
```

4. **Executar o servidor**

```bash
   uvicorn main:app --reload
```
