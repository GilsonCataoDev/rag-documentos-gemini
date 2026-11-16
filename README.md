# 📚 Assistente de Q&A em Documentos com RAG e Gemini

Este repositório contém a implementação de um pipeline de **Retrieval-Augmented Generation (RAG)** em Python utilizando **LangChain**, **ChromaDB** e a API do **Google Gemini**.

O sistema permite indexar documentos em formato PDF e realizar consultas contextuais em linguagem natural, garantindo respostas fundamentadas e com prevenção de alucinações.

---

## 🏗️ Arquitetura do Sistema

```
[PDF (Documento)]
       │
       ▼
[Chunking Recursivo (1000 chars / 200 overlap)]
       │
       ▼
[Embeddings Vetoriais (gemini-embedding-001)]
       │
       ▼
[Vector Store Local (ChromaDB)]
       │
       ▼ (Busca por Similaridade Semântica)
[Contexto Relevante (Top-K) + Pergunta]
       │
       ▼
[LLM (Gemini 3.8 Flash)] ➔ Resposta Fundamentada
```

---

## 🛠️ Tecnologias Utilizadas

- **Linguagem:** Python 3.11+
- **Orquestração de LLMs:** LangChain (LCEL)
- **Banco Vetorial:** ChromaDB
- **Embeddings:** `gemini-embedding-001` (Google GenAI)
- **Modelo de Linguagem:** `gemini-3.8-flash`
- **Manipulação de PDFs:** PyPDF

---

## ⚙️ Como Executar o Projeto

### 1. Clonar o repositório e acessar a pasta
```bash
git clone https://github.com/GilsonCataoDev/rag-documentos-gemini.git
cd rag-documentos-gemini
```

### 2. Criar e ativar o ambiente virtual
```bash
# Windows
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar as dependências
```bash
pip install -r requirements.txt
```

### 4. Configurar as variáveis de ambiente
Crie um arquivo `.env` baseado no `.env.example`:
```env
GEMINI_API_KEY="SUA_CHAVE_AQUI"
```

### 5. Executar a aplicação
Coloque seu documento PDF na pasta com o nome `documento.pdf` e execute:
```bash
python rag_basico.py
```

---

## 💡 Decisões Técnicas e Aprendizados

- **Text Splitting Recursivo:** Utilização do `RecursiveCharacterTextSplitter` para garantir cortes semânticos em parágrafos e frases, evitando perda de contexto nas bordas dos blocos.
- **Grounding e Prevenção de Alucinação:** Configuração estrita de prompt instruindo o modelo a responder exclusivamente com base nos trechos recuperados.
- **Arquitetura Desacoplada:** Código estruturado com LangChain Expression Language (LCEL), facilitando a troca futura de provedor de LLM ou de banco vetorial.
