import os
from dotenv import load_dotenv

# 1. Carregadores e divisores de texto
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 2. Banco vetorial e Embeddings
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# 3. Componentes de Prompt e Fluxo
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Lê a chave do arquivo .env
load_dotenv()


def main():
    caminho_pdf = "documento.pdf"

    # Verifica se você colocou um PDF na pasta
    if not os.path.exists(caminho_pdf):
        print(
            f"❌ Atenção: Coloque um arquivo PDF chamado '{caminho_pdf}' nesta pasta!"
        )
        return

    print("📄 1. Lendo o PDF...")
    loader = PyPDFLoader(caminho_pdf)
    documentos = loader.load()

    print("✂️ 2. Fatiando o texto em pedaços...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documentos)
    print(f"   ➔ Criados {len(chunks)} trechos de texto.")

    print("🧠 3. Criando banco vetorial com embeddings...")
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    vector_store = Chroma.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    print("🤖 4. Conectando ao Gemini...")
    llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0.2)

    template = """Você é um assistente prestativo. Responda à pergunta baseando-se APENAS no
    contexto abaixo.
Se a informação não estiver no contexto, responda: "Não encontrei essa informação no
documento."

Contexto:
{context}

Pergunta:
{question}

Resposta:"""

    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Constrói o pipeline RAG
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("\n✅ Pronto! Faça uma pergunta sobre o documento (ou digite 'sair'):")
    while True:
        pergunta = input("\nVocê: ")
        if pergunta.lower() in ["sair", "exit", "quit"]:
            break
        resposta = rag_chain.invoke(pergunta)
        print(f"\nAssistente:\n{resposta}")


if __name__ == "__main__":
    main()
