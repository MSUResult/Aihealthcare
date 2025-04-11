import os
import platform
import asyncio
import streamlit as st

# ✅ Fix Windows asyncio event loop policy
if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# ✅ Disable CUDA for CPU-only compatibility
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# ✅ Langchain imports
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

# ✅ Vectorstore path
DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    try:
        if not os.path.isdir(DB_FAISS_PATH):
            raise FileNotFoundError(f"Vectorstore path '{DB_FAISS_PATH}' does not exist.")

        embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
        return db
    except Exception as e:
        st.error(f"❌ Failed to load vectorstore: {e}")
        return None

def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

def load_llm(huggingface_repo_id, HF_TOKEN):
    if not HF_TOKEN:
        st.error("❌ Hugging Face Token (HF_TOKEN) not set in environment or secrets.")
        raise ValueError("Missing HF_TOKEN")

    return HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        max_length=512,
        huggingfacehub_api_token=HF_TOKEN
    )

def main():
    st.title("🧠 Ask Chatbot (MediBot)")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

    prompt = st.chat_input("Pass your prompt here")

    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        CUSTOM_PROMPT_TEMPLATE = """
        Use the pieces of information provided in the context to answer the user's question.
        If you don't know the answer, say you don't know. Don't make up any answer.
        Stick strictly to the context.

        Context: {context}
        Question: {question}

        Start the answer directly. Avoid greetings or closing remarks.
        """

        HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
        HF_TOKEN = os.environ.get("HF_TOKEN") or st.secrets.get("HF_TOKEN")

        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                return

            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(HUGGINGFACE_REPO_ID, HF_TOKEN),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            response = qa_chain.invoke({'query': prompt})
            result = response["result"]
            source_documents = response["source_documents"]

            formatted_sources = "\n\n".join(
                [f"- `{doc.metadata.get('source', 'No source')}`: {doc.page_content[:200]}..." for doc in source_documents]
            )

            final_response = f"{result}\n\n📚 **Source Documents:**\n{formatted_sources}"
            st.chat_message("assistant").markdown(final_response)
            st.session_state.messages.append({'role': 'assistant', 'content': final_response})

        except Exception as e:
            st.error(f"🚨 An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
