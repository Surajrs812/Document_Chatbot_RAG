import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # Embedding technique
import google.generativeai as genai
from langchain.vectorstores import FAISS  # Vector Storage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Google Generative AI with API Key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    """Extracts text from a list of PDF files."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    """Splits text into manageable chunks for processing."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    """Creates and saves a FAISS vector store from text chunks."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    """Creates a conversational chain for answering questions."""
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details. 
    If the answer is not in the provided context, just say, "The answer is not available in the context", 
    don't provide a wrong answer.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def user_input(user_question):
    """Handles user questions and generates responses using the vector store and conversational chain."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Load the vector store with the necessary parameter for safe deserialization
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    # Display the response in a neat markdown format
    st.markdown("### Response:")
    st.markdown(response["output_text"])

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(page_title="DocumentChatbot")
    st.title("Document Chatbot")
    st.write("This application allows you to ask questions based on the content of PDF files you upload. The answers are generated using Gemini.")

    st.sidebar.title("Menu")
    st.sidebar.write("Upload your PDF files and process them to start asking questions.")
    pdf_docs = st.sidebar.file_uploader("Upload PDF Files", accept_multiple_files=True)
    if st.sidebar.button("Submit & Process"):
        with st.spinner("Processing..."):
            raw_text = get_pdf_text(pdf_docs)
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks)
            st.sidebar.success("Processing complete!")

    user_question = st.text_input("Ask a Question from the PDF Files")
    if user_question:
        user_input(user_question)

if __name__ == "__main__":
    main()
