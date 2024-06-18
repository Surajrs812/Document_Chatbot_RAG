# Document_Chatbot_RAG

## Introduction

Document_Chatbot_RAG is an application that allows users to interact with the contents of PDF documents using natural language queries. The project leverages advanced techniques such as embeddings and vector stores to provide accurate and context-aware answers.

### Techniques Used

#### Embeddings
Embeddings are a way to represent text data in a numerical format. This project uses Google's Generative AI embeddings, which transform text into high-dimensional vectors that capture the semantic meaning of the content. These embeddings are crucial for understanding the context and relevance of the text in response to user queries.

#### Vector Store
A vector store is a database that stores these high-dimensional vectors. In this project, we use FAISS (Facebook AI Similarity Search) as the vector store. FAISS allows for efficient similarity searches, enabling the chatbot to quickly find relevant text chunks in the PDF documents that match the user's query.

#### RAG (Retrieval-Augmented Generation)
RAG is a method that combines retrieval of relevant documents with generative models to produce accurate and informative responses. This technique enhances the chatbot's ability to provide contextually appropriate answers by grounding its responses in the retrieved document content.

## How to Use

### Prerequisites
- Python 3.8 or higher
- Virtual environment (optional but recommended)

### Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/Document_Chatbot_RAG.git
   cd Document_Chatbot_RAG
   ```

2. **Create a Virtual Environment**
   ```bash
   python -m venv venv
   ```

3. **Activate the Virtual Environment**
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. **Install the Required Packages**
   ```bash
   pip install -r requirements.txt
   ```

5. **Set Up the API Key**
   - Create a `.env` file in the project root directory.
   - Add your Gemini API key to the `.env` file:
     ```
     GOOGLE_API_KEY=your_gemini_api_key
     ```

6. **Run the Application**
   ```bash
   streamlit run app.py
   ```

### Usage

1. **Upload PDF Files**
   - Use the sidebar to upload one or more PDF files.
   - Click on the "Submit & Process" button to process the uploaded files.

2. **Ask Questions**
   - Type your question in the text input field provided.
   - The chatbot will analyze the processed PDF content and return a relevant answer based on the context.


Document_Chatbot_RAG is a powerful tool for interacting with PDF documents using natural language. By leveraging embeddings and vector stores, it provides precise and context-aware answers to user queries. Enjoy using the chatbot and feel free to contribute to the project!

