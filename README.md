# Chatbot_with_pinecone_vectordb
This ChatBot is designed to interactively answer questions based on a combination of pre-defined knowledge and language model predictions. It integrates several components for text processing, embedding generation, question-answering, and user interaction.
Components
1. Document Loading and Processing

    Document Loader: Utilizes TextLoader from langchain.document_loaders to load text data from a file.
    Text Splitter: Employs CharacterTextSplitter from langchain.text_splitter to split the loaded text into smaller chunks for processing.

2. Embedding Generation

    Hugging Face Embeddings: Utilizes HuggingFaceEmbeddings from langchain. embeddings to generate embeddings for text chunks.

3. Vector Store Management

    Pinecone: Integrates Pinecone vector database service for storing and retrieving embeddings efficiently. It creates a Pinecone index named "langchain-demo" if it doesn't exist and populates it with the generated embeddings.

4. Language Model for Question-Answering

    Hugging Face LLM (Large Language Model): Utilizes HuggingFaceHub from langchain.llms to access a pre-trained language model (specified by repo_id) hosted on the Hugging Face Hub. It generates responses to user questions based on the provided prompts.

5. User Interaction

    Prompt Template: Defines a template for user interaction using PromptTemplate from langchain. prompts. It prompts the user to ask a question and formats the input question for further processing.
    Runnable Passthrough: This represents a placeholder for passing input data directly to the next component in the processing pipeline.
    Output Parser: Converts the output of the language model into a human-readable format using StrOutputParser from langchain.schema.output_parser.

Usage

    Make sure you have set up the required environment variables:
        PINECONE_API_KEY: API key for Pinecone.
        HUGGINGFACE_API_KEY: API token for Hugging Face Hub.

    Run the provided Python script to instantiate the ChatBot and start the question-answering loop:

python main.py

Interact with the ChatBot by entering questions. Type 'exit' to end the session.
