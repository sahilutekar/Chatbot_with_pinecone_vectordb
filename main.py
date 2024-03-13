from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import HuggingFaceHub
import pinecone
from dotenv import load_dotenv
import os
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser


class ChatBot:
    def __init__(self):
        load_dotenv()

        # Handle potential file not found error
        try:
            self.loader = TextLoader('./imp_doc-_Repaired_.txt')  # Or './gptt.txt'
        except FileNotFoundError:
            print("Error: Horoscope text file not found. Please check the path.")
            return

        self.documents = self.loader.load()
        self.text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=4)
        self.docs = self.text_splitter.split_documents(self.documents)

        self.embeddings = HuggingFaceEmbeddings()

        # Gracefully handle Pinecone API key error
        try:
            pinecone.init(
                api_key=os.getenv('PINECONE_API_KEY'),
                environment='gcp-starter'
            )
        except KeyError:
            print("Error: PINECONE_API_KEY environment variable not set.")
            return

        self.index_name = "langchain-demo"

        if self.index_name not in pinecone.list_indexes():
            pinecone.create_index(name=self.index_name, metric="cosine", dimension=768)
            self.docsearch = Pinecone.from_documents(self.docs, self.embeddings, index_name=self.index_name)
        else:
            self.docsearch = Pinecone.from_existing_index(self.index_name, self.embeddings)

        self.repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"

        # Handle potential HuggingFace API key error
        try:
            self.llm = HuggingFaceHub(
                repo_id=self.repo_id,
                model_kwargs={"temperature": 0.5, "top_p": 0.95, "top_k": 150},
                huggingfacehub_api_token=os.getenv('HUGGINGFACE_API_KEY')
            )
        except KeyError:
            print("Error: HUGGINGFACE_API_KEY environment variable not set.")
            return


        # Option 1: LLM generates entire response (no answer variable)
        self.template = PromptTemplate(template="""
            You are a seer. Ask me a question about your life, and I will use my knowledge to answer it.

            Question: {question}

            """, input_variables=["question"])

        

        self.rag_chain = (
            {"question": RunnablePassthrough()}
            | self.template
            | self.llm
            | StrOutputParser()
        )

    def ask_question_loop(self):
        while True:
            # Get user input
            user_input = input("hello , how can i help: ")

            # Check if the user wants to exit
            if user_input.lower() == 'exit':
                print("Exiting...")
                break

            # Invoke the RAG chain to get the answer
            data = {"question": user_input}  # Pass only question
            result = self.rag_chain.invoke(data)

            # Print the answer
            print(result)


# Instantiate the ChatBot and start the question-answering loop
bot = ChatBot()
bot.ask_question_loop()
