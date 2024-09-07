import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import SKLearnVectorStore
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from datetime import datetime


# Set API keys and environment variables
def set_env_variables():
    os.environ["OPENAI_API_KEY"] = (
        "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"
    )
    os.environ["LANGCHAIN_API_KEY"] = (
        "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"
    )
    os.environ["TAVILY_API_KEY"] = "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"
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGCHAIN_PROJECT"] = "govhack-rag-model"


# Function to load documents from a list of URLs
def load_documents():
    urls = [
        "https://www.stylemanual.gov.au/grammar-punctuation-and-conventions/titles-honours-forms-address/academics-and-professionals",
        "https://www.stylemanual.gov.au/grammar-punctuation-and-conventions/titles-honours-forms-address/australian-defence-force",
        "https://www.stylemanual.gov.au/grammar-punctuation-and-conventions/titles-honours-forms-address/awards-and-honours",
        "https://www.stylemanual.gov.au/grammar-punctuation-and-conventions/titles-honours-forms-address/diplomats",
        "https://www.stylemanual.gov.au/grammar-punctuation-and-conventions/titles-honours-forms-address/judiciary",
        "https://www.stylemanual.gov.au/grammar-punctuation-and-conventions/titles-honours-forms-address/parliaments-and-councils",
        "https://www.stylemanual.gov.au/grammar-punctuation-and-conventions/titles-honours-forms-address/royalty-vice-royalty-and-nobility",
    ]
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]
    return docs_list


# Function to split documents into chunks
def split_documents(docs_list):
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0
    )
    return text_splitter.split_documents(docs_list)


# Function to initialize the vector store and the retriever
def setup_vectorstore_and_retriever(doc_splits):
    vectorstore = SKLearnVectorStore.from_documents(
        documents=doc_splits,
        embedding=HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        ),
    )
    retriever = vectorstore.as_retriever(k=4)
    return retriever


def get_prompts(type):
    if type == "rewrite":
        prompt = PromptTemplate(
            template="""
        Your function is to rewrite or paraphrase the text based on the type of text based on the Australian Government Style Manual. 
        """, input_variables=["question", "documents"],
        )
    elif type == "feedback":
        prompt = PromptTemplate(
            """   
            Your function is to provide feedback on the inputs. 

            You will also give a scoring criteria (grammar, punctuation, titles, and honours) on a scale of 1 to 10 based on the Australian Government Style Manual. 
            """, input_variables=["question", "documents"],
        )

    return prompt


# Function to initialize the LLM and RAG chain
def setup_rag_chain(type):
    prompt_to_use = get_prompts(type)
    print(prompt_to_use)
    llm = ChatOllama(
        model="llama3.1",
        temperature=0,
    )

    return prompt_to_use | llm | StrOutputParser()


# Function to get an answer to a question based on retrieved documents
def answer_question(question, retriever, rag_chain):
    # Retrieve relevant documents from the vector store based on the question
    retrieved_docs = retriever.get_relevant_documents(question)

    # Concatenate the retrieved documents' content
    documents = "\n".join([doc.page_content for doc in retrieved_docs])

    # Define the input for the chain
    input_values = {"question": question, "documents": documents}
    print(input_values)
    # Generate the output by running the chain with the input values
    output = rag_chain.invoke(input_values)
    return output


def calculateDuration(start, end):
    duration = end - start
    duration_in_s = duration.total_seconds()
    return duration_in_s


# Main function to initialize and run the processes
def provide_output(question, type):
    # Set environment variables
    start = datetime.now()
    set_env_variables()
    end = datetime.now()
    duration_env = calculateDuration(start, end)
    print(f"Setting environment variables took {duration_env} seconds")

    start = datetime.now()
    docs_list = load_documents()
    end = datetime.now()
    duration_load = calculateDuration(start, end)
    print(f"Loading documents took {duration_load} seconds")

    start = datetime.now()
    doc_splits = split_documents(docs_list)
    end = datetime.now()
    duration_split = calculateDuration(start, end)
    print(f"Splitting documents took {duration_split} seconds")

    start = datetime.now()
    retriever = setup_vectorstore_and_retriever(doc_splits)
    end = datetime.now()
    duration_setup = calculateDuration(start, end)
    print(f"Setting up the retriever took {duration_setup} seconds")

    # Setup the RAG
    start = datetime.now()
    rag_chain = setup_rag_chain(type)
    end = datetime.now()
    duration_rag = calculateDuration(start, end)
    print(f"Setting up the RAG chain took {duration_rag} seconds")


    question_to_answer = question
    # Get the answer
    start = datetime.now()
    answer = answer_question(question_to_answer, retriever, rag_chain)
    end = datetime.now()
    duration_answer = calculateDuration(start, end)
    print(f"Answering the question took {duration_answer} seconds")

    return answer
