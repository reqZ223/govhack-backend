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


def get_urls(department):
    if department == "home_affairs":
        grammar_urls = [
        "https://www.stylemanual.gov.au/grammar-punctuation-and-conventions/types-words",
        "https://www.stylemanual.gov.au/grammar-punctuation-and-conventions/parts-sentences",
        "https://www.stylemanual.gov.au/grammar-punctuation-and-conventions/spelling",
        "https://www.stylemanual.gov.au/grammar-punctuation-and-conventions/titles-honours-forms-address",
        "https://www.stylemanual.gov.au/grammar-punctuation-and-conventions/types-words/adjectives",
        "https://www.stylemanual.gov.au/grammar-punctuation-and-conventions/types-words/adverbs",
        "https://www.stylemanual.gov.au/grammar-punctuation-and-conventions/types-words/conjunctions",
        "https://www.stylemanual.gov.au/grammar-punctuation-and-conventions/types-words/nouns",
        "https://www.stylemanual.gov.au/grammar-punctuation-and-conventions/types-words/prepositions",
        "https://www.stylemanual.gov.au/grammar-punctuation-and-conventions/types-words/pronouns",
        "https://www.stylemanual.gov.au/grammar-punctuation-and-conventions/types-words/verbs",
        "https://www.stylemanual.gov.au/grammar-punctuation-and-conventions/parts-sentences/clauses",
        "https://www.stylemanual.gov.au/grammar-punctuation-and-conventions/parts-sentences/phrases",
        "https://www.stylemanual.gov.au/grammar-punctuation-and-conventions/punctuation",
        ]
        structuring_content_urls = [
            "https://www.stylemanual.gov.au/structuring-content/paragraphs",
            "https://www.stylemanual.gov.au/structuring-content/types-structure",
            "https://www.stylemanual.gov.au/structuring-content/headings",
            "https://www.stylemanual.gov.au/structuring-content/links",
            "https://www.stylemanual.gov.au/structuring-content/lists",
            "https://www.stylemanual.gov.au/structuring-content/tables",
            "https://www.stylemanual.gov.au/structuring-content/text-boxes-and-callouts",
            "https://www.stylemanual.gov.au/structuring-content/types-structure/sequential-structure",
            "https://www.stylemanual.gov.au/structuring-content/types-structure/inverted-pyramid-structure",
            "https://www.stylemanual.gov.au/structuring-content/types-structure/hierarchical-structure",
            "https://www.stylemanual.gov.au/structuring-content/types-structure/narrative-structure",
        ]
        writing_and_designing_content_urls = [
            "https://www.stylemanual.gov.au/writing-and-designing-content/clear-language-and-writing-style/plain-language-and-word-choice",
            "https://www.stylemanual.gov.au/writing-and-designing-content/clear-language-and-writing-style/sentences",
            "https://www.stylemanual.gov.au/writing-and-designing-content/findable-content",
            "https://www.stylemanual.gov.au/writing-and-designing-content/security-classifications-and-protective-markings",
            "https://www.stylemanual.gov.au/writing-and-designing-content/clear-language-and-writing-style/voice-and-tone",
        ]
        content_types_urls = [
            "https://www.stylemanual.gov.au/content-types/easy-read",
            "https://www.stylemanual.gov.au/content-types/emails-and-letters",
            "https://www.stylemanual.gov.au/content-types/blogs",
            "https://www.stylemanual.gov.au/content-types/reports",
        ]
        referencing_and_attribution_urls = [
        "https://www.stylemanual.gov.au/referencing-and-attribution/author-date",
        "https://www.stylemanual.gov.au/referencing-and-attribution/documentary-note",
        "https://www.stylemanual.gov.au/referencing-and-attribution/legal-material",
        "https://www.stylemanual.gov.au/referencing-and-attribution/shortened-forms-used-referencing",
        "https://www.stylemanual.gov.au/referencing-and-attribution/author-date/broadcast-media-and-podcasts-film-video-television-and-radio-programs",
        "https://www.stylemanual.gov.au/referencing-and-attribution/author-date/classics",
        "https://www.stylemanual.gov.au/referencing-and-attribution/author-date/musical-compositions",
        "https://www.stylemanual.gov.au/referencing-and-attribution/author-date/plays-and-poetry",
        "https://www.stylemanual.gov.au/referencing-and-attribution/author-date/works-art",
        "https://www.stylemanual.gov.au/referencing-and-attribution/legal-material/bills-and-explanatory-material",
        "https://www.stylemanual.gov.au/referencing-and-attribution/legal-material/acts-parliament",
        "https://www.stylemanual.gov.au/referencing-and-attribution/legal-material/schedules",
        "https://www.stylemanual.gov.au/referencing-and-attribution/legal-material/delegated-legislation",
        "https://www.stylemanual.gov.au/referencing-and-attribution/legal-material/cases-and-legal-authorities",
        "https://www.stylemanual.gov.au/referencing-and-attribution/legal-material/treaties",
        "https://www.stylemanual.gov.au/referencing-and-attribution/legal-material/authoritative-reports",
    ]
        accessible_and_inclusive_content_urls = [
        "https://www.stylemanual.gov.au/accessible-and-inclusive-content/how-people-read",
        "https://www.stylemanual.gov.au/accessible-and-inclusive-content/inclusive-language/aboriginal-and-torres-strait-islander-peoples",
        "https://www.stylemanual.gov.au/accessible-and-inclusive-content/inclusive-language/age-diversity",
        "https://www.stylemanual.gov.au/accessible-and-inclusive-content/inclusive-language/cultural-and-linguistic-diversity",
        "https://www.stylemanual.gov.au/accessible-and-inclusive-content/inclusive-language/gender-and-sexual-diversity",
        "https://www.stylemanual.gov.au/accessible-and-inclusive-content/inclusive-language/people-disability",
        ]
        urls = ["https://www.stylemanual.gov.au/writing-and-designing-content/clear-language-and-writing-style/plain-language-and-word-choice"]
    elif department == "ATO":
        urls = [] # To add 
    elif department == "Treasury":
        urls = [] # to add
    else: 
        urls = [] # to add
    return urls


# Function to load documents from a list of URLs
def load_documents(department):
    department = department 
    urls = get_urls(department)
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]
    return docs_list


# Function to split documents into chunks
def split_documents(docs_list):

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=150, chunk_overlap=20
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
        Your function is to rewrite or paraphrase the text to plain australian english following the guidlines in the Australian Government Style Manual. 
        You will also provide the relevant citations when you rewrite the text at the end of your response. 

        Question: {question} 
        Documents: {documents}  

        """,
            input_variables=["question", "documents"],
        )
    elif type == "feedback":
        prompt = PromptTemplate(
            template="""   
            Your function is to provide feedback on the inputs. 

            You will also give a scoring criteria (grammar, punctuation, inclusive language, and plain language) on a scale of 1 to 10 based on the Australian Government Style Manual. 
            
            You will also provide the relevant citations for the feedback you provide at the end of your response.

            Question: {question} 
            Documents: {documents}  

            """,
            input_variables=["question", "documents"],
        )
    elif type == "ask": ## This is a sample prompt for now
        prompt = PromptTemplate(
            template="""
            You are an assistant for question-answering tasks on anything related to the Australian Government Style Manual.  
            You should also cite sources when you answer the question. 
            
            Question: {question} 
            Documents: {documents}    

            """, input_variables=["question", "documents"]
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
def provide_output(question, type, department):
    # Set environment variables
    start = datetime.now()
    set_env_variables()
    end = datetime.now()
    duration_env = calculateDuration(start, end)
    print(f"Setting environment variables took {duration_env} seconds")

    start = datetime.now()
    docs_list = load_documents(department)
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
