import os 

os.environ["OPENAI_API_KEY"] = "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"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["TAVILY_API_KEY"] = "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"

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "govhack-rag-model"

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_core.tools import tool
from langchain.embeddings import HuggingFaceEmbeddings


# List of URLs to load documents from
urls = [
    "https://www.stylemanual.gov.au/grammar-punctuation-and-conventions/titles-honours-forms-address/academics-and-professionals",
    "https://www.stylemanual.gov.au/grammar-punctuation-and-conventions/titles-honours-forms-address/australian-defence-force",
    "https://www.stylemanual.gov.au/grammar-punctuation-and-conventions/titles-honours-forms-address/awards-and-honours",
    "https://www.stylemanual.gov.au/grammar-punctuation-and-conventions/titles-honours-forms-address/diplomats",
    "https://www.stylemanual.gov.au/grammar-punctuation-and-conventions/titles-honours-forms-address/judiciary",
    "https://www.stylemanual.gov.au/grammar-punctuation-and-conventions/titles-honours-forms-address/parliaments-and-councils",
    "https://www.stylemanual.gov.au/grammar-punctuation-and-conventions/titles-honours-forms-address/royalty-vice-royalty-and-nobility"
]

# Load documents from the URLs
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

# Initialize a text splitter with specified chunk size and overlap
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)

# Split the documents into chunks
doc_splits = text_splitter.split_documents(docs_list)

# Add the document chunks to the "vector store" using NomicEmbeddings
vectorstore = SKLearnVectorStore.from_documents(
    documents=doc_splits,
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
)
retriever = vectorstore.as_retriever(k=4)

from langchain.schema import Document
from langchain_community.tools.tavily_search import TavilySearchResults

web_search_tool = TavilySearchResults()

from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = PromptTemplate(
    template="""You are an assistant for question-answering tasks. 

    Your functions are to paraphrase based on the Australian Government Style Manual. 

    You must also cite the relevant sources with websites that you used to paraphrase. 
    
    If you don't know the answer, just say that you don't know. 
    
    Question: {question} 
    Documents: {documents} 
    Answer: 
    """,
    input_variables=["question", "documents"],
)

llm = ChatOllama(
    model="llama3.1",
    temperature=0,
)

rag_chain = prompt | llm | StrOutputParser()

# Example question
question = '''
A super pension is a series of regular payments made as a super income stream. This doesn't include government payments such as the age pension.

You may receive these payments:

from an Australian super fund, life assurance company or retirement savings account (RSA) provider
from a fund established for the benefit of Commonwealth, state or territory employees and their dependants, such as
the Commonwealth Superannuation Scheme
the Public Sector Superannuation Scheme
as a result of another person's death (death benefit income stream).
Depending on your age and the type of income stream you receive, you may need to declare different items in your tax return. This includes:

a taxed element the part of your benefit on which tax has already been paid in the fund
an untaxed element the part of your benefit that is still taxable because tax has not been paid in the fund
a tax-free component the part of your benefit that is tax-free.
Your PAYG payment summary superannuation income stream from your super fund will show the amount you need to declare in your tax return. We pre-fill the amounts from your payment summary when you prepare and lodge you tax return online.
'''

# Retrieve relevant documents from the vector store based on the question
retrieved_docs = retriever.get_relevant_documents(question)

# Concatenate the retrieved documents' content
documents = "\n".join([doc.page_content for doc in retrieved_docs])

# Define the input for the chain
input_values = {
    "question": question,
    "documents": documents
}

# Generate the output by running the chain with the input values
output = rag_chain.invoke(input_values)

print("--------------------------------------------")
print(output)
