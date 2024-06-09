import faiss
import os
import streamlit as st
import pickle
import langchain
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
from langchain.docstore.document import Document


# Load environment variables
from secret_key import openAPI_key
os.environ['OPENAI_API_KEY'] = openAPI_key

st.title("News Proximity")
st.sidebar.title("News Article URLs")
urls = []
for i in range(6):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url = st.sidebar.button("Process URL")
main_placefolder = st.empty()

if process_url:
    # Loading data
    try:
        loader = UnstructuredURLLoader(urls=urls)
        main_placefolder.text("Loading data....")
        data = loader.load()
        if not data:
            st.error("No data loaded from the provided URLs.")
        #else:
            #st.write(f"Loaded {len(data)} documents.")
    except Exception as e:
        st.error(f"Error loading data: {e}")
        data = []

    # Split data
    if data:
        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n', '.', ',', '\n\n'],
            chunk_size=1000,
            chunk_overlap=200
        )
        docs = text_splitter.split_documents(data)

        if not docs:
            st.error("No documents found after splitting the data.")
        #else:
            #st.write(f"Split into {len(docs)} documents.")

        # Create embeddings
        try:
            embeddings = OpenAIEmbeddings()
            vector_base = FAISS.from_documents(docs, embeddings)

            # Save FAISS index to a file
            index_filename = "vector_base.index"
            faiss.write_index(vector_base.index, index_filename)

            # Save the other metadata
            metadata = {
                "document_texts": [doc.page_content for doc in docs],  # Extract the texts
                "embeddings_function_name": embeddings.__class__.__name__  # Save the class name of the embeddings
            }
            with open("vector_base_metadata.pkl", "wb") as f:
                pickle.dump(metadata, f)

            #st.write(f"FAISS index and metadata have been saved to {index_filename} and vector_base_metadata.pkl")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

query = main_placefolder.text_input("Question: ")
if query:
    try:
        # Load FAISS index
        index_filename = "vector_base.index"
        index = faiss.read_index(index_filename)

        # Load metadata
        metadata_filename = "vector_base_metadata.pkl"
        with open(metadata_filename, "rb") as f:
            metadata = pickle.load(f)

        # Recreate embeddings function
        embeddings_class_name = metadata["embeddings_function_name"]
        embeddings = getattr(langchain.embeddings, embeddings_class_name)()

        # Recreate document texts
        document_texts = metadata["document_texts"]
        documents = [Document(page_content=text, metadata={"source": "unknown"}) for text in document_texts]  # Ensure each document has a "source" metadata field

        # Create a new FAISS vector base object
        docstore = InMemoryDocstore(dict(enumerate(documents)))
        index_to_docstore_id = {i: i for i in range(len(documents))}
        vector_base = FAISS(
            index=index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id,
            embedding_function=embeddings.embed_query
        )

        #st.write("FAISS index and metadata have been loaded successfully.")

        # Perform the query
        retriever = vector_base.as_retriever()
        # Initialize the LLM
        llm = ChatOpenAI(model="gpt-3.5-turbo")  # Use the chat model

        # Create the QA chain
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=retriever)
        main_placefolder.text("generating answer....")
        result = chain({"question": query}, return_only_outputs=True)
        st.header("Answer:")
        st.subheader(result["answer"])

    except Exception as e:
        st.error(f"An unexpected error occurred while loading the index and metadata: {e}")
