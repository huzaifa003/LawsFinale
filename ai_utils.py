import os
import torch
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.vectorstores import Chroma
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers import ContextualCompressionRetriever

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Initialize embeddings with GPU support
embeddings = HuggingFaceEmbeddings(
    model_name="Omartificial-Intelligence-Space/GATE-AraBert-v1"
)

def load_docx(file_path):
    """Load a single .docx file."""
    try:
        loader = Docx2txtLoader(file_path)
        documents = loader.load()
        print(f"Loaded {len(documents)} documents from {file_path}")
        return documents
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return []

def get_docx_files(folder_path):
    """Retrieve all .docx file paths from the specified folder."""
    docx_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(".docx"):
                full_path = os.path.join(root, file)
                docx_files.append(full_path)
                print(f"Found .docx file: {full_path}")
    return docx_files

def load_documents_from_folder(folder_path):
    """Load all documents from multiple .docx files in a folder."""
    docx_files = get_docx_files(folder_path)
    all_docs = []
    for file in docx_files:
        docs = load_docx(file)
        all_docs.extend(docs)
    print(f"Total documents loaded: {len(all_docs)}")
    return all_docs

def chunk_documents(docs):
    """Split documents into semantic chunks."""
    try:
        splitter = SemanticChunker(embeddings)
        chunks = splitter.split_documents(docs)
        print(f"Number of chunks created: {len(chunks)}")
        return chunks
    except Exception as e:
        print(f"Error during chunking: {e}")
        return []

def save_chunks(chunks, folder_path):
    """Save each chunk as a separate text file in the specified folder."""
    os.makedirs(folder_path, exist_ok=True)
    for i, chunk in enumerate(chunks):
        chunk_filename = os.path.join(folder_path, f"chunk_{i}.txt")
        try:
            with open(chunk_filename, "w", encoding="utf-8") as f:
                f.write(chunk.page_content)
            print(f"Saved chunk {i} to {chunk_filename}")
        except Exception as e:
            print(f"Error saving chunk {i}: {e}")
    print(f"All chunks have been saved to {folder_path}")

def store_chunks_in_vector_db(chunks, persist_dir="vector_db"):
    """Store chunks in a Chroma vector database."""
    try:
        # Initialize Chroma with embeddings
        vector_db = Chroma.from_documents(chunks, embeddings, persist_directory=persist_dir)
        print(f"Vector database created at {persist_dir}")

        # Set up retriever with cross-encoder reranking
        vectorstore_retriever = vector_db.as_retriever(search_kwargs={"k": 3})
        model = HuggingFaceCrossEncoder(model_name="Omartificial-Intelligence-Space/ARA-Reranker-V1")
        compressor = CrossEncoderReranker(model=model, top_n=3)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=vectorstore_retriever
        )
        print("Vector database retriever with reranking is set up.")
        
        return vector_db, compression_retriever
    except Exception as e:
        print(f"Error storing chunks in vector DB: {e}")
        return None, None



def process_docs(input_folder, chunks_folder, vector_db_dir):
    """Main function to process .docx files, chunk them, save chunks, and store in vector DB."""
    # Step 1: Load all documents from the input folder
    print("Loading documents from folder...")
    documents = load_documents_from_folder(input_folder)
    
    if not documents:
        print("No documents to process. Exiting.")
        return

    # Step 2: Chunk the documents
    print("Chunking documents...")
    chunks = chunk_documents(documents)
    
    if not chunks:
        print("No chunks created. Exiting.")
        return

    # Step 3: Save chunks to the specified folder
    print("Saving chunks to folder...")
    save_chunks(chunks, chunks_folder)

    # Step 4: Store chunks in the vector database
    print("Storing chunks in vector database...")
    vector_db, retriever = store_chunks_in_vector_db(chunks, persist_dir=vector_db_dir)
    
    if vector_db:
        print("Processing completed successfully.")
    else:
        print("Processing encountered issues during vector DB storage.")


# def get_vector_db(persist_dir="vector_db"):
#     """Retrieve the vector database from the specified directory."""
#     print(f"Retrieving vector database from {persist_dir}")
#     return Chroma.from_documents([], embeddings, persist_directory=persist_dir)

def load_chunks_from_folder(folder_path):
    """Load all chunk files from a folder."""
    chunk_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(".txt"):
                full_path = os.path.join(root, file)
                chunk_files.append(full_path)
                print(f"Found chunk file: {full_path}")
    return chunk_files

def get_compression_retriever(persist_dir="vector_db", chunks_folder="chunks"):
    try: 
        """Retrieve the compression retriever from the specified directory."""
        # chunks = load_chunks_from_folder(chunks_folder)
        # print(f"Loaded {len(chunks)} chunks from {chunks_folder}")
        # print(chunks)
        # print(f"Retrieving compression retriever from {persist_dir}")
        vector_db: Chroma = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
        
        vectorstore_retriever = vector_db.as_retriever(search_kwargs={"k": 3})
        
        
        print(vectorstore_retriever)
        model = HuggingFaceCrossEncoder(model_name="Omartificial-Intelligence-Space/ARA-Reranker-V1", )
        compressor = CrossEncoderReranker(model=model, top_n=3)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=vectorstore_retriever
        )

        print(compression_retriever)
        return compression_retriever
    except Exception as e:
        print(f"Error retrieving compression retriever: {e}")
        return None



def query_retreiver(query):
    vector_db : Chroma = Chroma(persist_directory="VectorDB", embedding_function=embeddings)
    reponse = vector_db.similarity_search_with_score(query)
    return reponse