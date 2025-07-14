from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings


def load_pdfs_from_directory(pdf_dir: str):
    """
    Load all PDF documents from the given directory using PyPDFDirectoryLoader.

    Args:
        pdf_dir (str): Path to the directory containing PDF files.

    Returns:
        List[Document]: List of loaded document objects.
    """
    loader = PyPDFDirectoryLoader(pdf_dir)
    documents = loader.load()
    return documents


# def split_documents_into_chunks(documents, chunk_size=500, chunk_overlap=20):
#     """
#     Splits documents into text chunks using RecursiveCharacterTextSplitter.

#     Args:
#         documents (List[Document]): List of document objects to split.
#         chunk_size (int, optional): Max size of chunks. Default is 500.
#         chunk_overlap (int, optional): Number of chars to overlap between chunks. Default is 20.

#     Returns:
#         Tuple[List[Document], List[str]]:
#             - The list of chunked Document objects
#             - The list of chunk text content strings
#     """
#     splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
#     text_chunks = splitter.split_documents(documents)
#     # texts = [chunk.page_content for chunk in text_chunks]
#     return text_chunks,  #texts

def split_documents_into_chunks(documents, chunk_size=500, chunk_overlap=20):
    """
    Splits LangChain Documents into smaller chunks with page_content attribute.
    Returns a flat list of Document objects.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_docs = splitter.split_documents(documents)
    # Make sure it's flat and all are Document objects
    flattened = []
    for item in split_docs:
        if isinstance(item, list):
            flattened.extend(item)
        else:
            flattened.append(item)
    # Verify all are Document instances
    assert all(isinstance(doc, Document) for doc in flattened), "Not all chunks are Document instances!"
    return flattened


def download_hf_embedding(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """
    Download and return a HuggingFace embedding model.

    Args:
        model_name (str, optional): Pretrained model name to load.
            Defaults to "sentence-transformers/all-MiniLM-L6-v2".

    Returns:
        HuggingFaceEmbeddings: Embeddings model instance.
    """
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings

