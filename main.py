# from unstructured.partition.auto import partition
#
# elements = partition("test.pdf")
#
# print("\n\n".join([str(el) for el in elements]))
#
# print(elements)
# print(elements[0])
# print(elements[1])
# print(elements[2])
# print(elements[3])
# for i in range(20):
#     print(elements[i])
#     print(str(elements[i]))

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import NLTKTextSplitter, SentenceTransformersTokenTextSplitter
os.environ['OPENAI_API_KEY'] = "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"


# text_splitter = RecursiveCharacterTextSplitter(
#     # Set a really small chunk size, just to show.
#     chunk_size = 50,
#     chunk_overlap  = 10,
#     length_function = len,
#     add_start_index = True,
# )

text_splitter = SentenceTransformersTokenTextSplitter(tokens_per_chunk=100, chunk_overlap=10)

# loader = PyPDFLoader("test2.pdf")
loader = UnstructuredFileLoader("Professional-Services-Agreement-Nov-18-2022.pdf")
pages = loader.load_and_split(text_splitter=text_splitter)
print(type(pages))
print(len(pages))
faiss_index = FAISS.from_documents(pages, OpenAIEmbeddings())
docs = faiss_index.similarity_search("Consequence of a change in control of the company", k=5)
print(docs)
for doc in docs:
    print(str(doc.page_content))
    print(("\n"))





