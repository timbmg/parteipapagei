from pathlib import Path
from dataclasses import dataclass

import chromadb
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.ingestion import IngestionPipeline
from llama_index.readers.file import FlatReader
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from simple_parsing import ArgumentParser

@dataclass
class Args:
    persist_dir: str = "./chroma-db"
    chroma_collection: str = "wahlprogramme"
    override: bool = False
    base_dir: str = "data/clean"

def main(args):
    db = chromadb.PersistentClient(path=args.persist_dir)
    chroma_collection = db.get_or_create_collection(chroma_collection)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    markdown_documents = []
    for markdown_file in Path(args.base_dir).rglob("*.md"):
        print(markdown_file)
        markdown_document = FlatReader().load_data(Path(markdown_file))
        # add party metdata
        for doc in markdown_document:
            doc.metadata["party"] = markdown_file.stem
        markdown_documents.extend(markdown_document)

    markdown_parser = MarkdownNodeParser()
    sentence_splitter = SentenceSplitter(
        chunk_size=1024, chunk_overlap=0, paragraph_separator="\n"
    )
    pipeline = IngestionPipeline(transformations=[markdown_parser, sentence_splitter])
    nodes = pipeline.run(documents=markdown_documents, show_progress=True)
    index = VectorStoreIndex(
        nodes,
        show_progress=True,
    )
    index.storage_context.persist(args.persist_dir)


if __name__ == "__main__": 
    args = ArgumentParser().parse_args()
    main(args)
