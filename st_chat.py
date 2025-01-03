import os
import re
from functools import cache
from typing import Optional

import chromadb
import streamlit as st
import xxhash
from llama_index.core import (
    VectorStoreIndex,
    Settings,
    PromptTemplate,
    get_response_synthesizer,
)
import llama_index.core.instrumentation as instrument
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.bridge.pydantic import Field
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core import QueryBundle
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore
from llama_index.core.instrumentation.events.base import BaseEvent
from llama_index.core.instrumentation.event_handlers import BaseEventHandler
from llama_index.core.instrumentation.events.llm import LLMChatStartEvent
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.chroma import ChromaVectorStore

from party_data import party_data

os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
GEMINI_LLM_MODEL = "models/gemini-1.5-flash-002"
GEMINI_EMBEDDING_MODEL = "models/text-embedding-004"

QUERY_GEN_PROMPT = """Du bist ein hilfreicher Assistent, der basierend auf einer Frage 
eines Bürgers zum Wahlprogrammen einer Partei ähnliche Fragen erstellt. Überlege dir 
unterschiedliche Formulierungen und weitere Fragen um relevante Informationen aus dem
Wahlprogramm zu finden, die für die beantwortung der Frage nützlich sind. Generiere 
insgesamt {num_queries} Fragen, jede in einer eigenen Zeile, basierend auf der 
folgenden Frage: '{query}'\nGenerierte Fragen:\n"
"""

def create_anchor_from_text(text: str | None) -> str:
    # based on https://github.com/streamlit/streamlit/blob/833efa9fe408c692906bd07b201b5e715bcceae2/frontend/lib/src/components/shared/StreamlitMarkdown/StreamlitMarkdown.tsx#L121-L137
    new_anchor = ""
    
    # Check if the text is valid ASCII characters
    is_ascii = text and all(ord(c) < 128 for c in text)
    
    if is_ascii and text:
        new_anchor = text.lower().replace(" ", "-").replace("--", "-")
        new_anchor = re.sub(r"[.,:\!\?]", "", new_anchor)
    elif text:
        # If the text is not valid ASCII, use a hash of the text
        new_anchor = xxhash.xxh32(text, seed=0xabcd).hexdigest()[:16]
    
    return new_anchor

class ExampleEventHandler(BaseEventHandler):
    events: list[BaseEvent] = []

    def handle(self, event: BaseEvent, **kwargs) -> None:
        """Logic for handling event."""
        if isinstance(event, LLMChatStartEvent):
            print("-----------------------", flush=True)
            # all events have these attributes
            print(event.class_name(), flush=True)
            print(event.id_, flush=True)
            print(event.timestamp, flush=True)
            print(event.span_id, flush=True)

            # event specific attributes
            print(event.messages, flush=True)
            print(event.additional_kwargs, flush=True)
            print(event.model_dict, flush=True)
            print("-----------------------", flush=True)

        self.events.append(event)

dispatcher = instrument.get_dispatcher()
dispatcher.add_event_handler(ExampleEventHandler())


class RankCutoffPostprocessor(BaseNodePostprocessor):
    rank_cutoff: int = Field(default=10)

    def _postprocess_nodes(
        self, nodes: list[NodeWithScore], query_bundle: Optional[QueryBundle]
    ) -> list[NodeWithScore]:
        sorted_nodes = sorted(nodes, key=lambda x: x.score, reverse=True)
        print(f"CUTTING OFF from {len(sorted_nodes)} to {self.rank_cutoff}")
        return sorted_nodes[: self.rank_cutoff]


class CachedQueryFusionRetriever(QueryFusionRetriever):

    @cache
    def _get_queries(self, original_query: str):
        return super()._get_queries(original_query)


@st.cache_resource
def init_query_engines():

    CHROMA_PERSIST_DIR = "chroma"
    CHROMA_COLLECTION = "wahlprogramme"

    Settings.llm = Gemini(model=GEMINI_LLM_MODEL)
    Settings.embed_model = GeminiEmbedding(model=GEMINI_EMBEDDING_MODEL)
    
    db = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    chroma_collection = db.get_or_create_collection(CHROMA_COLLECTION)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    index = VectorStoreIndex.from_vector_store(vector_store)

    engines = {}
    for party in ["cdu", "afd", "fdp", "grüne", "spd"]:
        filters = MetadataFilters(
            filters=[
                MetadataFilter(key="party", value=party),
            ]
        )

        dense_retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=100,
            filters=filters,
        )
        bm25_retriever = BM25Retriever.from_persist_dir(f"bm25/{party}")
        bm25_retriever.similarity_top_k = 100

        fusion_retriever = CachedQueryFusionRetriever(
            [
                dense_retriever,
                bm25_retriever,
            ],
            mode="reciprocal_rerank",
            num_queries=4,
            use_async=False,
            similarity_top_k=10,
            verbose=True,
            query_gen_prompt=QUERY_GEN_PROMPT,
        )

        # configure response synthesizer
        response_synthesizer = get_response_synthesizer(streaming=True)

        # assemble query engine
        query_engine = RetrieverQueryEngine(
            retriever=fusion_retriever,
            response_synthesizer=response_synthesizer,
            node_postprocessors=[RankCutoffPostprocessor(rank_cutoff=10)],
        )
        text_qa_template = (
            "Du bist Politker der {party}. "
            "Ein Bürger möchte etwas über euer Wahlprogramm zur Bundestagswahl 2025 wissen. "
            "Relevante Passagen aus dem Wahlprogramm sind unten aufgeführt.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Beantworte die Frage basierend auf den Informationen im Wahlprogramm. "
            "Überlege wie die Informationen im Wahlprogramm auf die Frage des Bürgers anwendbar sind.\n"
            "Falls die Inhalte des Wahlprogramms nicht relevant sind, "
            "antworte dem Bürger das die {party} keine Position zu diesem Thema im Wahlprogramm hat.\n"
            "Füge jeder deiner Aussagen eine Referenz hinzu, auf welche Passage aus dem Wahlprogramm diese zurück zu führen ist. "
            "Verwende dazu die folgendes Format: <Antwort> [X, Y]' "
            "um zu belegen, dass deine Antwort sich auf Passagen X und Y zurück zu führen ist.\n"
            "Halten deine Antwort kurz und prägnant, um den Bürger zu überzeugen.\n"
            "Bürger Frage: {query_str}\n"
            "Deine Politker Antwort: "
        )

        def format_context_fn(**kwargs):
            # format context with reference numbers
            context_list = kwargs["context_str"].split("\n\n")
            fmtted_context = "\n\n".join(
                [f"[{i}] {c}" for i, c in enumerate(context_list, start=1)]
            )
            return fmtted_context

        text_qa_template = PromptTemplate(
            text_qa_template, function_mappings={"context_str": format_context_fn}
        ).partial_format(party=party_data[party]["template_name"])
        query_engine.update_prompts(
            {"response_synthesizer:text_qa_template": text_qa_template}
        )
        print(f"Initialized query engine for party {party}.")
        engines[party] = query_engine
    return engines


ENGINES = init_query_engines()


def response_generator(user_query: str, party: str):
    print(f"Party: {party} Query: {user_query}")
    response = ENGINES[party].query(user_query)

    for node in response.source_nodes:
        n = node.to_dict()
        text = n["node"]["text"]
        score = n["score"]
        id_ = n["node"]["id_"]
        header = n["node"]["metadata"]["header"]
        print(f"ID: {id_}, Score: {score:0.4f}, Header: {header[:50]}, Text: {text[:100]}")

    message_iter = iter(response.response_gen)
    while True:
        try:
            stream = next(message_iter)
        except StopIteration:
            break
        fragments = list(set(re.findall(r'\[\d+(?:,\s*\d+)*\]?|\[?\d+(?:,\s*\d+)*\]', stream)))
        for fragment in fragments:
            modified_fragment = ""
            last_char_was_digit = False
            parsed_number = ""
            for c in fragment:
                if c.isdigit():
                    parsed_number += c
                    last_char_was_digit = True
                else:
                    if last_char_was_digit:
                        _id = create_anchor_from_text(response.source_nodes[int(parsed_number) - 1].node.metadata["header"])
                        modified_fragment += f"[{parsed_number}]({party}#{_id})"
                    modified_fragment += c
                    last_char_was_digit = False
                    parsed_number = ""
            if last_char_was_digit:
                _id = create_anchor_from_text(response.source_nodes[int(parsed_number) - 1].node.metadata["header"])
                modified_fragment += f"[{parsed_number}]({party}#{_id})"
            stream = stream.replace(fragment, modified_fragment)

        yield stream


st.title("🗳️ ChatBTW")

st.segmented_control(
    label="Wähle Partein über deren Programm Du mehr erfahren willst.",
    options=[party for party, data in party_data.items() if data["enabled"]],
    format_func=lambda p: f"{party_data[p]['emoji']} {party_data[p]['name']}",
    selection_mode="multi",
    default=["cdu", "spd"],
    key="party_selection",
)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    if message["role"] == "assistant":
        chat_message_kwargs = {"avatar": party_data[message["party"]]["emoji"]}
    else:
        chat_message_kwargs = {}
    with st.chat_message(message["role"], **chat_message_kwargs):
        st.markdown(message["content"])

if prompt := st.chat_input(
    f"Deine Frage an ChatBTW",
):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    for party in st.session_state.party_selection:
        with st.chat_message("assistant", avatar=party_data[party]["emoji"]):
            response_stream = response_generator(user_query=prompt, party=party)
            response = st.write_stream(response_stream)
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response,
                "party": party,
            }
        )
