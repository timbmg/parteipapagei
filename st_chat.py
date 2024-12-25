import os

import chromadb
import streamlit as st
from llama_index.core import (
    VectorStoreIndex,
    Settings,
    PromptTemplate,
    get_response_synthesizer,
)
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.vector_stores.chroma import ChromaVectorStore

from party_data import party_data

st.title("🗳️ ChatBTW")

@st.cache_resource
def init_query_engines():

    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    gemini_llm = "models/gemini-1.5-flash-002"
    gemini_embedding = "models/text-embedding-004"
    persist_dir = "chroma"
    chroma_collection = "wahlprogramme"

    Settings.llm = Gemini(model=gemini_llm)
    Settings.embed_model = GeminiEmbedding(model=gemini_embedding, embed_batch_size=4)

    db = chromadb.PersistentClient(path=persist_dir)
    chroma_collection = db.get_or_create_collection(chroma_collection)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    index = VectorStoreIndex.from_vector_store(vector_store)

    engines = {}
    for party in ["cdu", "afd", "fdp", "grüne", "spd"]:
        filters = MetadataFilters(
            filters=[
                MetadataFilter(key="party", value=party),
            ]
        )

        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=5,
            filters=filters,
        )

        # configure response synthesizer
        response_synthesizer = get_response_synthesizer(streaming=True)

        # assemble query engine
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
            node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.3)],
        )
        text_qa_template = (
            "Du bist Politker der {party}. "
            "Ein Bürger möchte etwas über euer Wahlprogramm zur Bundestagswahl 2025 wissen. "
            "Relevante Informationen aus dem Wahlprogramm sind unten aufgeführt.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Beantworte die Frage ausschließlich mit den Informationen aus dem Wahlprogramm.\n"
            "Halten deine Antwort kurz und prägnant, um den Bürger zu überzeugen.\n"
            "Bürger Frage: {query_str}\n"
            "Deine Politker Antwort: "
        )
        text_qa_template = PromptTemplate(text_qa_template).partial_format(
            party=party_data[party]["template_name"]
        )
        query_engine.update_prompts(
            {"response_synthesizer:text_qa_template": text_qa_template}
        )
        print(f"Initialized query engine for party {party}.")
        engines[party] = query_engine
    return engines

ENGINES = init_query_engines()

def response_generator(user_query: str, party: str):
    print(user_query)
    response = ENGINES[party].query(user_query)
    for node in response.source_nodes:
        print(node.text)
    for message in response.response_gen:
        print(message)
        yield message


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
    f"Deine Frage an den Bundesbot",
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
