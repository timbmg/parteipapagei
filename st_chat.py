import logging
import os
import random
import re
import time
import uuid
from functools import cache
from typing import Any, Optional, cast

import chromadb
import llama_index.core.instrumentation as instrument
import Stemmer
import streamlit as st
import streamlit.components.v1 as components
import xxhash
from llama_index.core import (PromptTemplate, QueryBundle, Settings,
                              VectorStoreIndex, get_response_synthesizer)
from llama_index.core.bridge.pydantic import Field
from llama_index.core.callbacks import CallbackManager
from llama_index.core.callbacks.base_handler import BaseCallbackHandler
from llama_index.core.callbacks.schema import CBEventType, EventPayload
from llama_index.core.llms import ChatMessage
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import (QueryFusionRetriever,
                                         VectorIndexRetriever)
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.chroma import ChromaVectorStore
from streamlit_cookies_controller import CookieController
from supabase import Client, create_client

from party_data import party_data

logger = logging.getLogger(__name__)

POLICY = """
Bevor es losgeht, lies bitte die folgenden Nutzungsbedingungen.  

🔒 Bitte beachte unsere [Datenschutzbestimmungen](/data-protection) bevor Du fortfährst.

⚠️ Die Antworten von ParteiPapagei basieren auf dem Wahlprogramm der Parteien. 
Trotzdem kann ParteiPapagei Fehler machen und die tatsächliche Position einer Partei falsch 
wiedergeben. Alle von ParteiPapagei bereitgestellten Informationen sind unverbindlich und 
sollten unabhängig überprüft werden. Für Details siehe [Disclaimer](/disclaimer).  

🔬 Mit Deiner Zustimmung können die eingegebenen Fragen gespeichert werden um. Diese 
werden zur Verbesserung von ParteiPapagei verwendet und können wissenschaftlich 
ausgewertet und veröffentlicht werden. Falls es zu einer Veröffentlichung kommt, werden 
Deine Nachrichten auf mögliche personenbezogene Daten geprüft und anonymisiert oder von 
der Veröffentlichung ausgeschlossen. ParteiPapagei ist allerdings auch ohne diese 
Zustimmung nutzbar. Falls Du im Nachhinein diesen Bestimmungen widersprechen möchtest, 
nimm bitte Kontakt zu uns auf und gebe folgende ID an: `{pseudo_user_id}`. Bitte 
speichere diese ID jetzt. Sie kann ebenfalls in der 
[Freiwilligen Einwilligung](/informed-consent) aufgerufen werden, solange Du 
ParteiPapageis Cookies nicht löschst.
"""

GEMINI_LLM_MODEL = "models/gemini-1.5-flash-002"
GEMINI_EMBEDDING_MODEL = "models/text-embedding-004"

QUERY_GEN_PROMPT = """Du bist ein hilfreicher Assistent, der basierend auf einer Frage 
eines Bürgers zum Wahlprogrammen einer Partei ähnliche Fragen generiert. Überlege dir 
unterschiedliche Formulierungen der selben Frage und weitere verwandte Fragen um relevante 
Informationen aus dem Wahlprogramm zu finden, die für die beantwortung der 
ursprünglichen Frage nützlich sind. Falls die Frage sehr allgemein ist, formuliere 
Fragen die die wichtigsten Informationen aus dem Wahlprogramm aller Parteien abdecken.\n 
Generiere insgesamt {num_queries} Fragen, jede in einer eigenen Zeile.\n
Frage: {query}\n
Generierte Fragen:\n
"""
QA_PROMPT = """Du bist Politker der {party}. 
Ein Bürger möchte etwas über euer Wahlprogramm zur Bundestagswahl 2025 wissen. 
Relevante Passagen aus dem Wahlprogramm sind unten aufgeführt.\n
---------------------\n
{context_str}\n
---------------------\n
Beantworte die Frage basierend auf den Informationen im Wahlprogramm. 
Überlege, wie die Informationen im Wahlprogramm auf die Frage des Bürgers anwendbar sind.\n
Falls die Inhalte des Wahlprogramms nicht relevant sind, 
antworte dem Bürger, dass die {party} keine Position zu diesem Thema im Wahlprogramm hat.\n
Falls die Frage unangemessen, beleidigend oder diskriminierend ist, weise den Bürger 
darauf hin und bitte ihn um eine angemessene Frage.\n
Halten deine Antwort kurz und prägnant, um den Bürger von den Inhalten der {party} zu überzeugen.\n
{settings_str}
Füge jeder deiner Aussagen eine Referenz hinzu, auf welche Passage aus dem Wahlprogramm diese zurückzuführen ist. 
Verwende dazu folgendes Format: '<Antwort> [X, Y].' um zu belegen, dass deine 
Antwort, sich auf Passagen X und Y zurückzuführen ist.\n
Formatiere deine Antwort in Markdown, z.B. nutze Bulletpoints mit fettgedruckten Stichpunkten um einen neuen Punkt zu beginnen.\n
Bürgerfrage: {query_str}\n
Deine Politiker-Antwort 
"""
SIMPLE_LANGUAGE_PROMPT = """
Benutze eine klare, einfache und verständliche Sprache mit kurzen Sätzen und ohne 
Fachbegriffe. Vermeide oder erkläre Begriffe, die einem einfachen Bürger unbekannt sein 
könnten.
"""
SHORT_ANSWER_PROMPT = """
Konzentriere dich ausschließlich auf die wichtigsten Punkte. Deine Antwort darf nicht 
länger als 1-2 Sätze sein stellt nur den wichtigsten Punkt der Partei klar und prägnant 
dar. 
"""

def get_secret_or_env_var(key: str, default: Optional[str] = None) -> str:
    if not os.path.exists("./streamlit/secrets.toml"):
        return os.getenv(key, default)
    return st.secrets.get(key, default) or os.getenv(key, default)

# will be saved in the database to distinguish between test and production data
ENVIRONMENT = get_secret_or_env_var("ENVIRONMENT")

# make sure GOOGLE_API_KEY is set in env variable
os.environ["GOOGLE_API_KEY"] = get_secret_or_env_var("GOOGLE_API_KEY")


cookie_controller = CookieController()
# we need to wait for the cookie to be set, otherwise the cookie dialog will be shown again
time.sleep(0.2)


class RetrieverQueryEngineWithPromptSelection(RetrieverQueryEngine):

    def __init__(self, prompts: dict, *args, **kwargs):
        self.prompts = prompts
        super().__init__(*args, **kwargs)

    def query(self, query, prompt):
        selected_prompt = self.prompts[prompt]
        self.update_prompts({"response_synthesizer:text_qa_template": selected_prompt})

        return super().query(query)


@st.cache_resource
def init_supabase_connection() -> Client:
    url = get_secret_or_env_var("SUPABASE_URL")
    key = get_secret_or_env_var("SUPABASE_KEY")
    client = create_client(url, key)
    return client


@st.cache_resource(show_spinner="Lese Wahlprogramme...")
def init_query_engines():

    CHROMA_PERSIST_DIR = "chroma-256"
    CHROMA_COLLECTION = "wahlprogramme"

    Settings.llm = Gemini(model=GEMINI_LLM_MODEL)
    Settings.embed_model = GeminiEmbedding(model=GEMINI_EMBEDDING_MODEL)
    db = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    chroma_collection = db.get_or_create_collection(CHROMA_COLLECTION)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    index = VectorStoreIndex.from_vector_store(vector_store)
    prompt_callback_handler = PromptCallbackHandler()
    callback_manager = CallbackManager([prompt_callback_handler])

    engines = {}
    for party in party_data.keys():
        filters = MetadataFilters(
            filters=[
                MetadataFilter(key="party", value=party),
            ]
        )

        dense_retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=20,
            filters=filters,
        )
        bm25_retriever = BM25Retriever.from_persist_dir(f"bm25-256/{party}")
        bm25_retriever.similarity_top_k = 20
        bm25_retriever.stemmer = Stemmer.Stemmer("german")

        fusion_retriever = CachedQueryFusionRetriever(
            [
                dense_retriever,
                bm25_retriever,
            ],
            mode="reciprocal_rerank",
            num_queries=3,
            use_async=False,
            similarity_top_k=10,
            verbose=True,
            query_gen_prompt=QUERY_GEN_PROMPT,
            retriever_weights=[0.5, 0.5],
        )

        # configure response synthesizer
        response_synthesizer = get_response_synthesizer(
            streaming=True, callback_manager=callback_manager
        )

        def format_context_fn(**kwargs):
            # format context with reference numbers
            context_list = re.split(r"^#\s", kwargs["context_str"], flags=re.MULTILINE)
            context_list = [x for x in context_list if x]
            fmtted_context = "".join(
                [f"[{i}] # {c}" for i, c in enumerate(context_list, start=1)]
            )
            return fmtted_context

        base_template = PromptTemplate(
            QA_PROMPT, function_mappings={"context_str": format_context_fn}
        ).partial_format(party=party_data[party]["template_name"], settings_str="")

        short_template = PromptTemplate(
            QA_PROMPT, function_mappings={"context_str": format_context_fn}
        ).partial_format(
            party=party_data[party]["template_name"],
            settings_str=SHORT_ANSWER_PROMPT + "\n",
        )

        simple_template = PromptTemplate(
            QA_PROMPT, function_mappings={"context_str": format_context_fn}
        ).partial_format(
            party=party_data[party]["template_name"],
            settings_str=SIMPLE_LANGUAGE_PROMPT + "\n",
        )

        short_simple_template = PromptTemplate(
            QA_PROMPT, function_mappings={"context_str": format_context_fn}
        ).partial_format(
            party=party_data[party]["template_name"],
            settings_str=SIMPLE_LANGUAGE_PROMPT + "\n" + SHORT_ANSWER_PROMPT + "\n",
        )

        query_engine = RetrieverQueryEngineWithPromptSelection(
            prompts={
                "base": base_template,
                "short": short_template,
                "simple": simple_template,
                "short_simple": short_simple_template,
            },
            retriever=fusion_retriever,
            response_synthesizer=response_synthesizer,
            node_postprocessors=[RankCutoffPostprocessor(rank_cutoff=10)],
        )

        engines[party] = query_engine

    return engines


class ProfanityChekcer:
    def __init__(self):
        with open("profanity_words.txt") as f:
            self.wordlist = f.read().lower().splitlines()

    def __call__(self, text) -> bool:
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)
        for word in text.split():
            if word in self.wordlist:
                return True
        return False


class PromptCallbackHandler(BaseCallbackHandler):
    def __init__(
        self,
        event_starts_to_ignore: Optional[list[CBEventType]] = None,
        event_ends_to_ignore: Optional[list[CBEventType]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.logger: Optional[logging.Logger] = logger
        super().__init__(
            event_starts_to_ignore=event_starts_to_ignore or [],
            event_ends_to_ignore=event_ends_to_ignore or [],
        )
        self.last_prompt = None

    def start_trace(self, trace_id: Optional[str] = None) -> None:
        return

    def end_trace(
        self,
        trace_id: Optional[str] = None,
        trace_map: Optional[dict[str, list[str]]] = None,
    ) -> None:
        return

    def on_event_start(
        self,
        event_type: CBEventType,
        payload: Optional[dict[str, Any]] = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> str:

        if (
            event_type == CBEventType.LLM
            and payload is not None
            and EventPayload.MESSAGES in payload
        ):
            messages = cast(list[ChatMessage], payload.get(EventPayload.MESSAGES, []))
            messages_str = "\n".join([str(x) for x in messages])
            self.last_prompt = messages_str

        return event_id

    def on_event_end(self, event_type, payload=None, event_id="", **kwargs):
        return


class RankCutoffPostprocessor(BaseNodePostprocessor):
    rank_cutoff: int = Field(default=10)

    def _postprocess_nodes(
        self, nodes: list[NodeWithScore], query_bundle: Optional[QueryBundle]
    ) -> list[NodeWithScore]:
        sorted_nodes = sorted(nodes, key=lambda x: x.score, reverse=True)
        return sorted_nodes[: self.rank_cutoff]


class CachedQueryFusionRetriever(QueryFusionRetriever):

    @cache
    def _get_queries(self, original_query: str):
        return super()._get_queries(original_query)


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
        new_anchor = xxhash.xxh32(text, seed=0xABCD).hexdigest()[:16]
    return new_anchor


def response_generator(response, party: str):

    message_iter = iter(response.response_gen)
    while True:
        try:
            stream = next(message_iter)
        except StopIteration:
            break
        fragments = list(
            set(re.findall(r"\[\d+(?:,\s*\d+)*\]?|\[?\d+(?:,\s*\d+)*\]", stream))
        )
        for fragment in fragments:
            try:
                modified_fragment = ""
                last_char_was_digit = False
                parsed_number = ""
                for c in fragment:
                    if c.isdigit():
                        parsed_number += c
                        last_char_was_digit = True
                    else:
                        if last_char_was_digit:
                            _id = create_anchor_from_text(
                                response.source_nodes[
                                    int(parsed_number) - 1
                                ].node.metadata["header"]
                            )
                            modified_fragment += f"[{parsed_number}]({party}#{_id})"
                        modified_fragment += c
                        last_char_was_digit = False
                        parsed_number = ""
                if last_char_was_digit:
                    _id = create_anchor_from_text(
                        response.source_nodes[int(parsed_number) - 1].node.metadata[
                            "header"
                        ]
                    )
                    modified_fragment += f"[{parsed_number}]({party}#{_id})"
            except Exception as e:
                # if anything goes wrong, just keep the fragment as is
                logger.error(f"Error processing fragment\n{fragment=}\n{stream=}")
                modified_fragment = fragment
            stream = stream.replace(fragment, modified_fragment)

        yield stream


def save_consents(cookies: bool, data_protection: bool, science: bool):
    try:
        response = (
            supabase.table("consents")
            .insert(
                {
                    "cookies": cookies,
                    "data_protection": data_protection,
                    "science": science,
                    "environment": ENVIRONMENT,
                    "user_id": sb_user_id,
                    "pseudo_user_id": cookie_controller.get("pseudo-user-id"),
                }
            )
            .execute()
        )
    except Exception as e:
        logger.error(
            f"Error saving consents; {cookies=}, {data_protection=}, {science=}"
        )
        print(e)


def save_query(user_query: str, parties: list[str]) -> int:
    try:
        response = (
            supabase.table("queries")
            .insert(
                {
                    "query": user_query,
                    "parties": parties,
                    "environment": ENVIRONMENT,
                    "user_id": sb_user_id,
                    "pseudo_user_id": cookie_controller.get("pseudo-user-id"),
                }
            )
            .execute()
        )
        return response.data[0]["id"]
    except Exception as e:
        logger.error(f"Error saving query; {user_query=}, {parties=}")
        print(e)


def save_response(
    query_id: int,
    response: str,
    party: str,
    prompt: str,
):
    try:
        response = (
            supabase.table("responses")
            .insert(
                {
                    "response": response,
                    "party": party,
                    "prompt": prompt,
                    "environment": ENVIRONMENT,
                    "query_id": query_id,
                    "user_id": sb_user_id,
                }
            )
            .execute()
        )
    except Exception as e:
        logger.error(f"Error saving response; {response=}, {party=}, {prompt=}")
        print(e)
        return None
    return response.data[0]["id"]


def sample_question_click(question):
    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.sample_query = question


def new_chat_click():
    st.session_state.messages = []


def add_user_query_to_session():
    st.session_state.messages.append(
        {"role": "user", "content": st.session_state.user_query}
    )


@st.dialog("🤬 Profanity Alert", width="small")
def profanity_dilaog():
    st.warning(
        "ParteiPapagei hat möglicherweise unangemessene Sprache in deiner Anfrage erkannt. "
        "Auch wenn Du mit einer KI schreibst, bleibe bitte respektvoll."
    )
    st.session_state.messages = st.session_state.messages[:-1]
    if st.button("Nochmal netter formulieren"):
        new_chat_click()
        st.rerun()


def pretty_print_messages():
    for message in st.session_state.get("messages", []):
        print(message["role"], "::", message["content"][:20])


if cookie_controller.get("pseudo-user-id") is None:
    cookie_controller.set("pseudo-user-id", str(uuid.uuid4()))

# supabase, sb_user_id, sb_access_token, sb_refresh_token = init_supabase_connection()
supabase = init_supabase_connection()
session = supabase.auth.get_session()
if session is None:
    user_data = supabase.auth.sign_in_with_password(
        {
            "email": get_secret_or_env_var("SUPABASE_EMAIL"),
            "password": get_secret_or_env_var("SUPABASE_PASSWORD"),
        }
    )
    sb_user_id = user_data.user.id
    supabase.auth.set_session(
        access_token=user_data.session.access_token,
        refresh_token=user_data.session.refresh_token,
    )
else:
    sb_user_id = session.user.id

engines = init_query_engines()

pc = ProfanityChekcer()

if "messages" not in st.session_state:
    st.session_state.messages = []


@st.dialog("🤝 Nutzungsbedingungen von ParteiPapagei", width="large")
def accept_policy():
    st.info(
        "👋 Willkommen bei ParteiPapagei! ParteiPapagei ist eine KI mit der Du die Inhalte der Wahlprogramme der Parteien zur Bundestagswahl 2025 entdecken, vergleichen und verstehen kannst."
    )
    st.markdown(POLICY.format(pseudo_user_id=cookie_controller.get("pseudo-user-id")))
    # consent_cols = st.columns(3)
    cookie_policy = st.checkbox(
        "Ich stimme der Verwendung von Cookies zu.", key="cookie_policy", value=True
    )
    data_protection_policy = st.checkbox(
        "Ich akzeptiere die [Datenschutzbestimmungen](/data-protection).",
        key="data_protection_policy",
        value=True,
    )
    science_policy = st.checkbox(
        "Ich habe die [Freiwillige Einwilligung](/informed-consent) gelesen und stimme der Speicherung, Verarbeitung und **anonymen** Veröffentlichung meiner Nachrichten zu.",
        key="science_policy",
        value=True,
    )
    if st.button(
        "Zustimmen",
        type="primary",
        disabled=not (cookie_policy and data_protection_policy),
    ):
        cookie_controller.set("policy-accepted", True)
        cookie_controller.set("science-consent", science_policy)
        save_consents(cookie_policy, data_protection_policy, science_policy)
        st.rerun()
    else:
        st.stop()


if not cookie_controller.get("policy-accepted"):
    accept_policy()

st.title("🗳️ ParteiPapagei", anchor=False)
header = st.container(key="container-header")
st.markdown(
    """<style>
    h1 {
        padding-top:1rem !important
    }
    footer {
        visibility: hidden !important;
    }
    #MainMenu {
        visibility: hidden !important;
    }
    header {
        visibility: hidden !important;
    }
    .block-container {
        padding-top: 0rem;
    }
    div[data-testid="stVerticalBlock"] div:has(div.fixed-header) {
        position: sticky;
        top: 0.001rem;
        background-color: white;
        z-index: 999;
    }
    div[data-testid="stColumn"] {
        width: fit-content !important;
        flex: unset;
    }
    div[data-testid="stColumn"] * {
        margin-top: 0 !important;
    }
    div.st-key-btn-party-selection-help button {
        border: none !important;
        margin-left: -0.975rem;
    }
    div.st-key-btn-party-selection-help button:hover {
        cursor: help !important;
    }
    button[data-testid="stBaseButton-segmented_control"],
    button[data-testid="stBaseButton-segmented_controlActive"] {
        line-height: 1.6;
        min-height: 2.5rem;
    }
    div.st-key-simple-language {
        line-height: 1.6;
        min-height: 2.5rem;
        display: flex;
    }
    div.st-key-simple-language label {
        margin-bottom: 0 !important;
        display: flex;
        place-items: center start !important;
    }
    div.st-key-short-answer {
        line-height: 1.6;
        min-height: 2.5rem;
        display: flex;
    }
    div.st-key-short-answer label {
        margin-bottom: 0 !important;
        display: flex;
        place-items: center start !important;
    }
    div.stCheckbox {
        margin-bottom: 0 !important;
    }
    div.stMarkdown:has(#party-selection-help) {
        line-height: 1.6;
        min-height: 2.5rem;
        display: flex;
    }
    div.stMarkdown label {
        margin-bottom: 0 !important;
        display: flex;
        place-items: center start !important;
    }
    @media (max-width: 768px) { /* Adjust 768px to your desired breakpoint */
        div[data-testid="stVerticalBlock"] div:has(div.fixed-header) {
            position: static; /* Disable sticky behavior */
        }
        
        div.stColumn:has(#party-selection-help):not(:has(div.stColumn)) {
            display: none; /* Hide the column with the party selection help */
        }
    }
</style>""",
    unsafe_allow_html=True,
)
header.write("<div class='fixed-header'/>", unsafe_allow_html=True)
control_cols = header.columns(5, gap="small", vertical_alignment="bottom", border=False)
control_cols[0].button(
    "💬 Neuer Chat",
    on_click=new_chat_click,
    disabled=len(st.session_state.get("messages", [])) == 0,
    type="secondary",
)
control_cols[1].segmented_control(
    label=None,
    options=[party for party, data in party_data.items() if data["enabled"]],
    format_func=lambda p: f"{party_data[p]['emoji']} {party_data[p]['name']}",
    selection_mode="multi",
    default=st.session_state.get(
        "party_selection",
        random.sample(
            [party for party in party_data.keys() if party_data[party]["enabled"]], 2
        ),
    ),
    key="party_selection",
    help="Wähle die Parteien mit denen Du dich über ihr Programm unterhalten willst.",
    label_visibility="hidden",
)
control_cols[2].markdown(
    "<div id='party-selection-help'/>",
    help="Wähle die Parteien mit denen Du dich über ihr Programm unterhalten willst.",
    unsafe_allow_html=True,
)
control_cols[3].toggle(
    "Einfache Sprache",
    key="simple-language",
    value=False,
    help="Aktiviere diese Option, um die Antworten in einfacher Sprache zu erhalten.",
)
control_cols[4].toggle(
    "Kurze Antwort",
    key="short-answer",
    value=False,
    help="Aktiviere diese Option, um kurze Antworten zu erhalten.",
)
if len(st.session_state.messages) == 0:
    st.info(
        "_Stelle eine eigene Frage oder wähle aus den Beispielen. ParteiPapagei wird eine Antwort für die Parteien basierend auf deren Wahlprogrammen generieren._",
        icon=":material/info:",
    )
    st.session_state.sample_query = None
    sample_questions = [
        "Was ist Ihr Plan, um Deutschlands Wirtschaft wieder wachsen zu lassen?",
        "Was muss sich Ihrer Meinung nach in der Zuwanderungs- und Asylpolitik ändern?",
        "Wie können Klimaschutz und Wirtschaftswachstum vereint werden?",
        "Wie kann die Digitalisierung in Deutschland vorangetrieben werden?",
    ]
    sample_question_cols = st.columns(len(sample_questions), gap="small")
    for col, question in zip(sample_question_cols, sample_questions):
        col.button(
            question,
            type="secondary",
            on_click=sample_question_click,
            kwargs={"question": question},
        )
else:
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        if message["role"] == "assistant":
            chat_message_kwargs = {"avatar": party_data[message["party"]]["emoji"]}
        else:
            chat_message_kwargs = {}
        with st.chat_message(message["role"], **chat_message_kwargs):
            st.markdown(message["content"])

user_query = st.chat_input(
    f"Deine Frage an ParteiPapagei",
    on_submit=add_user_query_to_session,
    key="user_query",
)

simple_language = st.session_state.get("simple-language", False)
short_answer = st.session_state.get("short-answer", False)

if simple_language and short_answer:
    prompt_selection = "short_simple"
elif simple_language:
    prompt_selection = "simple"
elif short_answer:
    prompt_selection = "short"
else:
    prompt_selection = "base"

if user_query or st.session_state.get("sample_query", None):
    query_type = None
    if st.session_state.get("sample_query", None):
        query_type = "sample"
        user_query = st.session_state.pop("sample_query")
    else:
        query_type = "user"
        if pc(user_query):
            profanity_dilaog()
            st.stop()

        if cookie_controller.get("science-consent"):
            query_id = save_query(user_query, st.session_state.party_selection)
    logger.info(f"Query: {user_query}")

    for party in st.session_state.party_selection:
        with st.chat_message("assistant", avatar=party_data[party]["emoji"]):
            engine_response = engines[party].query(user_query, prompt=prompt_selection)
            response_stream = response_generator(response=engine_response, party=party)
            response = st.write_stream(response_stream)
            st.empty()
            logger.info(f"Response by {party}: {response}")
        if query_type == "user" and cookie_controller.get("science-consent"):
            prompt = engines[party].callback_manager.handlers[0].last_prompt
            _ = save_response(
                query_id=query_id, response=response, party=party, prompt=prompt
            )
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response,
                "party": party,
            }
        )

components.html("""
    <script data-collect-dnt="true" async src="https://scripts.simpleanalyticscdn.com/latest.js"></script>
    <noscript><img src="https://queue.simpleanalyticscdn.com/noscript.gif?collect-dnt=true" alt="" referrerpolicy="no-referrer-when-downgrade"/></noscript>
    """
)
