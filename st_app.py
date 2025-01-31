import base64
import os
import re
from functools import partial
from pathlib import Path

import streamlit as st

from party_data import party_data


st.set_page_config(
    page_title="ParteiPapagei",
    page_icon="🗳️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def party_program_page(party):
    st.title(f"{party_data[party]['emoji']} {party_data[party]['name']}")
    # insert link to original program
    st.markdown(f"[PDF]({party_data[party]['url_to_program']})")
    with open(f"data/clean/{party}.md") as fh:
        content = fh.read()
    content.replace("\n", "\n\n")
    st.markdown(content)


for party, data in party_data.items():
    party_data[party]["page"] = st.Page(
        partial(party_program_page, party=party),
        title=f"{data['name']}",
        icon=data["emoji"],
        url_path=party,
    )

about = st.Page(
    partial(
        st.markdown,
        open("sites/about.md").read(),
        unsafe_allow_html=True,
    ),
    title="Über ParteiPapagei",
    icon="🧑🏼‍💻",
    url_path="about",
)
disclaimer = st.Page(
    partial(st.markdown, open("sites/disclaimer.md").read(), unsafe_allow_html=True),
    title="Disclaimer",
    icon="⚠️",
    url_path="disclaimer",
)
data_protection = st.Page(
    "st_data_protection.py", title="Datenschutz", icon="🔒", url_path="data-protection"
)
informed_consent = st.Page(
    "st_informed_consent.py",
    title="Freiwillige Einwilligung",
    icon="🔬",
    url_path="informed-consent",
)
impressum = st.Page(
     partial(st.markdown, open("sites/impressum.md").read(), unsafe_allow_html=True),
    title="Impressum",
    icon="🧑‍⚖️",
    url_path="impressum",
)

pg = st.navigation(
    {
        "Chat": [st.Page("st_chat.py", title="ParteiPapagei", icon="🗳️", default=True)],
        "Wahlprogramme": [data["page"] for data in party_data.values()],
        "Über": [about, disclaimer, data_protection, informed_consent, impressum],
    },
    expanded=False,
)

pg.run()
