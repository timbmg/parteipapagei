from functools import partial

import streamlit as st

from party_data import party_data

st.set_page_config(
    page_title="🗳️ ChatBTW",
    page_icon="🗳️",
    layout="wide",
    initial_sidebar_state="collapsed"
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
        url_path=party
    )

about = st.Page(
    partial(st.markdown, open("about.md").read(), unsafe_allow_html=True),
    title="Über ChatBTW",
    icon="🧑🏼‍💻",
    url_path="about"
)
disclaimer = st.Page(
    partial(st.markdown, open("disclaimer.md").read(), unsafe_allow_html=True),
    title="Disclaimer",
    icon="⚠️",
    url_path="disclaimer"
)
pg = st.navigation(
    {
        "Chat": [st.Page("st_chat.py", title="ChatBTW", icon="🗳️", default=True)],
        "Wahlprogramme": [data["page"] for data in party_data.values()],
        "Über": [about, disclaimer]
    },
    expanded=False
)

pg.run()
