import base64
import os
import re
from functools import partial
from pathlib import Path

import streamlit as st

from party_data import party_data


def markdown_images(markdown):
    # example image markdown:
    # ![Test image](images/test.png "Alternate text")
    images = re.findall(
        r'(!\[(?P<image_title>[^\]]+)\]\((?P<image_path>[^\)"\s]+)\s*([^\)]*)\))',
        markdown,
    )
    return images


def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded


def img_to_html(img_path, img_alt):
    img_format = img_path.split(".")[-1]
    img_html = f'<img src="data:image/{img_format.lower()};base64,{img_to_bytes(img_path)}" alt="{img_alt}" style="max-width: 100%;">'

    return img_html


def markdown_insert_images(markdown):
    images = markdown_images(markdown)

    for image in images:
        image_markdown = image[0]
        image_alt = image[1]
        image_path = image[2]
        if os.path.exists(image_path):
            markdown = markdown.replace(
                image_markdown, img_to_html(image_path, image_alt)
            )
    return markdown


st.set_page_config(
    page_title="🗳️ ChatBTW",
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
        markdown_insert_images(open("sites/about.md").read()),
        unsafe_allow_html=True,
    ),
    title="Über ChatBTW",
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
pg = st.navigation(
    {
        "Chat": [st.Page("st_chat.py", title="ChatBTW", icon="🗳️", default=True)],
        "Wahlprogramme": [data["page"] for data in party_data.values()],
        "Über": [about, disclaimer, data_protection, informed_consent],
    },
    expanded=False,
)

pg.run()
