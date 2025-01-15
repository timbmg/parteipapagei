import streamlit as st

from streamlit_cookies_controller import CookieController

cookie_controller = CookieController()

pseudo_user_id = cookie_controller.get("pseudo-user-id")

with open("sites/data_protection.md") as md_file:
    st.markdown(md_file.read().format(pseudo_user_id=pseudo_user_id))
