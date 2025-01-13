import streamlit as st

from streamlit_cookies_controller import CookieController

cookie_controller = CookieController()

pseudo_user_id = cookie_controller.get("pseudo-user-id")

st.title("🔒 Datenschutz")

st.markdown(f"""Deine Pseudo-User-ID: `{pseudo_user_id}`""")
