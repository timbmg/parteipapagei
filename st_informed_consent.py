import streamlit as st

from streamlit_cookies_controller import CookieController

cookie_controller = CookieController()

pseudo_user_id = cookie_controller.get("pseudo-user-id")
science_consent = cookie_controller.get("science-consent")

with open("sites/informed_consent.md") as md_file:
    st.markdown(md_file.read())

if science_consent:
    st.markdown("""
        ## Widerspruch
        Sie haben das Recht, ihre Teilnahme an der Studie zu widerrufen. 
        Wenn Sie von Ihrem Widerspruchsrecht Gebrauch machen möchten, genügt eine E-Mail an 
        [baumgaertner.t@gmail.com](mailto:baumgaertner.t@gmail.com) mit dem Betreff 
        "Widerspruch ChatBTW" unter Angabe folgender ID `{pseudo_user_id}`.
        """.format(pseudo_user_id=pseudo_user_id)
    )
else:
    pass
    # st.markdown("""
    #     ## Einwilligung
    #     Wenn Sie im nachinhein Ihre Einwilligung erteilen möchten, können Sie dies tun, indem Sie auf den folgenden Button klicken.
    #     """
    # )
    # if st.button("Einwilligung erteilen"):
    #     cookie_controller.set("science-consent", True)
    #     st.success("Vielen Dank für Ihre Einwilligung.")
