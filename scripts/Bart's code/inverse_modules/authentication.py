# -*- coding: utf-8 -*-
"""
Authentication Module

Handles password protection and user access control for the PDP inverse application.
"""

import streamlit as st


def check_password() -> bool:
    """
    Returns `True` if the user had the correct password.
    
    Uses Streamlit session state to remember authentication status.
    Password is "pdp2025" (hardcoded for simplicity).
    
    Returns:
        bool: True if authenticated, False otherwise
    """

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        # Use .get() to safely access the password key, avoiding KeyError if not present
        entered_password = st.session_state.get("password", "")
        if entered_password == "pdp2025":
            st.session_state["password_correct"] = True
            # Safely delete password from state if it exists
            if "password" in st.session_state:
                del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True
