"""Streamlit UI package: business wiring and pure rendering helpers.

Importing any module in this package must never perform a network or LLM
API call. Real service objects (retriever, LLM client, agent) are only
constructed when explicitly requested at runtime (e.g. on a button click).
"""
