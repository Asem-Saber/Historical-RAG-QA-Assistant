import json

import streamlit as st
import requests

API_URL = "http://localhost:8000/api"

st.set_page_config(page_title="Amoon Chatbot", page_icon="🤖")
st.title("Amoon Chatbot")
st.markdown("Ask me anything based on the ancient Egyptian civilization")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What would you like to know?"):

    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        try:
            response = requests.post(
                f"{API_URL}/chat/stream",
                json={"query": prompt},
                stream=True,
                timeout=120,
            )
            response.raise_for_status()

            answer_placeholder = st.empty()
            full_answer = ""
            sources = []

            for line in response.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue

                payload = json.loads(line[6:])

                if payload["type"] == "chunk":
                    full_answer += payload["content"]
                    answer_placeholder.markdown(full_answer + "▌")

                elif payload["type"] == "sources":
                    sources = payload.get("documents", [])

                elif payload["type"] == "error":
                    full_answer = f"Something went wrong: {payload.get('message', 'Unknown error')}"
                    answer_placeholder.error(full_answer)
                    break

                elif payload["type"] == "done":
                    break

            answer_placeholder.markdown(full_answer)
            answer = full_answer

            if sources:
                with st.expander("Sources & Citations"):
                    for doc in sources:
                        citation = doc.get("citation", "")
                        content = doc.get("content", "")
                        preview = content[:300] + "..." if len(content) > 300 else content
                        st.markdown(f"**[{citation}]** {preview}")
                        st.divider()

        except requests.exceptions.ConnectionError:
            answer = "Could not connect to the API. Make sure the FastAPI server is running."
            st.error(answer)
        except requests.exceptions.RequestException as e:
            answer = f"API error: {e}"
            st.error(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
