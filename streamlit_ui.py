import streamlit as st

from src.llm.chat import get_chat_generator, OpenAIModelSelection
from src.llm.embed import get_embedder, OpenAIEmbedderSelection
from src.etl.model import Document
from src.etl.qdrant import instantiate_qclient, search_documents
from src.util.util import PROJECT_ROOT, load_config



INPUT_PROMPT = """\
[INST] <<SYS>>
You are helpful assistant. \
<</SYS>>
The best matching reference is:
{reference_1}

Please provide an answer to the following question, based on the given references above:
{question}

[/INST]"""

st.title("Hallo, ich bin AInstein. Wie kann ich dir helfen?")

embedder = get_embedder(OpenAIEmbedderSelection.SMALL)

chat_generator = get_chat_generator(OpenAIModelSelection.GPT3)

# TASK 3.1: Add a text input widget
# Checkout https://docs.streamlit.io/library/api-reference/text and https://docs.streamlit.io/library/api-reference/widgets
user_input = st.text_input("write something")

if st.button("Generate"):
    if user_input:

        qdrant_config = config["qdrant"]
        qdrant = instantiate_qclient(qdrant_config["storage_path"])
        # TASK 3.2: Embed the user input
        embedded_question = embedder.embed_query(user_input)
        top_10_documents = search_documents(
            qdrant, qdrant_config["document_collection"], embedded_question
        )

        if not top_10_documents:
            st.warning("No results found.")
            st.stop()

        # TASK: Create a list of Documents based on the qdrant response
        # TASK 3.3: You will need to complete 'from_qdrant_scored_point' method to the Document class in model.py
        document_results = [
            (rank, Document.from_qdrant_scored_point(document))
            for rank, document in enumerate(top_10_documents, start=1)
        ]
        text_of_top_result = document_results[0][1].content
            
    st.write("LLM Response:")
    
    # TASK 3.4: Add reasonable prompts for the chatbot - in case you are unhappy with the provided prompt
    input_prompt = INPUT_PROMPT.format(
        reference_1=text_of_top_result,
        question=user_input)

    # TASK 3.5: Invoke the chat generator with the input prompt
    response = chat_generator.invoke(input_prompt)

    # TASK 3.6: Display the response
    st.write(response)
else:
    st.warning("Please enter some text.")
