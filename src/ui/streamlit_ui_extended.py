import streamlit as st

from langchain.text_splitter import RecursiveCharacterTextSplitter

from llm.chat import get_chat_generator
from llm.embed import get_embedder
from etl.model import Document
from etl.qdrant import instantiate_qclient, search_documents
from util.util import PROJECT_ROOT, load_config
import etl.ingest as ingest

from langchain.schema.messages import HumanMessage, SystemMessage
from langchain.prompts import PromptTemplate

st.set_page_config(layout="wide")
config = load_config(PROJECT_ROOT / "config" / "config.yml")

st.title("My chatbot 'built in a day'")

embedder = config["embedder"]
embedder = get_embedder(embedder)

chat_generator = config["chat_generator"]
chat_generator = get_chat_generator(chat_generator)

answer_generated = False

prompt_instructions = """\
    You're a helpful assistant.\
    When answering a question, be mildly rude and complain about how much work you have to do, but then provide an adequate answer.'\
    """

# sidebar buttons etc.
with st.sidebar:
    st.subheader("Display options")
    st.toggle("Show prompt template", False, key="show_prompt_template")
    st.toggle("Show chunks", False, key="show_chunks")
    st.write("")
    st.write("")
    st.subheader("Chunking strategy")
    st.slider("Use top k chunks", 1, 10, 1, key="k")
    with st.expander("Ingestion", expanded=True):
        st.number_input("Chunk size", 100, 3000, 200, step=300, key="chunk_size", format="%d")
        st.number_input("Chunk overlap", 0, 500, 0, step=50, key="chunk_overlap", format="%d")
        if st.button("Run ingestion", key="ingestion"):
            # text splitter that is used. can theoretically be changed to other text splitters
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=st.session_state["chunk_size"], chunk_overlap=st.session_state["chunk_overlap"], length_function=len,
                    add_start_index=True, is_separator_regex = False,)
            ingest.main(
                text_splitter=text_splitter)
            st.success("Ingestion complete")

prompt_template_string = """Task: Answer the question based on the context.
Kontext: {context}
Question: {user_input}
Answer:"""

if st.session_state.show_prompt_template:
    st.subheader("Prompt template")
    prompt_template_string = st.text_area(label="This is the prompt template used when calling the LLM", value=prompt_template_string,
                                          label_visibility="hidden", height=150)

prompt_template = PromptTemplate(input_variables=["user_input", "context"], template=prompt_template_string)

st.subheader("Question")
user_input = st.text_area(
    "What do you want to know?", ""
)

if st.button("Generate"):
    if st.session_state.show_chunks:
        qa_field, chunks = st.columns(2)
    else:
        qa_field = st.container()
    with qa_field:
        if user_input:

            qdrant_config = config["qdrant"]
            qdrant = instantiate_qclient(qdrant_config["storage_path"])
            embedded_question = embedder.embed_query(user_input)
            top_10_documents = search_documents(qdrant, qdrant_config["document_collection"], embedded_question)

            if not top_10_documents:
                st.warning("No results found.")
                st.stop()

            st.session_state["top_10_chunks"] = [(rank, Document.from_qdrant_scored_point(document), document.payload["source"]) for rank, document in
                enumerate(top_10_documents, start=1)]

        st.subheader("LLM Response:")

        # build context blob
        st.session_state["context"] = "...\n\n...".join([chunk.content for _, chunk, _ in st.session_state["top_10_chunks"][0:st.session_state["k"]]])

        # create message history to call LLM with
        messages = [SystemMessage(content=prompt_instructions), HumanMessage(
            content=prompt_template.format(user_input=user_input,
                                           context=st.session_state["context"])), ]

        st.session_state["response"] = chat_generator.invoke(messages)

        st.write(st.session_state["response"])

        # display chunks
        if st.session_state.show_chunks:
            with chunks:
                if st.session_state.show_chunks:
                    st.subheader("Chunks")
                    if st.session_state["context"]:
                        for rank, chunk, source in st.session_state["top_10_chunks"][0:st.session_state["k"]]:
                            st.write(f"**{rank}. ({source})**\n\n{chunk.content}")  # nicer to look at
                        # st.write(st.session_state["context"])  # the actual text that is pasted into prompt template
else:
    st.warning("Please enter some text.")
