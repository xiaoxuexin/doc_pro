from streamlit_feedback import streamlit_feedback
# from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma
# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
import os
import tempfile
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter

new_key = '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'
model_name = 'gpt-4-turbo'

st.set_page_config(page_title="LangChain: Interact with Your Documents", page_icon="🦜")
st.title("🤖Document📃Processor🔧")
uploaded_files = st.sidebar.file_uploader(
    label="Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True
)


@st.cache_resource
def self_upload(files):
    # Read documents
    docs = []
    temp_dir = tempfile.TemporaryDirectory()
    for file in files:
        temp_filepath = os.path.join(temp_dir.name, file.name)
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())
        self_loader = PyPDFLoader(temp_filepath)
        docs.extend(self_loader.load())
    return docs


def configure_retriever(docs):
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)
    # Create embeddings and store in vectordb
    embedding = OpenAIEmbeddings(openai_api_key=new_key)
    # vectordb = DocArrayInMemorySearch.from_documents(splits, embeddings)
    persist_directory = 'docs/chroma/'
    # rm -rf ./docs/chroma
    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embedding,
        persist_directory=persist_directory
    )
    conf_retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 2, "fetch_k": 3})
    return conf_retriever


class PrintRetrievalHandler(BaseCallbackHandler):

    def __init__(self, container):
        self.status = container.status("**Context Retrieval**")

    def on_retriever_start(self, serialized: dict, query: str, **kwargs):
        self.status.write(f"**Question:** {query}")
        self.status.update(label=f"**Context Retrieval:** {query}")

    def on_retriever_end(self, documents, **kwargs):
        for idx, doc in enumerate(documents):
            source = os.path.basename(doc.metadata["source"])
            self.status.write(f"**Document {idx} from {source}**")
            self.status.markdown(doc.page_content)
        self.status.update(state="complete")


if "messages" not in st.session_state:
    st.session_state.messages = []

llm = ChatOpenAI(
    model_name=model_name,
    openai_api_key=new_key,
    temperature=0,
    streaming=True
)
msgs = StreamlitChatMessageHistory(key="langchain_messages")

memory = ConversationBufferMemory(memory_key="chat_history",
                                  chat_memory=msgs,
                                  return_messages=True,
                                  input_key='question',
                                  output_key='answer')

template = """Answer the question.
Use the provided context and chat history to answer the question.
If you don't know the answer, try to ask question for clarification.
Try to provide more info, and the answer should be no less than 10 sentences.
Keep the answer as concise as possible. Don't make up the material.
{context}
{chat_history}
{question}
Helpful Answer:"""

text = []

if uploaded_files:
    text = self_upload(uploaded_files)
    st.info('Files uploaded.')

if not uploaded_files:
    st.info("Please upload the files.")
    loaders = [
        # TextLoader("./files/sample_file.txt"),
        PyPDFLoader("./files/2022-report-economic-households-202305.pdf"),
    ]
    for loader in loaders:
        text.extend(loader.load())

retriever = configure_retriever(text)

prompt = PromptTemplate(
    input_variables=["context", "chat_history", "question"],
    template=template)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm,
    return_source_documents=True,
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={'prompt': prompt},
)

if user_query := st.chat_input(placeholder="Please input your question: 🙋"):
    # user input
    with st.chat_message("user"):
        # show the user input
        st.markdown(user_query)
        # append the user input to streamlit memory
        st.session_state.messages.append({"role": "user", "content": user_query})
        print('user input')
    # machine response
    with st.chat_message("assistant"):
        retrieval_handler = PrintRetrievalHandler(st.container())
        result = qa_chain({"question": user_query}, callbacks=[retrieval_handler])
        page = result['source_documents'][0].metadata['page']
        head, tail = os.path.split(result['source_documents'][0].metadata['source'])
        response = 'Based on 《' + str(tail).replace(".pdf", "》") + ' Page ' + \
                   str(page + 1) + ', we can get the ' \
                   'following info' + \
                   '\n\n' + result["answer"]
        # show the result in streamlit
        st.markdown(response)
    feedback = streamlit_feedback(
        feedback_type="thumbs",
        optional_text_label="[Optional] Please provide an explanation",
    )
