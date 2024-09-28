__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

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
model_name = 'gpt-4o'

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
    conf_retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 3, "fetch_k": 5})
    return conf_retriever


class StreamHandler(BaseCallbackHandler):
    print('inside stream handler')

    def __init__(self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""):
        self.container = container
        self.text = initial_text
        self.run_id_ignore_token = None

    def on_llm_start(self, serialized: dict, prompts: list, **kwargs):
        # Workaround to prevent showing the rephrased question as output
        if prompts[0].startswith("Human"):
            self.run_id_ignore_token = kwargs.get("run_id")

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if self.run_id_ignore_token == kwargs.get("run_id", False):
            return
        self.text += token
        self.container.markdown(self.text)


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


text = []
if uploaded_files:
    text = self_upload(uploaded_files)
    st.info('Files uploaded.')

if not uploaded_files:
    st.info("Please upload the files.")
    loaders = [
        # TextLoader("./files/sample_file.txt"),
        # PyPDFLoader("./files/Data Analysis Handbook.pdf"),
    ]
    for loader in loaders:
        text.extend(loader.load())

if "messages" not in st.session_state:
    st.session_state.messages = []

llm = ChatOpenAI(
    model_name=model_name,
    openai_api_key=new_key,
    temperature=0,
    streaming=True
)
msgs = StreamlitChatMessageHistory(key="langchain_messages")

if len(msgs.messages) == 0 or st.sidebar.button("Clear message history"):
    msgs.clear()
    msgs.add_ai_message("How can I help you today?")

avatars = {"human": "user", "ai": "assistant"}
for msg in msgs.messages:
    st.chat_message(avatars[msg.type]).write(msg.content)

memory = ConversationBufferMemory(memory_key="chat_history",
                                  chat_memory=msgs,
                                  return_messages=True,
                                  input_key='question',
                                  output_key='answer')

template = """Answer the question with the best of your knowledge.
Use the provided context and chat history to answer the question.
If you don't know the answer, try to ask question for clarification.
Try to provide more info, and the answer should be no less than 10 sentences.
Keep the answer as concise as possible. Don't make up the material.
{context}
{chat_history}
{question}
Helpful Answer:"""


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
        stream_handler = StreamHandler(st.empty())
        result = qa_chain({"question": user_query}, callbacks=[retrieval_handler, stream_handler])
        response = 'Based on '
        for doc in result['source_documents']:
            page = doc.metadata['page']
            head, tail = os.path.split(doc.metadata['source'])
            response = response + \
                       '《' + str(tail).replace(".pdf", "》") + ' Page ' + \
                       str(page + 1)+' content "'+doc.page_content+'"'+'\n\n'
        response = response + 'We can get the following info' + \
                       '\n\n' + result["answer"]
        # show the result in streamlit
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.markdown(response)
    feedback = streamlit_feedback(
        feedback_type="thumbs",
        optional_text_label="[Optional] Please provide an explanation",
        align="flex-start"
    )
