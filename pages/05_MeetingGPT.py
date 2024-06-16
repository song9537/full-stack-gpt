import streamlit as st
import subprocess
import math
from pydub import AudioSegment
import openai
import glob
import os
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import StrOutputParser
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.storage import LocalFileStore
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models import ChatOpenAI


class ChatCallbackHandler(BaseCallbackHandler):

    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


llm = ChatOpenAI(
    temperature=0.1,
)

llm2 = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    callbacks=[
        ChatCallbackHandler(),
    ],
)

has_transcript = os.path.exists("./.video.txt")

splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=800,
    chunk_overlap=100,
)


@st.cache_data(show_spinner="Embedding file....")
def embed_file(file_path):
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=800,
        chunk_overlap=100,
    )
    loader = TextLoader(file_path, autodetect_encoding=True)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)

    retriever = vectorstore.as_retriever()
    return retriever


if "messages" not in st.session_state:
    st.session_state["messages"] = []


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


@st.cache_data()
def extract_audio_from_video(video_path):
    audio_path = video_path.replace("mp4", "mp3")
    command = ["wsl", "ffmpeg", "-i", video_path, "-vn", audio_path]
    subprocess.run(command)


@st.cache_data()
def cut_audio_in_chunks(audio_path, chunk_size, chunks_folder):
    track = AudioSegment.from_mp3(audio_path)
    chunk_len = chunk_size * 60 * 1000

    chunks = math.ceil(len(track) / chunk_len)

    for i in range(chunks):
        start_time = i * chunk_len
        end_time = (i + 1) * chunk_len

        chunk = track[start_time:end_time]

        chunk.export(f"{chunks_folder}/chunk_{i}.mp3", format="mp3")


@st.cache_data()
def transcribe_chunks(chunk_folder, destination):
    files = glob.glob(f"{chunk_folder}/*.mp3")
    files.sort()
    for file in files:
        with open(file, "rb") as audio_file, open(
            destination, "a", encoding="utf-8"
        ) as text_file:
            transcript = openai.Audio.transcribe("whisper-1", audio_file)
            text_file.write(transcript["text"])


st.set_page_config(page_title="MeetingGPT", page_icon="💼")
st.markdown(
    """
    # MeetingGPT
            
    Welcome to MeetingGPT, upload a video and I will give you a transcript, a summary and a chat bot to ask any questions about it.
    
    Get started by uploading a vide file in the sidebar.
"""
)

with st.sidebar:
    video = st.file_uploader(
        "Video",
        type=["mp4", "avi", "mkv", "mov"],
    )

if video:
    chunks_folder = "./.cache/chunks"
    with st.status("Loading video....") as status:
        video_content = video.read()
        video_path = f"./.cache/{video.name}"

        audio_path = video_path.replace("mp4", "mp3")
        transcript_path = video_path.replace("mp4", "txt")
        with open(video_path, "wb") as f:
            f.write(video_content)

        status.update(label="Extracting audio...")
        extract_audio_from_video(video_path)

        status.update(label="Cutting audio segments...")
        cut_audio_in_chunks(audio_path, 1, chunks_folder)

        status.update(label="Transcribing auddio...")
        transcribe_chunks(chunks_folder, transcript_path)

    transcript_tab, summary_tab, qna_tab = st.tabs(
        [
            "Trasncript",
            "Summary",
            "Q&A",
        ]
    )

    with transcript_tab:
        with open(transcript_path, "r", encoding="utf-8") as file:
            st.write(file.read())

    with summary_tab:
        start = st.button("Generate summary")

        if start:
            loader = TextLoader(transcript_path, autodetect_encoding=True)
            docs = loader.load_and_split(text_splitter=splitter)
            first_summary_prompt = ChatPromptTemplate.from_template(
                """
                Write a concise summary of the following:
                "{text}"
                CONCISE SUMMARY:
            """
            )

            first_summary_chain = first_summary_prompt | llm | StrOutputParser()

            summary = first_summary_chain.invoke({"text": docs[0].page_content})

            refine_prompt = ChatPromptTemplate.from_template(
                """
                Your job is to produce a final summary. We have provided an existing summary up to a certain point: {existing_summary}
                We have the opportunity to refine the existing summary (only if needed) with some more context below.
                ------------
                {context}
                ------------
                Given the new context, refine the original summary.
                If the context isn't useful, REUTRN the original summary.
                """
            )

            refine_chain = refine_prompt | llm | StrOutputParser()

            with st.status("Summarizing...") as status:
                for i, doc in enumerate(docs[1:]):
                    status.update(label=f"Processing document {i+1}/{len(docs)-1}")
                    summary = refine_chain.invoke(
                        {
                            "existing_summary": summary,
                            "context": doc.page_content,
                        }
                    )
            st.write(summary)

    message = st.chat_input("Ask anything about the file.")

    with qna_tab:

        retriever = embed_file(transcript_path)
        send_message(
            "Alright, I'm ready! How can I help you today with this file?",
            "ai",
            save=False,
        )
        paint_history()
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                Answer the question using ONLY the following context. If you don't know the answer just say you don't know. DON'T make anything  up.
                
                Context: {context}     
                """,
                ),
                ("human", "{question}"),
            ]
        )
        chain = (
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
        )

        if message:
            send_message(message, "human")
            response = chain.invoke(message)
            send_message(response, "ai")

else:
    st.session_state["messages"] = []
