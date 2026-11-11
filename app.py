
import streamlit as st
import os

from pypdf import PdfReader
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from typing import TypedDict


# ---------------------------------
# 1. إعداد واجهة التطبيق
# ---------------------------------

st.set_page_config(
    page_title="Smart PDF Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Smart PDF Assistant Agent")

st.write(
    "Upload a PDF and let your AI Agent analyze it."
)


# ---------------------------------
# 2. الاتصال بنموذج الذكاء الاصطناعي Groq
# ---------------------------------

api_key = st.secrets.get(
    "GROQ_API_KEY",
    os.environ.get("GROQ_API_KEY", "")
)

if not api_key:
    st.error(
        "Groq API Key is missing. "
        "Please configure GROQ_API_KEY in Streamlit Secrets."
    )
    st.stop()

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    api_key=api_key
)


# ---------------------------------
# 3. تعريف بيانات الـ Agent
# ---------------------------------

class AgentState(TypedDict):
    task: str
    document_text: str
    question: str
    answer: str


# ---------------------------------
# 4. إنشاء وظائف الـ Agent
# ---------------------------------

def summarize_node(state: AgentState):

    prompt = f"""
    You are an expert PDF document assistant.

    Summarize the following document in Arabic.

    Requirements:
    - Organize the summary into clear sections.
    - Mention the most important information.
    - Highlight important numbers and dates.
    - Use clear and professional Arabic.

    Document:
    {state["document_text"][:20000]}
    """

    response = llm.invoke(prompt)

    return {
        "answer": response.content
    }


def question_node(state: AgentState):

    prompt = f"""
    You are a PDF assistant.

    Answer the user's question using only the
    information available in the document.

    Rules:
    - Answer in Arabic.
    - Do not invent information.
    - If the answer is not available in the document,
      clearly say that it is not mentioned.
    - Give a clear and direct answer.

    Document:
    {state["document_text"][:20000]}

    Question:
    {state["question"]}
    """

    response = llm.invoke(prompt)

    return {
        "answer": response.content
    }


def actions_node(state: AgentState):

    prompt = f"""
    Analyze this document and extract the following:

    1. Required actions.
    2. Recommendations.
    3. Important deadlines or dates.
    4. Responsible parties.
    5. Important follow-up tasks.

    Requirements:
    - Answer in Arabic.
    - Organize the information into sections.
    - Use bullet points.
    - Do not invent information that is not in the document.

    Document:
    {state["document_text"][:20000]}
    """

    response = llm.invoke(prompt)

    return {
        "answer": response.content
    }


# ---------------------------------
# 5. تحديد المسار المناسب للمهمة
# ---------------------------------

def route_task(state: AgentState):

    return state["task"]


# ---------------------------------
# 6. بناء Agent باستخدام LangGraph
# ---------------------------------

builder = StateGraph(AgentState)

builder.add_node(
    "summarize",
    summarize_node
)

builder.add_node(
    "question",
    question_node
)

builder.add_node(
    "actions",
    actions_node
)


# تحديد نقطة البداية

builder.add_conditional_edges(
    START,
    route_task,
    {
        "summarize": "summarize",
        "question": "question",
        "actions": "actions"
    }
)


# تحديد نقطة النهاية

builder.add_edge(
    "summarize",
    END
)

builder.add_edge(
    "question",
    END
)

builder.add_edge(
    "actions",
    END
)


# تشغيل Agent

agent = builder.compile()


# ---------------------------------
# 7. رفع ملف PDF
# ---------------------------------

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)


if uploaded_file is not None:

    try:

        pdf_reader = PdfReader(uploaded_file)

        extracted_text = ""

        for page_number, page in enumerate(
            pdf_reader.pages
        ):

            page_text = page.extract_text() or ""

            extracted_text += (
                f"\nPage {page_number + 1}:\n"
                f"{page_text}"
            )

        st.success(
            "PDF uploaded successfully!"
        )

        st.write(
            f"Number of pages: {len(pdf_reader.pages)}"
        )


        # ---------------------------------
        # 8. اختيار المهمة
        # ---------------------------------

        st.subheader("Choose your task")

        task_labels = {

            "Summarize PDF": "summarize",

            "Ask a Question": "question",

            "Extract Action Items": "actions"

        }

        selected_task = st.selectbox(
            "What would you like the Agent to do?",
            list(task_labels.keys())
        )


        question = ""

        if selected_task == "Ask a Question":

            question = st.text_input(
                "Write your question:"
            )


        # ---------------------------------
        # 9. تشغيل الـ Agent
        # ---------------------------------

        if st.button(
            "Run AI Agent",
            type="primary"
        ):

            if not extracted_text.strip():

                st.warning(
                    "No readable text found in this PDF."
                )

            elif (
                selected_task == "Ask a Question"
                and not question.strip()
            ):

                st.warning(
                    "Please enter your question."
                )

            else:

                task = task_labels[selected_task]

                initial_state = {

                    "task": task,

                    "document_text": extracted_text,

                    "question": question,

                    "answer": ""

                }

                try:

                    with st.spinner(
                        "AI Agent is working..."
                    ):

                        result = agent.invoke(
                            initial_state
                        )

                    st.subheader(
                        "AI Agent Result"
                    )

                    st.markdown(
                        result["answer"]
                    )

                except Exception as error:

                    st.error(
                        f"AI Error: {error}"
                    )

    except Exception as error:

        st.error(
            f"PDF Error: {error}"
        )
