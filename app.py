import os
import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List

# LangChain & Gemini Imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.output_parsers import PydanticOutputParser

# 1. Load environment variables from the .env file (if present)
load_dotenv()

# 2. Configure the Streamlit Web Page
st.set_page_config(
    page_title="Technical Interview Live Simulator & Coach",
    page_icon="🎓",
    layout="wide"
)

# Total number of interview questions
MAX_TURNS = 5


# =====================================================================
# PART A: DATA SCHEMAS FOR THE FINAL SCORECARD (Pydantic)
# =====================================================================
# Why we need this: LLMs normally output unstructured text. By defining
# Pydantic classes, we force Gemini to return exact scores (out of 10)
# and structured model answers that we can neatly display in Streamlit.

class QuestionReview(BaseModel):
    question_number: int = Field(description="Question number (1 to 5)")
    question_asked: str = Field(description="The exact technical question asked by the interviewer")
    candidate_answer_summary: str = Field(description="Brief summary of the candidate's response")
    critique: str = Field(description="Constructive feedback on what was good and what was missing")
    model_answer: str = Field(description="Ideal, comprehensive technical answer expected in a placement interview")

class EvaluationScorecard(BaseModel):
    communication_score: int = Field(description="Score out of 10 for clarity, structure, and articulation")
    technical_depth_score: int = Field(description="Score out of 10 for conceptual accuracy and deep understanding")
    problem_solving_score: int = Field(description="Score out of 10 for analytical thinking, edge cases, and trade-offs")
    overall_summary: str = Field(description="Overall executive summary of the candidate's interview performance")
    key_strengths: List[str] = Field(description="Top 2-3 strengths demonstrated")
    areas_for_improvement: List[str] = Field(description="Top 2-3 areas to study before campus placements")
    question_reviews: List[QuestionReview] = Field(description="Detailed breakdown and model answers for each of the 5 questions")


# =====================================================================
# PART B: SESSION STATE & MEMORY INITIALIZATION
# =====================================================================
# Why we need this: Streamlit re-runs the entire Python script from top
# to bottom every time a user clicks a button or sends a message.
# st.session_state keeps variables alive across re-runs.

if "interview_active" not in st.session_state:
    st.session_state.interview_active = False
if "interview_completed" not in st.session_state:
    st.session_state.interview_completed = False
if "turn_count" not in st.session_state:
    st.session_state.turn_count = 0
if "scorecard" not in st.session_state:
    st.session_state.scorecard = None

# Initialize LangChain's Streamlit-native memory store
msgs = StreamlitChatMessageHistory(key="langchain_messages")


# =====================================================================
# PART C: AI LOGIC (INTERVIEWER CHAIN & EVALUATOR CHAIN)
# =====================================================================
def extract_text(content) -> str:
    """Extracts clean text if Gemini returns a list of content blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "") for block in content 
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return str(content)

def get_llm(api_key: str, temperature: float = 0.6):
    """Initializes the Google Gemini 3 Flash model."""
    return ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview",
        google_api_key=api_key,
        temperature=temperature
    )

def build_interview_chain(api_key: str, role: str, experience: str, focus_topics: str):
    """Builds the conversational AI chain with memory for the 5-turn interview."""
    llm = get_llm(api_key, temperature=0.6)

    system_prompt = f"""You are a Senior Technical Panelist conducting a campus placement interview.
Target Role: {role}
Candidate Experience Level: {experience}
Core Topics to Cover: {focus_topics}

INTERVIEW RULES:
1. You are conducting an adaptive {MAX_TURNS}-question technical interview.
2. Ask ONLY ONE question at a time. Keep your tone professional, encouraging, yet rigorous.
3. ADAPTIVE FOLLOW-UPS: Analyze the candidate's previous answer carefully:
   - If their answer is strong, probe deeper into edge cases, scalability, or real-world trade-offs.
   - If their answer is vague, partially incorrect, or brief, ask a clarifying follow-up or pivot to assess foundational understanding.
4. Do NOT provide the full answer or lecture the candidate during the interview—save feedback for the final evaluation.
5. Briefly acknowledge their answer in 1 sentence before asking the next question.
"""

    # Combine system instructions, chat history, and latest user input
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])

    chain = prompt | llm

    # Wrap the chain with message history so it remembers previous turns
    return RunnableWithMessageHistory(
        chain,
        lambda session_id: msgs,
        input_messages_key="input",
        history_messages_key="history"
    )

def generate_scorecard(api_key: str, role: str, experience: str) -> EvaluationScorecard:
    """Reads the entire 5-turn chat transcript and outputs the structured scorecard."""
    # Lower temperature (0.2) ensures consistent, objective grading
    llm = get_llm(api_key, temperature=0.2)
    parser = PydanticOutputParser(pydantic_object=EvaluationScorecard)

    # Convert stored chat messages into a clean text transcript
    transcript = ""
    for m in msgs.messages:
        speaker = "Interviewer" if m.type == "ai" else "Candidate"
        clean_text = extract_text(m.content)
        if "START_INTERVIEW_SESSION" in clean_text:
            continue
        transcript += f"{speaker}: {clean_text}\n\n"
    eval_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert Campus Placement Evaluation Coach.
Analyze the following technical interview transcript and generate a strict, constructive, and comprehensive Evaluation Scorecard.
{format_instructions}"""),
        ("human", """Target Role: {role}
Experience Level: {experience}

Interview Transcript:
{transcript}

Evaluate the candidate fairly based on their target experience level and return ONLY valid JSON matching the schema.""")
    ])

    chain = eval_prompt | llm | parser
    return chain.invoke({
        "role": role,
        "experience": experience,
        "transcript": transcript,
        "format_instructions": parser.get_format_instructions()
    })

def reset_interview():
    """Clears chat memory and resets all counters to start fresh."""
    msgs.clear()
    st.session_state.interview_active = False
    st.session_state.interview_completed = False
    st.session_state.turn_count = 0
    st.session_state.scorecard = None


# =====================================================================
# PART D: SIDEBAR USER INTERFACE (SETUP & CONTROLS)
# =====================================================================
with st.sidebar:
    st.title("⚙️ Interview Setup")
    st.caption("Campus Placement Simulator")

    env_key = os.getenv("GOOGLE_API_KEY", "")
    api_key = st.text_input(
        "Google Gemini API Key",
        value=env_key,
        type="password",
        help="Get your free API key from https://aistudio.google.com/"
    )

    st.divider()

    role_options = [
        "Backend Developer",
        "Cloud Engineer",
        "Data Scientist",
        "Full Stack Developer",
        "Machine Learning Engineer",
        "DevOps / SRE Engineer",
        "Cybersecurity Analyst",
        "Frontend Engineer"
    ]
    selected_role = st.selectbox(
        "🎯 Target Role",
        options=role_options,
        disabled=st.session_state.interview_active
    )

    experience_options = [
        "Fresher / Final Year Student (0 yrs)",
        "Internship Experience (0 - 1 yr)",
        "Early Career (1 - 2 yrs)",
        "Experienced (2+ yrs)"
    ]
    selected_exp = st.selectbox(
        "📈 Experience Level",
        options=experience_options,
        disabled=st.session_state.interview_active
    )

    focus_topics = st.text_input(
        "🛠️ Optional Focus Areas",
        placeholder="e.g., Python, SQL, AWS, System Design, DSA",
        value="Core fundamentals, problem-solving, and practical scenarios",
        disabled=st.session_state.interview_active
    )

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        start_btn = st.button(
            "🚀 Start",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.interview_active or not api_key
        )
    with col2:
        reset_btn = st.button(
            "🔄 Reset",
            use_container_width=True
        )

    if reset_btn:
        reset_interview()
        st.rerun()

    # Trigger Question 1 when the user clicks Start
    if start_btn and api_key:
        reset_interview()
        st.session_state.interview_active = True
        st.session_state.turn_count = 1

        with st.spinner("Interviewer is preparing Question 1..."):
            chain = build_interview_chain(api_key, selected_role, selected_exp, focus_topics)
            chain.invoke(
                {"input": f"START_INTERVIEW_SESSION: Greet the candidate briefly and ask Question 1 of {MAX_TURNS} for the {selected_role} interview."},
                config={"configurable": {"session_id": "placement_session"}}
            )
        st.rerun()


# =====================================================================
# PART E: MAIN CHAT INTERFACE & 5-TURN LOOP
# =====================================================================
st.title("🎓 Technical Interview Live Simulator & Feedback Coach")
st.markdown(
    f"**Role:** `{selected_role}` | **Level:** `{selected_exp}` | **Format:** `5-Turn Adaptive Technical Dialogue`"
)

if not api_key:
    st.info("👈 Please enter your **Google Gemini API Key** in the sidebar and click **Start** to begin your mock placement interview.")
    st.stop()

if not st.session_state.interview_active and not st.session_state.interview_completed:
    st.info("👈 Configure your target role and experience level in the sidebar, then click **🚀 Start** to launch your live technical interview.")
    st.stop()

# Visual Progress Bar (Question 1 to 5)
progress_val = min(st.session_state.turn_count / MAX_TURNS, 1.0)
if st.session_state.interview_completed:
    st.progress(1.0, text="✅ Interview Completed! View your Evaluation Scorecard below.")
else:
    st.progress(progress_val, text=f"📍 Question {st.session_state.turn_count} of {MAX_TURNS}")

# Render Existing Chat Messages
chat_container = st.container()
with chat_container:
    for msg in msgs.messages:
        clean_text = extract_text(msg.content)
        if "START_INTERVIEW_SESSION" in clean_text:
            continue
        avatar = "🧑‍💻" if msg.type == "human" else "👔"
        with st.chat_message(msg.type, avatar=avatar):
            st.markdown(clean_text)

# Accept Student Answer
if st.session_state.interview_active and not st.session_state.interview_completed:
    user_answer = st.chat_input(f"Type your answer for Question {st.session_state.turn_count}...")

    if user_answer:
        with chat_container:
            with st.chat_message("human", avatar="🧑‍💻"):
                st.markdown(user_answer)

        # If turns 1 through 4: generate the next adaptive question
        if st.session_state.turn_count < MAX_TURNS:
            st.session_state.turn_count += 1
            next_q_num = st.session_state.turn_count

            with st.spinner(f"Analyzing response & formulating Question {next_q_num} of {MAX_TURNS}..."):
                chain = build_interview_chain(api_key, selected_role, selected_exp, focus_topics)
                prompt_with_directive = (
                    f"{user_answer}\n\n"
                    f"[System Note: That was the candidate's response to Question {next_q_num - 1}. "
                    f"Now ask Question {next_q_num} of {MAX_TURNS}. Make it an adaptive follow-up based on their answer or probe a new relevant technical concept.]"
                )
                chain.invoke(
                    {"input": prompt_with_directive},
                    config={"configurable": {"session_id": "placement_session"}}
                )
                # Keep history clean by removing the system note after invocation
                msgs.messages[-2].content = user_answer

            st.rerun()

        # If turn 5 (final question answered): end interview and trigger scorecard
        else:
            msgs.add_user_message(user_answer)
            msgs.add_ai_message(
                "Thank you for completing all 5 questions of the technical interview! "
                "Please wait a moment while I compile your comprehensive Evaluation Scorecard and model answers."
            )
            st.session_state.interview_active = False
            st.session_state.interview_completed = True
            st.rerun()


# =====================================================================
# PART F: SCORECARD & MODEL ANSWERS DASHBOARD
# =====================================================================
if st.session_state.interview_completed:
    st.divider()
    st.header("📊 Comprehensive Evaluation Scorecard")

    if st.session_state.scorecard is None:
        with st.spinner("🧠 Evaluating your technical depth, problem-solving, and communication skills..."):
            try:
                st.session_state.scorecard = generate_scorecard(api_key, selected_role, selected_exp)
            except Exception as e:
                st.error(f"Error generating scorecard: {e}")
                st.stop()

    sc: EvaluationScorecard = st.session_state.scorecard

    # 1. Display Top-Level Score Cards (Out of 10)
    avg_score = round((sc.communication_score + sc.technical_depth_score + sc.problem_solving_score) / 3, 1)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🗣️ Communication", f"{sc.communication_score} / 10")
    m2.metric("⚙️ Technical Depth", f"{sc.technical_depth_score} / 10")
    m3.metric("🧩 Problem-Solving", f"{sc.problem_solving_score} / 10")
    m4.metric("🏆 Overall Rating", f"{avg_score} / 10")

    # 2. Executive Summary
    st.subheader("📝 Executive Summary")
    st.info(sc.overall_summary)

    # 3. Strengths & Areas for Improvement
    col_str, col_imp = st.columns(2)
    with col_str:
        st.success("### ✅ Key Strengths")
        for s in sc.key_strengths:
            st.markdown(f"- {s}")

    with col_imp:
        st.warning("### 🎯 Areas for Improvement")
        for imp in sc.areas_for_improvement:
            st.markdown(f"- {imp}")

    # 4. Expandable Question-by-Question Review & Model Answers
    st.divider()
    st.subheader("🔍 Question-by-Question Breakdown & Model Answers")

    for qr in sc.question_reviews:
        with st.expander(f"Question {qr.question_number}: {qr.question_asked}", expanded=(qr.question_number == 1)):
            st.markdown(f"**🗣️ Your Answer Summary:**\n> {qr.candidate_answer_summary}")
            st.markdown(f"**⚖️ Panelist Critique:**\n{qr.critique}")
            st.markdown("**💡 Ideal Model Answer:**")
            st.success(qr.model_answer)