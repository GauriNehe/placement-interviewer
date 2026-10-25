import os
import base64
from typing import List
import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

# 1. Load Environment Variables
load_dotenv()

st.set_page_config(
    page_title="Technical Interview Live Simulator & Coach",
    page_icon="🎓",
    layout="wide"
)

# 2. Pydantic Schemas for the Final Evaluation Scorecard
class QuestionCritique(BaseModel):
    question_number: int = Field(description="Question number (1 to 5)")
    question_asked: str = Field(description="The technical question asked by the interviewer")
    candidate_answer_summary: str = Field(description="Brief summary of the candidate's response")
    feedback: str = Field(description="Constructive critique highlighting strengths and gaps")
    ideal_answer: str = Field(description="Concise, high-scoring model answer for this question")

class InterviewScorecard(BaseModel):
    communication_score: int = Field(description="Score from 1 to 10 for clarity, structure, and articulation")
    technical_depth_score: int = Field(description="Score from 1 to 10 for conceptual accuracy and system depth")
    problem_solving_score: int = Field(description="Score from 1 to 10 for edge cases, scalability, and trade-offs")
    overall_summary: str = Field(description="Executive summary of the candidate's overall interview performance")
    key_strengths: List[str] = Field(description="Top 2-3 technical or communication strengths demonstrated")
    areas_for_improvement: List[str] = Field(description="Top 2-3 actionable areas to improve before campus placements")
    question_breakdown: List[QuestionCritique] = Field(description="Detailed critique and model answer for each of the 5 questions")


# 3. Helper to Clean Model Output Text
def extract_clean_text(content) -> str:
    """Safely extracts plain string from string or list-based model outputs."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
        return "\n".join(parts).strip()
    return str(content).strip()


# 4. Fast LLM Initializer with Instant Quota Fallbacks
# 4. Fast LLM Initializer with Instant Quota Fallbacks
def get_llm(api_key: str, temperature: float = 0.6):
    """Initializes Gemini with fast failovers so 429 quota limits never hang the app."""
    primary = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        google_api_key=api_key,
        temperature=temperature,
        max_retries=1,
        timeout=15
    )
    backup1 = ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
        google_api_key=api_key,
        temperature=temperature,
        max_retries=1,
        timeout=15
    )
    backup2 = ChatGoogleGenerativeAI(
        model="gemma-3-27b-it",
        google_api_key=api_key,
        temperature=temperature,
        max_retries=1,
        timeout=15
    )
    return primary.with_fallbacks([backup1, backup2])

# 5. Ultra-Fast In-Memory Voice Transcription (< 1.5s)
def transcribe_voice_fast(audio_bytes: bytes, api_key: str) -> str:
    """Transcribes audio bytes directly in memory without slow file uploads."""
    llm = get_llm(api_key=api_key, temperature=0.0)
    b64_audio = base64.b64encode(audio_bytes).decode("utf-8")
    
    msg = HumanMessage(
        content=[
            {
                "type": "text",
                "text": (
                    "Transcribe this technical interview answer accurately into English text. "
                    "Return ONLY the spoken words without any intro, quotes, or commentary."
                )
            },
            {
                "type": "media",
                "mime_type": "audio/wav",
                "data": b64_audio
            }
        ]
    )
    response = llm.invoke([msg])
    return extract_clean_text(response.content)


# 6. Conversational Interview Chain with Memory
def get_interview_chain(api_key: str, chat_history: StreamlitChatMessageHistory):
    llm = get_llm(api_key=api_key, temperature=0.6)
    
    system_prompt = """You are a senior, encouraging, yet rigorous Technical Placement Interviewer conducting a 5-question live technical interview.
Target Role: {role}
Candidate Experience Level: {experience}
Focus Areas: {focus_areas}

STRICT INTERVIEW RULES:
1. Ask EXACTLY ONE clear, scenario-based or conceptual technical question at a time.
2. NEVER provide the answer or lengthy explanations during the interview.
3. Adapt your follow-up question dynamically based on the candidate's previous answer:
   - If their answer was strong, probe deeper into scalability, edge cases, or trade-offs.
   - If their answer was weak or incomplete, pivot gently to test fundamental understanding.
4. Keep your responses concise (2-4 sentences max) and clearly state the next question."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])
    
    chain = prompt | llm
    return RunnableWithMessageHistory(
        chain,
        lambda session_id: chat_history,
        input_messages_key="input",
        history_messages_key="history"
    )


# 7. Scorecard Evaluation Generator
def generate_scorecard(api_key: str, role: str, experience: str, messages) -> InterviewScorecard:
    llm = get_llm(api_key=api_key, temperature=0.2)
    parser = PydanticOutputParser(pydantic_object=InterviewScorecard)
    
    transcript_lines = []
    for msg in messages:
        speaker = "Interviewer" if msg.type == "ai" else "Candidate"
        transcript_lines.append(f"{speaker}: {extract_clean_text(msg.content)}")
    transcript_text = "\n\n".join(transcript_lines)
    
    eval_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert technical hiring manager evaluating a campus placement interview transcript. Provide an objective, constructive evaluation strictly following the JSON schema."),
        ("human", """Evaluate this 5-turn technical interview for:
Role: {role}
Experience Level: {experience}

INTERVIEW TRANSCRIPT:
{transcript}

{format_instructions}""")
    ])
    
    chain = eval_prompt | llm | parser
    return chain.invoke({
        "role": role,
        "experience": experience,
        "transcript": transcript_text,
        "format_instructions": parser.get_format_instructions()
    })


# 8. Initialize Streamlit Session State
if "interview_active" not in st.session_state:
    st.session_state.interview_active = False
if "interview_completed" not in st.session_state:
    st.session_state.interview_completed = False
if "question_count" not in st.session_state:
    st.session_state.question_count = 0
if "scorecard" not in st.session_state:
    st.session_state.scorecard = None

chat_history = StreamlitChatMessageHistory(key="langchain_messages")


# 9. Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Interview Setup")
    st.caption("Campus Placement Simulator")
    
    default_key = ""
    try:
        default_key = st.secrets.get("GOOGLE_API_KEY", "")
    except Exception:
        default_key = os.getenv("GOOGLE_API_KEY", "")
    if not default_key:
        default_key = os.getenv("GOOGLE_API_KEY", "")
        
    api_key = st.text_input(
        "Google Gemini API Key",
        value=default_key,
        type="password",
        help="Enter your Google AI Studio API Key"
    )
    
    st.divider()
    
    role = st.selectbox(
        "Target Role",
        [
            "Backend Developer",
            "Full Stack Developer",
            "Data Scientist / AI & ML Engineer",
            "Cloud / DevOps Engineer",
            "Frontend Developer",
            "Software Development Engineer (SDE)"
        ]
    )
    
    experience = st.selectbox(
        "Experience Level",
        [
            "Fresher / Final Year Student (0 yrs)",
            "Intern / Junior (0-1 yrs)",
            "Associate Engineer (1-2 yrs)"
        ]
    )
    
    focus_areas = st.text_input(
        "Optional Focus Areas",
        value="Core fundamentals, problem-solving, and practical scenarios"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        start_btn = st.button("🚀 Start", use_container_width=True, type="primary")
    with col2:
        reset_btn = st.button("🔄 Reset", use_container_width=True)

    if reset_btn:
        chat_history.clear()
        st.session_state.interview_active = False
        st.session_state.interview_completed = False
        st.session_state.question_count = 0
        st.session_state.scorecard = None
        st.rerun()

    if start_btn:
        if not api_key:
            st.error("Please enter your Google Gemini API Key in the sidebar first.")
        else:
            chat_history.clear()
            st.session_state.interview_active = True
            st.session_state.interview_completed = False
            st.session_state.question_count = 1
            st.session_state.scorecard = None
            
            with st.spinner("Starting your technical interview..."):
                try:
                    chain = get_interview_chain(api_key, chat_history)
                    opening_instruction = (
                        f"Start the interview now. Greet the candidate briefly and ask Question 1 of 5 "
                        f"tailored for a {experience} {role} focusing on {focus_areas}."
                    )
                    chain.invoke(
                        {"input": opening_instruction, "role": role, "experience": experience, "focus_areas": focus_areas},
                        config={"configurable": {"session_id": "placement_session"}}
                    )
                    # Replace the hidden trigger prompt in history so only the AI greeting shows first
                    if len(chat_history.messages) >= 2:
                        ai_opening = chat_history.messages[-1]
                        chat_history.clear()
                        chat_history.add_ai_message(extract_clean_text(ai_opening.content))
                    st.rerun()
                except Exception as e:
                    st.session_state.interview_active = False
                    st.error(f"Could not start the interview: {e}")


# 10. Main Application UI
st.title("🎓 Technical Interview Live Simulator & Feedback Scoring Coach")
st.markdown(
    f"**Role:** `{role}` | **Level:** `{experience}` | **Format:** `5-Turn Adaptive Technical Dialogue`"
)

if not st.session_state.interview_active and not st.session_state.interview_completed:
    st.info("👈 Choose a target role and experience level in the sidebar, then select **Start**.")

# Display Progress Bar
if st.session_state.interview_active:
    q_num = min(st.session_state.question_count, 5)
    st.markdown(f"📍 **Question {q_num} of 5**")
    st.progress(q_num / 5.0)

# Render Conversation Transcript
for msg in chat_history.messages:
    role_label = "assistant" if msg.type == "ai" else "user"
    with st.chat_message(role_label):
        st.markdown(extract_clean_text(msg.content))


# 11. Handle Candidate Answer (Text OR Fast Voice Input)
def process_candidate_answer(user_answer: str):
    """Processes the candidate's response and either asks the next question or ends the interview."""
    chain = get_interview_chain(api_key, chat_history)
    current_q = st.session_state.question_count
    
    if current_q < 5:
        next_q = current_q + 1
        prompt_with_guidance = (
            f"Candidate's Answer to Question {current_q}: {user_answer}\n\n"
            f"[System Note: Briefly acknowledge the answer in 1 sentence and ask follow-up Question {next_q} of 5.]"
        )
        with st.spinner(f"Interviewer is preparing Question {next_q} of 5..."):
            response = chain.invoke(
                {"input": prompt_with_guidance, "role": role, "experience": experience, "focus_areas": focus_areas},
                config={"configurable": {"session_id": "placement_session"}}
            )
            # Store clean user message in UI history instead of system note wrapper
            chat_history.messages[-2].content = user_answer
            chat_history.messages[-1].content = extract_clean_text(response.content)
            st.session_state.question_count = next_q
            st.rerun()
    else:
        # Final 5th answer received -> Wrap up and evaluate
        chat_history.add_user_message(user_answer)
        chat_history.add_ai_message(
            "Thank you for completing all 5 questions! Generating your detailed Evaluation Scorecard below..."
        )
        st.session_state.interview_active = False
        st.session_state.interview_completed = True
        st.rerun()


if st.session_state.interview_active:
    q_num = st.session_state.question_count
    
    # Dynamic key ensures the audio recorder resets cleanly on each new question
    audio_val = st.audio_input(
        f"🎙️ Record voice answer for Question {q_num} (or type below):",
        key=f"voice_q_{q_num}"
    )
    text_val = st.chat_input(f"Type your answer for Question {q_num}...")
    
    if text_val:
        process_candidate_answer(text_val.strip())
    elif audio_val is not None:
        audio_bytes = audio_val.read()
        last_hash_key = f"processed_audio_{q_num}"
        if st.session_state.get(last_hash_key) != audio_bytes:
            st.session_state[last_hash_key] = audio_bytes
            with st.spinner("⚡ Transcribing your voice response..."):
                try:
                    transcribed_text = transcribe_voice_fast(audio_bytes, api_key)
                    if transcribed_text:
                        process_candidate_answer(transcribed_text)
                    else:
                        st.warning("Could not hear anything clearly. Please try recording again or type your answer.")
                except Exception as e:
                    st.error(f"Voice transcription error: {e}. Please type your answer below.")


# 12. Render Post-Interview Evaluation Scorecard
if st.session_state.interview_completed:
    st.divider()
    st.subheader("📊 Post-Interview Evaluation Scorecard")
    
    if st.session_state.scorecard is None:
        with st.spinner("Evaluating your technical depth, problem-solving, and communication skills..."):
            try:
                st.session_state.scorecard = generate_scorecard(
                    api_key=api_key,
                    role=role,
                    experience=experience,
                    messages=chat_history.messages
                )
            except Exception as e:
                st.error(f"Error generating structured scorecard: {e}")
    
    sc: InterviewScorecard = st.session_state.scorecard
    if sc:
        m1, m2, m3, m4 = st.columns(4)
        avg_score = round((sc.communication_score + sc.technical_depth_score + sc.problem_solving_score) / 3, 1)
        m1.metric("🗣️ Communication", f"{sc.communication_score} / 10")
        m2.metric("⚙️ Technical Depth", f"{sc.technical_depth_score} / 10")
        m3.metric("🧩 Problem-Solving", f"{sc.problem_solving_score} / 10")
        m4.metric("🏆 Overall Rating", f"{avg_score} / 10")
        
        st.info(f"**Executive Summary:** {sc.overall_summary}")
        
        c1, c2 = st.columns(2)
        with c1:
            st.success("### ✅ Key Strengths")
            for s in sc.key_strengths:
                st.markdown(f"- {s}")
        with c2:
            st.warning("### 🎯 Areas for Improvement")
            for a in sc.areas_for_improvement:
                st.markdown(f"- {a}")
                
        st.subheader("📝 Question-by-Question Critique & Model Answers")
        for item in sc.question_breakdown:
            with st.expander(f"Question {item.question_number}: {item.question_asked}", expanded=False):
                st.markdown(f"**Your Response Summary:** {item.candidate_answer_summary}")
                st.markdown(f"**Panelist Critique:** {item.feedback}")
                st.markdown(f"**💡 Ideal Model Answer:**\n> {item.ideal_answer}")