import os
import base64
import hashlib
import io
import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List

# LangChain & Gemini Imports
from google import genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.output_parsers import PydanticOutputParser

# 1. Load environment variables from the .env file (if present)
load_dotenv()

# 2. Configure the Streamlit Web Page
st.set_page_config(
    page_title="AI Technical Placement Interviewer",
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
    question_number: int = Field(ge=1, le=MAX_TURNS, description="Question number (1 to 5)")
    question_asked: str = Field(description="The exact technical question asked by the interviewer")
    candidate_answer_summary: str = Field(description="Brief summary of the candidate's response")
    critique: str = Field(description="Constructive feedback on what was good and what was missing")
    model_answer: str = Field(description="Ideal, comprehensive technical answer expected in a placement interview")

class EvaluationScorecard(BaseModel):
    communication_score: int = Field(ge=0, le=10, description="Score out of 10 for clarity, structure, and articulation")
    technical_depth_score: int = Field(ge=0, le=10, description="Score out of 10 for conceptual accuracy and deep understanding")
    problem_solving_score: int = Field(ge=0, le=10, description="Score out of 10 for analytical thinking, edge cases, and trade-offs")
    overall_summary: str = Field(description="Overall executive summary of the candidate's interview performance")
    key_strengths: List[str] = Field(description="Top 2-3 strengths demonstrated")
    areas_for_improvement: List[str] = Field(description="Top 2-3 areas to study before campus placements")
    question_reviews: List[QuestionReview] = Field(
        min_length=MAX_TURNS,
        max_length=MAX_TURNS,
        description="Detailed breakdown and model answers for each of the 5 questions"
    )


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
if "interview_role" not in st.session_state:
    st.session_state.interview_role = None
if "interview_experience" not in st.session_state:
    st.session_state.interview_experience = None
if "interview_focus" not in st.session_state:
    st.session_state.interview_focus = None
if "pending_answer" not in st.session_state:
    st.session_state.pending_answer = None
if "interview_error" not in st.session_state:
    st.session_state.interview_error = None
if "scorecard_error" not in st.session_state:
    st.session_state.scorecard_error = None
if "interviewer_audio" not in st.session_state:
    st.session_state.interviewer_audio = {}
if "voice_output_enabled" not in st.session_state:
    st.session_state.voice_output_enabled = True
if "voice_output_error" not in st.session_state:
    st.session_state.voice_output_error = None
if "voice_input_error" not in st.session_state:
    st.session_state.voice_input_error = None
if "last_processed_audio_hash" not in st.session_state:
    st.session_state.last_processed_audio_hash = None

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
    primary = ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
        google_api_key=api_key,
        temperature=temperature
    )
    backup1 = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        google_api_key=api_key,
        temperature=temperature
    )
    backup2 = ChatGoogleGenerativeAI(
        model="gemma-3-27b-it",
        google_api_key=api_key,
        temperature=temperature
    )
    return primary.with_fallbacks([backup1, backup2])

def transcribe_audio(api_key: str, audio_bytes: bytes, mime_type: str) -> str:
    """Transcribes a recorded answer with the candidate's configured Gemini API key."""
    client = genai.Client(api_key=api_key)
    audio_file = client.files.upload(
        file=io.BytesIO(audio_bytes),
        config={"mime_type": mime_type}
    )
    try:
        interaction = client.interactions.create(
            model="gemini-3.5-transcribe",
            input=[{
                "type": "audio",
                "uri": audio_file.uri,
                "mime_type": audio_file.mime_type
            }]
        )
    finally:
        client.files.delete(name=audio_file.name)
    transcript = (interaction.output_text or "").strip()
    if not transcript:
        raise ValueError("Gemini did not return a transcription. Please try recording again.")
    return transcript

def generate_speech(api_key: str, text: str) -> bytes:
    """Creates a WAV audio response using Gemini text-to-speech."""
    client = genai.Client(api_key=api_key)
    interaction = client.interactions.create(
        model="gemini-3.8-flash-tts",
        input=[{
            "type": "user_input",
            "content": [{
                "type": "text",
                "text": text,
                "annotations": [{
                    "type": "speech_metadata",
                    "style": "Clear, professional, and encouraging."
                }]
            }]
        }],
        response_format={"type": "audio"},
        generation_config={"speech_config": [{"voice": "Kore"}]}
    )
    if not interaction.output_audio or not interaction.output_audio.data:
        raise ValueError("Gemini did not return spoken audio.")
    return base64.b64decode(interaction.output_audio.data)

def build_interview_chain(api_key: str, role: str, experience: str, focus_topics: str):
    """Builds the conversational AI chain with memory for the 5-turn interview."""
    llm = get_llm(api_key, temperature=0.6)

    system_prompt = f"""You are a Senior Technical Panelist conducting a campus placement interview.
Target Role: {role}
Candidate Experience Level: {experience}
Core Topics to Cover: {focus_topics}

INTERVIEW RULES:
1. You are conducting an adaptive {MAX_TURNS}-question technical interview. The candidate's first question is already in the conversation.
2. Ask ONLY ONE question at a time. Keep your tone professional, encouraging, yet rigorous.
3. ADAPTIVE FOLLOW-UPS: Analyze the candidate's previous answer carefully:
   - If their answer is strong, probe deeper into edge cases, scalability, or real-world trade-offs.
   - If their answer is vague, partially incorrect, or brief, ask a clarifying follow-up or pivot to assess foundational understanding.
4. Do NOT provide the full answer or lecture the candidate during the interview—save feedback for the final evaluation.
5. Briefly acknowledge their answer in 1 sentence before asking the next question. Do not add a second question or a multi-part list.
6. Keep every question appropriate to the target role and stated years of experience.
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
Score each category from 0 to 10 inclusive. Include exactly five question reviews, numbered 1 through 5, each with a role-appropriate model answer.
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
    for key in list(st.session_state):
        if key.startswith(("candidate_audio_", "answer_draft_")):
            del st.session_state[key]
    msgs.clear()
    st.session_state.interview_active = False
    st.session_state.interview_completed = False
    st.session_state.turn_count = 0
    st.session_state.scorecard = None
    st.session_state.interview_role = None
    st.session_state.interview_experience = None
    st.session_state.interview_focus = None
    st.session_state.pending_answer = None
    st.session_state.interview_error = None
    st.session_state.scorecard_error = None
    st.session_state.interviewer_audio = {}
    st.session_state.voice_output_error = None
    st.session_state.voice_input_error = None
    st.session_state.last_processed_audio_hash = None


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

    setup_locked = st.session_state.interview_active or st.session_state.interview_completed
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
        disabled=setup_locked
    )

    selected_exp = st.number_input(
        "📈 Years of experience",
        min_value=0,
        max_value=40,
        value=0,
        step=1,
        disabled=setup_locked,
        help="Use 0 for a student or new graduate."
    )

    focus_topics = st.text_input(
        "🛠️ Optional Focus Areas",
        placeholder="e.g., Python, SQL, AWS, System Design, DSA",
        value="Core fundamentals, problem-solving, and practical scenarios",
        disabled=setup_locked
    )
    voice_output_enabled = st.checkbox(
        "🔊 Speak interviewer questions",
        value=True,
        disabled=setup_locked,
        help="Uses Gemini text-to-speech with your configured API key."
    )

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        start_btn = st.button(
            "🚀 Start",
            type="primary",
            use_container_width=True,
            disabled=setup_locked or not api_key
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
        st.session_state.interview_role = selected_role
        st.session_state.interview_experience = selected_exp
        st.session_state.interview_focus = focus_topics
        st.session_state.voice_output_enabled = voice_output_enabled

        start_succeeded = False
        with st.spinner("Interviewer is preparing Question 1..."):
            try:
                chain = build_interview_chain(api_key, selected_role, f"{selected_exp} years", focus_topics)
                chain.invoke(
                    {"input": f"START_INTERVIEW_SESSION: Greet the candidate briefly and ask Question 1 of {MAX_TURNS} for the {selected_role} interview."},
                    config={"configurable": {"session_id": "placement_session"}}
                )
                if voice_output_enabled:
                    try:
                        st.session_state.interviewer_audio[len(msgs.messages) - 1] = generate_speech(
                            api_key,
                            extract_text(msgs.messages[-1].content)
                        )
                    except Exception as exc:
                        st.session_state.voice_output_error = f"Could not create spoken interviewer audio: {exc}"
                st.session_state.interview_active = True
                st.session_state.turn_count = 1
                start_succeeded = True
            except Exception as exc:
                st.session_state.interview_error = f"Could not start the interview: {exc}"
        if start_succeeded:
            st.rerun()


# =====================================================================
# PART E: MAIN CHAT INTERFACE & 5-TURN LOOP
# =====================================================================
st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1180px;}
    [data-testid="stMetric"] {background: rgba(127, 127, 127, 0.08); padding: 1rem; border-radius: 0.8rem;}
    </style>
    """,
    unsafe_allow_html=True
)

role_label = st.session_state.interview_role or selected_role
experience_label = st.session_state.interview_experience
st.title("Technical Placement Interviewer")
st.markdown(
    "Practice a realistic, adaptive technical interview and get actionable feedback with model answers."
)

if not api_key and st.session_state.scorecard is None:
    st.info("Add your Google Gemini API key in the sidebar to start your mock interview.")
    st.stop()

if not st.session_state.interview_active and not st.session_state.interview_completed:
    if st.session_state.interview_error:
        st.error(st.session_state.interview_error)
    st.info("Choose a target role and your years of experience in the sidebar, then select **Start**.")
    st.stop()

st.markdown(
    f"**{role_label}** &nbsp; · &nbsp; {experience_label} years of experience "
    f"&nbsp; · &nbsp; {MAX_TURNS}-question adaptive interview"
)

if st.session_state.interview_completed:
    st.progress(1.0, text="Interview completed")
else:
    st.progress(
        min(st.session_state.turn_count / MAX_TURNS, 1.0),
        text=f"Question {st.session_state.turn_count} of {MAX_TURNS}"
    )

if st.session_state.voice_output_error:
    st.warning(st.session_state.voice_output_error)

chat_container = st.container()
with chat_container:
    for index, msg in enumerate(msgs.messages):
        clean_text = extract_text(msg.content)
        if "START_INTERVIEW_SESSION" in clean_text:
            continue
        avatar = "🧑‍💻" if msg.type == "human" else "👔"
        role = "user" if msg.type == "human" else "assistant"
        with st.chat_message(role, avatar=avatar):
            st.markdown(clean_text)
            if msg.type == "ai" and index in st.session_state.interviewer_audio:
                st.audio(st.session_state.interviewer_audio[index], format="audio/wav")

    if st.session_state.pending_answer:
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(st.session_state.pending_answer)

if st.session_state.interview_active and not st.session_state.interview_completed:
    turn_number = st.session_state.turn_count
    audio_key = f"candidate_audio_{turn_number}"
    draft_key = f"answer_draft_{turn_number}"

    st.caption(
        "Record an answer or type it below. Recordings are uploaded to Gemini for transcription and deleted afterward."
    )
    recorded_audio = st.audio_input(
        "Record your answer",
        sample_rate=16000,
        key=audio_key
    )
    if recorded_audio:
        audio_bytes = recorded_audio.getvalue()
        audio_hash = hashlib.sha256(audio_bytes).hexdigest()
        if audio_hash != st.session_state.last_processed_audio_hash:
            st.session_state.last_processed_audio_hash = audio_hash
            st.session_state.voice_input_error = None
            try:
                with st.spinner("Transcribing your answer..."):
                    transcript = transcribe_audio(
                        api_key,
                        audio_bytes,
                        recorded_audio.type or "audio/wav"
                    )
                st.session_state[draft_key] = transcript
                st.rerun()
            except Exception as exc:
                st.session_state.voice_input_error = f"Could not transcribe the recording: {exc}"

    if st.session_state.voice_input_error:
        st.error(st.session_state.voice_input_error)
        if st.button("Retry voice transcription"):
            st.session_state.last_processed_audio_hash = None
            st.session_state.voice_input_error = None
            st.rerun()

    retry_clicked = False
    if st.session_state.interview_error:
        st.error(st.session_state.interview_error)
        retry_clicked = st.button("Retry sending your answer", type="primary")

    answer_text = st.text_area(
        "Your answer (you can edit the transcript)",
        key=draft_key,
        height=140,
        placeholder="Type your answer here, or record it above."
    )
    submit_clicked = st.button(
        "Submit answer",
        type="primary",
        disabled=not answer_text.strip() or bool(st.session_state.interview_error)
    )
    if submit_clicked:
        st.session_state.pending_answer = answer_text.strip()

    should_process = bool(st.session_state.pending_answer) and (
        submit_clicked or retry_clicked
    )
    if should_process:
        answer = st.session_state.pending_answer

        if st.session_state.turn_count < MAX_TURNS:
            next_q_num = st.session_state.turn_count + 1
            prompt_with_directive = (
                f"{answer}\n\n"
                f"[System Note: That was the candidate's response to Question "
                f"{st.session_state.turn_count}. Now ask Question {next_q_num} "
                f"of {MAX_TURNS}. Make it an adaptive follow-up based on their "
                "answer or probe a new relevant technical concept. Ask exactly one question.]"
            )
            try:
                with st.spinner(f"Reviewing your answer and preparing Question {next_q_num}..."):
                    chain = build_interview_chain(
                        api_key,
                        role_label,
                        f"{experience_label} years",
                        st.session_state.interview_focus
                    )
                    chain.invoke(
                        {"input": prompt_with_directive},
                        config={"configurable": {"session_id": "placement_session"}}
                    )
                msgs.messages[-2].content = answer
                if st.session_state.voice_output_enabled:
                    try:
                        st.session_state.interviewer_audio[len(msgs.messages) - 1] = generate_speech(
                            api_key,
                            extract_text(msgs.messages[-1].content)
                        )
                        st.session_state.voice_output_error = None
                    except Exception as exc:
                        st.session_state.voice_output_error = (
                            f"Could not create spoken interviewer audio: {exc}"
                        )
                st.session_state.turn_count = next_q_num
                st.session_state.pending_answer = None
                st.session_state.interview_error = None
                st.session_state.last_processed_audio_hash = None
            except Exception as exc:
                st.session_state.interview_error = f"Could not generate the next question: {exc}"
            st.rerun()
        else:
            msgs.add_user_message(answer)
            msgs.add_ai_message(
                "Thank you for completing all five questions. Your scorecard and model answers are being prepared."
            )
            st.session_state.pending_answer = None
            st.session_state.interview_active = False
            st.session_state.interview_completed = True
            st.rerun()


# =====================================================================
# PART F: SCORECARD & MODEL ANSWERS DASHBOARD
# =====================================================================
if st.session_state.interview_completed:
    st.divider()
    st.header("Evaluation Scorecard")

    retry_scorecard = False
    if st.session_state.scorecard_error:
        st.error(st.session_state.scorecard_error)
        retry_scorecard = st.button("Retry scorecard")

    if st.session_state.scorecard is None and not st.session_state.scorecard_error:
        retry_scorecard = True

    if retry_scorecard and st.session_state.scorecard is None:
        if not api_key:
            st.error("Enter your Gemini API key in the sidebar to generate the scorecard.")
        else:
            try:
                with st.spinner("Evaluating your interview and preparing model answers..."):
                    st.session_state.scorecard = generate_scorecard(
                        api_key,
                        role_label,
                        f"{experience_label} years"
                    )
                st.session_state.scorecard_error = None
            except Exception as exc:
                st.session_state.scorecard_error = f"Could not generate the scorecard: {exc}"
            st.rerun()

    if st.session_state.scorecard is not None:
        sc: EvaluationScorecard = st.session_state.scorecard
        avg_score = round(
            (sc.communication_score + sc.technical_depth_score + sc.problem_solving_score) / 3,
            1
        )
        metric_columns = st.columns(4)
        metric_columns[0].metric("Communication", f"{sc.communication_score} / 10")
        metric_columns[1].metric("Technical Depth", f"{sc.technical_depth_score} / 10")
        metric_columns[2].metric("Problem-Solving", f"{sc.problem_solving_score} / 10")
        metric_columns[3].metric("Overall Rating", f"{avg_score} / 10")

        st.subheader("Overall feedback")
        st.info(sc.overall_summary)

        strengths_col, improvements_col = st.columns(2)
        with strengths_col:
            st.markdown("#### Key strengths")
            for strength in sc.key_strengths:
                st.markdown(f"- {strength}")
        with improvements_col:
            st.markdown("#### Areas to improve")
            for improvement in sc.areas_for_improvement:
                st.markdown(f"- {improvement}")

        st.divider()
        st.subheader("Question-by-question review")
        for review in sc.question_reviews:
            with st.expander(
                f"Question {review.question_number}: {review.question_asked}",
                expanded=review.question_number == 1
            ):
                st.markdown(f"**Your answer:** {review.candidate_answer_summary}")
                st.markdown(f"**Feedback:** {review.critique}")
                st.markdown("**Model answer**")
                st.success(review.model_answer)