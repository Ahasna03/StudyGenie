import streamlit as st
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import JsonOutputParser
import PyPDF2
import os
from datetime import datetime
import io
import hashlib
import json

# Custom CSS for styling
st.markdown("""
    <style>
    .main { background-color: #f5f5f5; padding: 20px; }
    .stButton>button {
        background-color: #01796f; color: white; border-radius: 5px; padding: 10px 20px;
        border: 2px solid #01796f; margin: 5px 0; width: 100%;
    }
    .stButton>button:hover {
        background-color: #01796f;
    }
    .stRadio label { font-size: 16px; margin-bottom: 10px; }
    .question { font-weight: bold; font-size: 18px; margin-top: 20px; color: #333; }
    .section-header { font-size: 24px; color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 5px; }
    .sidebar .stTextInput>label { font-size: 14px; color: #555; }
    .sidebar .stRadio>label { font-size: 16px; font-weight: bold; }
    .stTabs [data-baseweb="tab"] { font-size: 18px; }
    .chat-message { padding: 10px; border-radius: 5px; margin: 5px 0; }
    .user-message { background-color: #e1f5fe; }
    .bot-message { background-color: #f0f0f0; }
    </style>
""", unsafe_allow_html=True)

# Initialize Google Gemini API
def initialize_llm():
    api_key = st.secrets.get("GOOGLE_API_KEY") if hasattr(st, "secrets") else None
    if not api_key:
        api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        api_key = st.session_state.get("api_key", "")
        api_key = st.sidebar.text_input(
            "Google API Key",
            type="password",
            value=api_key,
            placeholder="Enter your API key",
            help="Obtain from Google Cloud Console. Store in .streamlit/secrets.toml or .env."
        )
        if api_key:
            st.session_state["api_key"] = api_key
    if not api_key:
        st.sidebar.error(
            "Please provide a Google API key in the sidebar, or set it in:\n"
            "- .streamlit/secrets.toml as `GOOGLE_API_KEY=\"your_key\"`\n"
            "- Environment variable `GOOGLE_API_KEY`"
        )
        return None
    try:
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
    except Exception as e:
        st.error(f"Error initializing LLM: {e}")
        return None

# Extract text from uploaded file
def extract_text(uploaded_file):
    max_size_mb = 10
    if uploaded_file.size > max_size_mb * 1024 * 1024:
        raise ValueError(f"File size exceeds {max_size_mb}MB limit.")
    if uploaded_file.name.endswith(".pdf"):
        reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        text = text.strip()
        if not text:
            raise ValueError("No text extracted from PDF.")
    else:
        text = uploaded_file.read().decode("utf-8").strip()
    if len(text) < 50:
        raise ValueError("Notes are too short to generate a quiz or chat.")
    return text

# Generate quiz from notes
def generate_quiz(llm, notes, num_questions=3):
    prompt_template = PromptTemplate(
        input_variables=["notes", "num_questions"],
        template="""
        Based on the following notes, generate {num_questions} multiple-choice questions in JSON format. Each question should have a question text, 4 answer options, and the correct answer index (0-3).
        Notes: {notes}
        Output format:
        [
            {{
                "question": "Question text",
                "options": ["Option 1", "Option 2", "Option 3", "Option 4"],
                "correct_answer": 0
            }},
            ...
        ]
        """
    )
    chain = prompt_template | llm | JsonOutputParser()
    try:
        quiz = chain.invoke({"notes": notes, "num_questions": num_questions})
        if not validate_quiz(quiz, num_questions):
            st.error("Invalid quiz format returned by LLM. Please try again.")
            return []
        return quiz
    except Exception as e:
        st.error(f"Error generating quiz: {e}")
        return []

# Validate quiz format
def validate_quiz(quiz, num_questions):
    if not isinstance(quiz, list) or len(quiz) != num_questions:
        return False
    for q in quiz:
        if not all(key in q for key in ["question", "options", "correct_answer"]):
            return False
        if not isinstance(q["options"], list) or len(q["options"]) != 4:
            return False
        if not isinstance(q["correct_answer"], int) or q["correct_answer"] not in range(4):
            return False
    return True

# Chat bot response
def get_chat_response(llm, notes, question, chat_history):
    prompt_template = PromptTemplate(
        input_variables=["notes", "question", "chat_history"],
        template="""
        You are a study assistant for the module with the following notes:
        {notes}

        The user asked: {question}

        Previous conversation:
        {chat_history}

        Provide a concise, accurate answer based on the notes. If the question is unrelated, politely say so and offer to answer a relevant question. Keep the tone friendly and educational.
        """
    )
    chain = prompt_template | llm
    try:
        response = chain.invoke({
            "notes": notes,
            "question": question,
            "chat_history": "\n".join([f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}" for msg in chat_history])
        })
        return response.content.strip()
    except Exception as e:
        st.error(f"Error generating chat response: {e}")
        return "Sorry, I couldn't process your question. Please try again."

# Initialize SQLite database
def init_db():
    conn = sqlite3.connect("quiz_performance.db")
    c = conn.cursor()
    # Table for modules
    c.execute("""
        CREATE TABLE IF NOT EXISTS modules (
            module_id INTEGER PRIMARY KEY AUTOINCREMENT,
            module_name TEXT UNIQUE
        )
    """)
    # Table for quiz performance
    c.execute("""
        CREATE TABLE IF NOT EXISTS performance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            module_id INTEGER,
            timestamp TEXT,
            score INTEGER,
            total_questions INTEGER,
            FOREIGN KEY (module_id) REFERENCES modules (module_id)
        )
    """)
    # Table for quizzes
    c.execute("""
        CREATE TABLE IF NOT EXISTS quizzes (
            quiz_id INTEGER PRIMARY KEY AUTOINCREMENT,
            module_id INTEGER,
            timestamp TEXT,
            notes_hash TEXT,
            FOREIGN KEY (module_id) REFERENCES modules (module_id)
        )
    """)
    # Table for questions
    c.execute("""
        CREATE TABLE IF NOT EXISTS questions (
            question_id INTEGER PRIMARY KEY AUTOINCREMENT,
            quiz_id INTEGER,
            question_text TEXT,
            options TEXT,
            correct_answer INTEGER,
            FOREIGN KEY (quiz_id) REFERENCES quizzes (quiz_id)
        )
    """)
    # Table for user answers
    c.execute("""
        CREATE TABLE IF NOT EXISTS user_answers (
            answer_id INTEGER PRIMARY KEY AUTOINCREMENT,
            quiz_id INTEGER,
            question_id INTEGER,
            user_answer TEXT,
            is_correct INTEGER,
            FOREIGN KEY (quiz_id) REFERENCES quizzes (quiz_id),
            FOREIGN KEY (question_id) REFERENCES questions (question_id)
        )
    """)
    conn.commit()
    return conn

# Manage modules (add/delete)
def manage_modules(conn):
    c = conn.cursor()
    st.markdown("<div class='section-header'>Manage Modules</div>", unsafe_allow_html=True)
    new_module = st.text_input("Add a new module:", placeholder="e.g., Mathematics, Biology")
    if st.button("Add Module"):
        if new_module.strip():
            try:
                c.execute("INSERT INTO modules (module_name) VALUES (?)", (new_module.strip(),))
                conn.commit()
                st.success(f"Module '{new_module}' added!")
            except sqlite3.IntegrityError:
                st.error("Module already exists.")
        else:
            st.error("Module name cannot be empty.")

    # Display and delete modules
    c.execute("SELECT module_id, module_name FROM modules")
    modules = c.fetchall()
    if modules:
        st.markdown("### Existing Modules")
        for module_id, module_name in modules:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(module_name)
            with col2:
                if st.button("Delete", key=f"delete_{module_id}"):
                    c.execute("DELETE FROM modules WHERE module_id = ?", (module_id,))
                    c.execute("DELETE FROM quizzes WHERE module_id = ?", (module_id,))
                    c.execute("DELETE FROM performance WHERE module_id = ?", (module_id,))
                    conn.commit()
                    st.success(f"Module '{module_name}' deleted.")
                    st.experimental_rerun()

# Save quiz and answers
def save_quiz_and_answers(conn, quiz, answers, notes, module_id):
    c = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    notes_hash = hashlib.sha256(notes.encode()).hexdigest()
    c.execute(
        "INSERT INTO quizzes (module_id, timestamp, notes_hash) VALUES (?, ?, ?)",
        (module_id, timestamp, notes_hash)
    )
    quiz_id = c.lastrowid
    score = 0
    for i, q in enumerate(quiz):
        options_json = json.dumps(q["options"])
        c.execute(
            "INSERT INTO questions (quiz_id, question_text, options, correct_answer) VALUES (?, ?, ?, ?)",
            (quiz_id, q["question"], options_json, q["correct_answer"])
        )
        question_id = c.lastrowid
        user_answer = answers[i]
        is_correct = int(q["options"].index(user_answer) == q["correct_answer"])
        score += is_correct
        c.execute(
            "INSERT INTO user_answers (quiz_id, question_id, user_answer, is_correct) VALUES (?, ?, ?, ?)",
            (quiz_id, question_id, user_answer, is_correct)
        )
    c.execute(
        "INSERT INTO performance (module_id, timestamp, score, total_questions) VALUES (?, ?, ?, ?)",
        (module_id, timestamp, score, len(quiz))
    )
    conn.commit()
    return score, len(quiz)

# Fetch performance data
def fetch_performance(conn, module_id):
    query = """
        SELECT timestamp, score, total_questions
        FROM performance
        WHERE module_id = ?
        ORDER BY timestamp
    """
    df = pd.read_sql_query(query, conn, params=(module_id,))
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["percentage"] = (df["score"] / df["total_questions"]) * 100
    return df

# Fetch past quizzes
def fetch_past_quizzes(conn, module_id):
    query = """
        SELECT q.quiz_id, q.timestamp, q.notes_hash,
               ques.question_id, ques.question_text, ques.options, ques.correct_answer,
               ua.user_answer, ua.is_correct
        FROM quizzes q
        LEFT JOIN questions ques ON q.quiz_id = ques.quiz_id
        LEFT JOIN user_answers ua ON ques.question_id = ua.question_id
        WHERE q.module_id = ?
        ORDER BY q.timestamp DESC, ques.question_id
    """
    df = pd.read_sql_query(query, conn, params=(module_id,))
    quizzes = []
    for quiz_id, group in df.groupby("quiz_id"):
        quiz = {
            "quiz_id": quiz_id,
            "timestamp": group["timestamp"].iloc[0],
            "notes_hash": group["notes_hash"].iloc[0],
            "questions": []
        }
        for _, row in group.iterrows():
            if pd.notna(row["question_id"]):
                options = json.loads(row["options"])
                quiz["questions"].append({
                    "question_id": row["question_id"],
                    "question_text": row["question_text"],
                    "options": options,
                    "correct_answer": options[int(row["correct_answer"])],
                    "user_answer": row["user_answer"],
                    "is_correct": bool(row["is_correct"])
                })
        quizzes.append(quiz)
    return quizzes

# Plot progress
def plot_progress(df):
    plt.figure(figsize=(10, 5))
    plt.plot(df["timestamp"], df["percentage"], marker="o", linestyle="-", color="#4CAF50")
    plt.title("Quiz Performance Over Time", fontsize=16)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Score (%)", fontsize=12)
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    return buf

# Main quiz page for a module with tabs
def module_quiz_page(llm, conn, module_id, module_name):
    st.title(f"🪄 StudyGenie - {module_name}")
    st.markdown(f"Your AI-powered study companion for {module_name}!", unsafe_allow_html=True)

    # Create tabs
    quiz_tab, chat_tab, progress_tab = st.tabs(["Quiz Generator", "Ask Questions?", "Progress"])

    # Quiz Generator Tab
    with quiz_tab:
        st.markdown("<div class='section-header'>Generate Quiz</div>", unsafe_allow_html=True)
        col1, col2 = st.columns([3, 1])
        with col1:
            uploaded_file = st.file_uploader(
                "Choose a TXT or PDF file",
                type=["txt", "pdf"],
                help="Upload your study notes to generate a quiz.",
                key=f"uploader_{module_id}"
            )
        with col2:
            num_questions = st.number_input(
                "Number of Questions", min_value=1, max_value=20, value=3, key=f"num_q_{module_id}"
            )
            generate_button = st.button("Generate Quiz", disabled=not (uploaded_file and llm), key=f"gen_{module_id}")

        if uploaded_file and llm:
            try:
                notes = extract_text(uploaded_file)
                st.session_state[f"notes_{module_id}"] = notes
                st.success("✅ Notes uploaded successfully!")
            except Exception as e:
                st.error(f"❌ Error processing file: {e}")
                notes = None

            if generate_button and notes:
                with st.spinner("Generating your quiz..."):
                    quiz = generate_quiz(llm, notes, num_questions)
                    if quiz:
                        st.session_state[f"quiz_{module_id}"] = quiz
                        st.session_state[f"answers_{module_id}"] = [None] * len(quiz)
                        st.session_state[f"submitted_{module_id}"] = False
                    else:
                        st.error("❌ Failed to generate quiz. Please try again.")

        if f"quiz_{module_id}" in st.session_state and st.session_state[f"quiz_{module_id}"]:
            st.markdown("<div class='section-header'>Your Quiz</div>", unsafe_allow_html=True)
            for i, q in enumerate(st.session_state[f"quiz_{module_id}"]):
                st.markdown(f"<div class='question'>Question {i+1}: {q['question']}</div>", unsafe_allow_html=True)
                answer = st.radio(
                    "Select an answer:",
                    q["options"],
                    key=f"q_{module_id}_{i}",
                    index=None,
                    disabled=st.session_state.get(f"submitted_{module_id}", False)
                )
                st.session_state[f"answers_{module_id}"][i] = answer

            if not st.session_state.get(f"submitted_{module_id}", False):
                all_answered = all(answer is not None for answer in st.session_state[f"answers_{module_id}"])
                if st.button("Submit Quiz", key=f"submit_{module_id}", disabled=not all_answered):
                    score, total_questions = save_quiz_and_answers(
                        conn, st.session_state[f"quiz_{module_id}"], st.session_state[f"answers_{module_id}"],
                        st.session_state[f"notes_{module_id}"], module_id
                    )
                    st.session_state[f"submitted_{module_id}"] = True
                    st.success(f"🎉 Your score: {score}/{total_questions}")

            if st.session_state.get(f"submitted_{module_id}", False):
                st.markdown("<div class='section-header'>Quiz Results</div>", unsafe_allow_html=True)
                for i, q in enumerate(st.session_state[f"quiz_{module_id}"]):
                    is_correct = q["options"].index(st.session_state[f"answers_{module_id}"][i]) == q["correct_answer"]
                    status = "✅ Correct" if is_correct else "❌ Incorrect"
                    st.markdown(f"<div class='question'>Question {i+1}: {q['question']}</div>", unsafe_allow_html=True)
                    st.write(f"{status}")
                    st.write(f"**Correct Answer**: {q['options'][q['correct_answer']]}")
                    st.write(f"**Your Answer**: {st.session_state[f'answers_{module_id}'][i]}")
                    st.markdown("---")

    # Chat Bot Tab
    with chat_tab:
        st.markdown("<div class='section-header'>Chat with StudyGenie</div>", unsafe_allow_html=True)
        if f"notes_{module_id}" not in st.session_state:
            st.info("ℹ️ Please upload notes in the Quiz Generator tab to enable the chat bot.")
        elif not llm:
            st.error("❌ Please provide a valid Google API key to use the chat bot.")
        else:
            # Initialize chat history
            if f"chat_history_{module_id}" not in st.session_state:
                st.session_state[f"chat_history_{module_id}"] = []

            # Display chat history
            for message in st.session_state[f"chat_history_{module_id}"]:
                with st.container():
                    if message["role"] == "user":
                        st.markdown(f"<div class='chat-message user-message'>You: {message['content']}</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div class='chat-message bot-message'>StudyGenie: {message['content']}</div>", unsafe_allow_html=True)

            # User input
            user_question = st.text_input("Ask a question:", key=f"chat_input_{module_id}")
            if st.button("Send", key=f"chat_send_{module_id}"):
                if user_question.strip():
                    st.session_state[f"chat_history_{module_id}"].append({"role": "user", "content": user_question})
                    with st.spinner("Thinking..."):
                        response = get_chat_response(
                            llm, st.session_state[f"notes_{module_id}"], user_question,
                            st.session_state[f"chat_history_{module_id}"]
                        )
                        st.session_state[f"chat_history_{module_id}"].append({"role": "assistant", "content": response})
                    st.experimental_rerun()
                else:
                    st.error("Please enter a question.")

    # Progress Tab
    with progress_tab:
        st.markdown("<div class='section-header'>Performance Over Time</div>", unsafe_allow_html=True)
        df = fetch_performance(conn, module_id)
        if not df.empty:
            buf = plot_progress(df)
            st.pyplot(plt)
            st.download_button(
                label="Download Progress Chart",
                data=buf,
                file_name=f"{module_name}_progress_chart.png",
                mime="image/png"
            )
        else:
            st.info("ℹ️ No quiz attempts yet. Take a quiz to see your progress!")

        st.markdown("<div class='section-header'>Past Quizzes</div>", unsafe_allow_html=True)
        quizzes = fetch_past_quizzes(conn, module_id)
        if quizzes:
            for quiz in quizzes:
                with st.expander(f"Quiz from {quiz['timestamp']}"):
                    for q in quiz["questions"]:
                        is_correct = "✅ Correct" if q["is_correct"] else "❌ Incorrect"
                        st.markdown(f"<div class='question'>{q['question_text']}</div>", unsafe_allow_html=True)
                        st.write(f"{is_correct}")
                        st.write(f"**Correct Answer**: {q['correct_answer']}")
                        st.write(f"**Your Answer**: {q['user_answer']}")
                        st.markdown("---")
        else:
            st.info("ℹ️ No past quizzes found. Complete a quiz to save it here.")

# Main Streamlit app
def main():
    # Initialize session state
    if "page" not in st.session_state:
        st.session_state["page"] = "manage_modules"
    if "module_id" not in st.session_state:
        st.session_state["module_id"] = None

    # Sidebar for navigation
    st.sidebar.header("🪄 StudyGenie")
    st.sidebar.markdown("Navigate your study modules below.")

    # Initialize components
    llm = initialize_llm()
    conn = init_db()

    # Fetch modules
    c = conn.cursor()
    c.execute("SELECT module_id, module_name FROM modules")
    modules = c.fetchall()

    # Navigation buttons
    if st.sidebar.button("Manage Modules", key="nav_manage"):
        st.session_state["page"] = "manage_modules"
        st.session_state["module_id"] = None
    for module_id, module_name in modules:
        if st.sidebar.button(f"{module_name}", key=f"nav_{module_id}"):
            st.session_state["page"] = "module"
            st.session_state["module_id"] = module_id

    with st.sidebar.expander("How to Use"):
        st.markdown("""
        1. Enter your Google API key (if not set).
        2. Go to 'Manage Modules' to add or delete study modules.
        3. Select a module to access its Quiz Generator, Chat Bot, and Progress tabs.
        4. Upload notes in the Quiz Generator tab to create quizzes or enable the chat bot.
        5. Use the Chat Bot tab to ask questions about your notes.
        6. Review performance and past quizzes in the Progress tab.
        """)

    # Render page based on state
    if st.session_state["page"] == "manage_modules":
        manage_modules(conn)
    elif st.session_state["page"] == "module" and st.session_state["module_id"]:
        module_id = st.session_state["module_id"]
        module_name = next((m[1] for m in modules if m[0] == module_id), "Unknown")
        module_quiz_page(llm, conn, module_id, module_name)
    else:
        st.warning("Please select a module or manage modules.")

    conn.close()

if __name__ == "__main__":
    main()

