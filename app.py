import streamlit as st
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from collections import defaultdict
import time

# Rate limiting: 2 requests per IP
if "request_counts" not in st.session_state:
    st.session_state.request_counts = defaultdict(list)

def get_ip():
    try:
        ip = st.context.headers.get("X-Forwarded-For", "unknown")
        return ip.split(",")[0].strip()
    except:
        return "unknown"

def is_rate_limited():
    ip = get_ip()
    now = time.time()
    window = 24 * 60 * 60  # 24 hour window
    # Clean old requests
    st.session_state.request_counts[ip] = [
        t for t in st.session_state.request_counts[ip]
        if now - t < window
    ]
    if len(st.session_state.request_counts[ip]) >= 2:
        return True
    st.session_state.request_counts[ip].append(now)
    return False

st.set_page_config(page_title="Cover Letter Generator ✉️", page_icon="✉️")

st.title("✉️ Cover Letter Generator")
st.subheader("Upload your resume, paste a job description, get a human-sounding cover letter.")
st.markdown("---")

# --- INPUTS ---
col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("📄 Upload your resume (PDF)", type="pdf")

with col2:
    tone = st.selectbox(
        "🎯 Choose a tone",
        ["Professional", "Confident", "Conversational"],
        help="Professional = formal and polished. Confident = direct and assertive. Conversational = warm and natural."
    )

job_description = st.text_area(
    "📋 Paste the job description here",
    height=200,
    placeholder="Copy and paste the full job description from the job posting..."
)

generate = st.button("✉️ Generate Cover Letter", use_container_width=True)

# --- GENERATION ---
if generate:
    if not uploaded_file:
        st.error("Please upload your resume PDF first.")
        st.stop()
    if not job_description.strip():
        st.error("Please paste a job description first.")
        st.stop()
    if is_rate_limited():
        st.error("⚠️ You've used your 2 free generations for today. Come back tomorrow!")
        st.stop()

    with st.spinner("Writing your cover letter..."):
        # Extract resume text
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        resume_text = "\n".join([doc.page_content for doc in docs])

        if not resume_text.strip():
            st.error("Could not read text from this PDF. Is it a scanned image?")
            st.stop()

        # Connect to Groq
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=st.secrets["GROQ_API_KEY"]
        )

        prompt = PromptTemplate.from_template("""
You are a professional career coach and expert cover letter writer with 15 years 
of experience helping candidates land jobs at top companies.

Using ONLY the information in the resume and job description below, write a cover letter that:
- Opens with a specific, compelling hook — never start with "I am writing to apply" or "I am excited"
- In the first paragraph: mention one specific thing from the job description that genuinely interests you and why
- In the second paragraph: pick 2 real achievements or projects from the resume and connect them directly to the job requirements. Use specific details, not vague claims.
- In the third paragraph: show you understand the company's needs and end with a confident, direct call to action
- Use natural, conversational language — avoid words like: passionate, leverage, synergy, utilize, spearhead, dynamic, results-driven
- Sound like a thoughtful human wrote it, not an AI
- Keep it under 250 words total
- Do not include placeholders like [Company Name] — if you don't know it, write "your team" instead

Tone: {tone}

Resume:
<resume>
{resume_text}
</resume>

Job Description:
<job_description>
{job_description}
</job_description>

Write only the cover letter. No explanations, no notes, no preamble.
""")

        chain = prompt | llm | StrOutputParser()
        cover_letter = chain.invoke({
            "resume_text": resume_text,
            "job_description": job_description,
            "tone": tone
        })

    # --- OUTPUT ---
    st.markdown("---")
    st.subheader("✅ Your Cover Letter")
    st.write(cover_letter)

    st.markdown("---")

    # Download button
    st.download_button(
        label="⬇️ Download as .txt",
        data=cover_letter,
        file_name="cover_letter.txt",
        mime="text/plain",
        use_container_width=True
    )

    # Word count
    word_count = len(cover_letter.split())
    st.caption(f"Word count: {word_count}")
    if word_count > 250:
        st.warning("⚠️ Cover letter is over 250 words. Consider regenerating.")
print("✅ app.py written!")
