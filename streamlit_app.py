import streamlit as st
import os
import asyncio
from uuid import uuid4
from app.headhunter_core import langgraph_app
import sys
import asyncio

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

UPLOAD_FOLDER = os.path.join("app", "uploads")
SCREENSHOT_FOLDER = os.path.join("app", "screenshots")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

st.set_page_config(page_title="Resume Analyzer", layout="wide")
st.title("🔍 Resume Screening Tool")

# --- Upload form ---
with st.form("job_form"):
    job_title = st.text_input("Enter the job title")
    job_pdf = st.file_uploader("Upload the job description PDF", type=["pdf"])
    submitted = st.form_submit_button("Analyze")

if submitted:
    if not job_title or not job_pdf:
        st.error("Please provide both the job title and a PDF file.")
    else:
        # Save uploaded PDF
        pdf_filename = f"{uuid4().hex}_{job_pdf.name}"
        pdf_path = os.path.join(UPLOAD_FOLDER, pdf_filename)
        with open(pdf_path, "wb") as f:
            f.write(job_pdf.read())

        # Create input state
        initial_state = {
            "original_description": job_title,
            "job_title": job_title,
            "job_description_pdf": pdf_path,
        }

        # Run async processing
        with st.spinner("Analyzing job description..."):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            final_state = loop.run_until_complete(langgraph_app.ainvoke(initial_state))

        # Display results
        st.subheader("📄 Job Description")
        st.write(final_state.get("job_description", "N/A"))

        st.subheader("✅ Key Requirements")
        for req in final_state.get("requirements", []):
            st.markdown(f"- {req}")

        st.subheader("🔎 Search Query")
        st.code(final_state.get("search_query", ""), language="bash")

        st.subheader("👥 Filtered Candidates")
        candidates = final_state.get("filtered_candidates", [])
        if candidates:
            for idx, candidate in enumerate(candidates, 1):
                st.markdown(f"**Candidate {idx}**")
                st.json(candidate)
        else:
            st.info("No candidates found.")
