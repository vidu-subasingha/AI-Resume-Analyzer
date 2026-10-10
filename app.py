import streamlit as st
import plotly.graph_objects as go
from analyzer import (
    extract_text_from_pdf,
    clean_text,
    run_gap_analysis,
)

st.set_page_config(
    page_title="AI Resume Analyzer & ATS Scorer",
    page_icon="📄",
    layout="wide",
)

st.title("📄 AI Resume Analyzer & Match Engine")
st.caption("Evaluate resume alignment with job requirements using Dense Sentence Embeddings and Skill Graph Analysis.")

col1, col2 = st.columns([1, 1], gap="medium")

with col1:
    st.subheader("1. Candidate Resume")
    uploaded_file = st.file_uploader("Upload Resume (PDF format)", type=["pdf"])

with col2:
    st.subheader("2. Target Job Description")
    job_description = st.text_area(
        "Paste Job Description text here:",
        height=200,
        placeholder="Paste requirements, responsibilities, and qualifications...",
    )

if st.button("Run Evaluation", type="primary", use_container_width=True):
    if not uploaded_file:
        st.warning("Please upload a PDF resume.")
    elif not job_description.strip():
        st.warning("Please provide a job description.")
    else:
        with st.spinner("Analyzing document structure and computing semantic embeddings..."):
            raw_resume_text = extract_text_from_pdf(uploaded_file)
            cleaned_resume = clean_text(raw_resume_text)
            cleaned_jd = clean_text(job_description)

            results = run_gap_analysis(cleaned_resume, cleaned_jd)

        st.divider()

        # Score Visualizations
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Overall Match Score", f"{results['overall_score']}%")
        m_col2.metric("Semantic Similarity", f"{results['semantic_score']}%")
        m_col3.metric("Skill Coverage", f"{results['keyword_score']}%")

        # Gauge Chart
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=results["overall_score"],
                title={"text": "Overall ATS Fit Index"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#1f77b4"},
                    "steps": [
                        {"range": [0, 45], "color": "#ffcccc"},
                        {"range": [45, 75], "color": "#fff3cd"},
                        {"range": [75, 100], "color": "#d4edda"},
                    ],
                },
            )
        )
        fig.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

        # Detailed Breakdown Tabs
        tab1, tab2 = st.tabs(["🎯 Skill Breakdown", "💡 Strategic Recommendations"])

        with tab1:
            col_match, col_miss = st.columns(2)
            with col_match:
                st.markdown("### ✅ Matched Competencies")
                if results["matched_skills"]:
                    for skill in results["matched_skills"]:
                        st.badge(skill) if hasattr(st, "badge") else st.markdown(f"- **{skill}**")
                else:
                    st.info("No explicit common technical skills detected.")

            with col_miss:
                st.markdown("### ⚠️ Missing Required Skills")
                if results["missing_skills"]:
                    for skill in results["missing_skills"]:
                        st.markdown(f"- 🔴 **{skill}**")
                else:
                    st.success("Great job! No major required skills are missing.")

        with tab2:
            st.markdown("### 📝 Actionable Fixes")
            if results["missing_skills"]:
                st.write(
                    f"Consider incorporating experiences with **{', '.join(results['missing_skills'][:4])}** "
                    "into your project or experience bullet points."
                )
            st.markdown(
                """
                - **Use Quantifiable Metrics:** Structure accomplishments using the **X-Y-Z formula** (*Accomplished [X], measured by [Y], by doing [Z]*).
                - **Header Standard:** Ensure common section headers (*Work Experience*, *Projects*, *Skills*, *Education*) are used so ATS parsers index properly.
                """
            )