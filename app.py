import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from analyzer import (
    extract_text_from_pdf,
    clean_text,
    run_gap_analysis,
)

st.set_page_config(
    page_title="TalentIQ | AI Resume & ATS Diagnostics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #2b5c8f;
    }
    .badge-match {
        display: inline-block;
        padding: 4px 10px;
        margin: 3px;
        border-radius: 12px;
        background-color: #e6f4ea;
        color: #137333;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-miss {
        display: inline-block;
        padding: 4px 10px;
        margin: 3px;
        border-radius: 12px;
        background-color: #fce8e6;
        color: #c5221f;
        font-weight: 600;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("⚡ TalentIQ — Next-Gen AI Resume & ATS Diagnostics")
st.caption("Dense vector embeddings · Multi-axis domain alignment · ATS structural diagnostics")

with st.sidebar:
    st.header("⚙️ Configuration")
    similarity_weight = st.slider("Semantic Weight", 0.1, 1.0, 0.6, 0.1, help="Weight given to deep conceptual sentence meaning vs exact keyword matching.")
    st.info("💡 **Pro-Tip:** A balance of 60% Semantic / 40% Keyword best mimics modern enterprise ATS engines like Workday and Greenhouse.")

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📄 Upload Candidate Resume")
    uploaded_file = st.file_uploader("Upload PDF Resume", type=["pdf"], label_visibility="collapsed")

with col2:
    st.subheader("🎯 Target Job Description")
    job_description = st.text_area(
        "Paste Job Description text",
        height=180,
        placeholder="Paste role requirements, technical expectations, and responsibilities...",
        label_visibility="collapsed"
    )

if st.button("🚀 Run Comprehensive Diagnostics", type="primary", use_container_width=True):
    if not uploaded_file:
        st.warning("Please upload a candidate PDF resume first.")
    elif not job_description.strip():
        st.warning("Please paste a target job description to match against.")
    else:
        with st.spinner("Analyzing semantic structure, extracting skills, and linting document..."):
            resume_raw = extract_text_from_pdf(uploaded_file)
            cleaned_resume = clean_text(resume_raw)
            cleaned_jd = clean_text(job_description)

            results = run_gap_analysis(cleaned_resume, cleaned_jd)

        st.divider()

        # Top Metric Row
        m1, m2, m3 = st.columns(3)
        m1.metric("Overall Fit Index", f"{results['overall_score']}%")
        m2.metric("Semantic Alignment", f"{results['semantic_score']}%")
        m3.metric("Hard Skill Coverage", f"{results['coverage_score']}%")

        # Visualizations (Gauge + Radar)
        v1, v2 = st.columns([1, 1], gap="medium")
        
        with v1:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=results["overall_score"],
                title={"text": "ATS Compatibility Score"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#1f77b4"},
                    "steps": [
                        {"range": [0, 50], "color": "#fce8e6"},
                        {"range": [50, 75], "color": "#fef7e0"},
                        {"range": [75, 100], "color": "#e6f4ea"}
                    ]
                }
            ))
            fig_gauge.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with v2:
            radar_df = results["radar_data"]
            fig_radar = px.line_polar(
                radar_df, 
                r="Match", 
                theta="Category", 
                line_close=True,
                title="Domain Proficiency Breakdown"
            )
            fig_radar.update_traces(fill="toself", fillcolor="rgba(31, 119, 180, 0.2)", line_color="#1f77b4")
            fig_radar.update_layout(height=320, margin=dict(l=40, r=40, t=40, b=20))
            st.plotly_chart(fig_radar, use_container_width=True)

        # Tabbed Diagnostics
        t1, t2, t3 = st.tabs(["🎯 Skill Match Breakdown", "🩺 ATS Health & Diagnostics", "✍️ AI Impact Rewrites"])

        with t1:
            c_left, c_right = st.columns(2)
            with c_left:
                st.markdown("#### ✅ Detected Relevant Competencies")
                if results["matched_skills"]:
                    chips = "".join([f"<span class='badge-match'>{s}</span>" for s in results["matched_skills"]])
                    st.markdown(chips, unsafe_allow_html=True)
                else:
                    st.info("No direct keyword competencies overlapped with the job description.")

            with c_right:
                st.markdown("#### ⚠️ High-Priority Missing Skills")
                if results["missing_skills"]:
                    chips = "".join([f"<span class='badge-miss'>{s}</span>" for s in results["missing_skills"]])
                    st.markdown(chips, unsafe_allow_html=True)
                else:
                    st.success("Target job competencies are fully addressed.")

        with t2:
            st.markdown("#### ATS Parseability & Content Health")
            for item in results["ats_checks"]:
                icon = "✅" if item["passed"] else "⚠️"
                st.markdown(f"**{icon} {item['check']}:** {item['detail']}")

        with t3:
            st.markdown("#### 💡 Quantifiable Bullet Point Suggestions (X-Y-Z Method)")
            st.caption("Incorporate missing target competencies with measurable impact:")
            if results["rewrites"]:
                for r in results["rewrites"]:
                    with st.expander(f"✨ Recommendation for **{r['skill']}**"):
                        st.markdown(f"**Framework:** {r['formula']}")
                        st.markdown(f"**Example Bullet Point:**\n> {r['example']}")
            else:
                st.info("No missing skills to generate suggestions for.")