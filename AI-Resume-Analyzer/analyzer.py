import re
import pdfplumber
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Load a lightweight, high-performance semantic embedding model
@st_cache_resource if "st_cache_resource" in globals() else lambda f: f
def load_embedder():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

COMMON_SKILLS = {
    # Data Science & Machine Learning
    "python", "r", "sql", "pandas", "numpy", "scikit-learn", "tensorflow", 
    "pytorch", "keras", "machine learning", "deep learning", "nlp", 
    "computer vision", "transformers", "xgboost", "data analysis",
    # Big Data & Cloud
    "aws", "azure", "gcp", "docker", "kubernetes", "spark", "hadoop", "airflow",
    # Visualization & BI
    "tableau", "power bi", "plotly", "matplotlib", "seaborn", "streamlit",
    # Software Engineering & Tools
    "git", "github", "linux", "rest api", "fastapi", "flask", "ci/cd"
}

def extract_text_from_pdf(pdf_file) -> str:
    """Extract text cleanly from multi-page PDFs using pdfplumber."""
    extracted_text = []
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                extracted_text.append(text)
    return "\n".join(extracted_text)

def clean_text(text: str) -> str:
    """Normalize whitespace and remove non-standard characters."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def extract_skills(text: str) -> set:
    """Extract matching technical skills from pre-defined domain vocabulary."""
    text_lower = text.lower()
    found_skills = set()
    for skill in COMMON_SKILLS:
        # Match standalone words/phrases
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, text_lower):
            found_skills.add(skill.title())
    return found_skills

def calculate_semantic_similarity(resume_text: str, jd_text: str) -> float:
    """Compute cosine similarity between dense embeddings of Resume and JD."""
    embeddings = embedder.encode([resume_text, jd_text])
    score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return float(np.clip(score * 100, 0, 100))

def run_gap_analysis(resume_text: str, jd_text: str):
    """Compute overall score, matched skills, and missing recommendations."""
    resume_skills = extract_skills(resume_text)
    jd_skills = extract_skills(jd_text)

    matched_skills = sorted(list(resume_skills.intersection(jd_skills)))
    missing_skills = sorted(list(jd_skills - resume_skills))
    additional_skills = sorted(list(resume_skills - jd_skills))

    semantic_score = calculate_semantic_similarity(resume_text, jd_text)
    
    # Keyword coverage ratio (avoid division by zero)
    keyword_score = (len(matched_skills) / len(jd_skills) * 100) if jd_skills else semantic_score

    # Balanced Overall ATS Compatibility Score
    overall_score = round(0.6 * semantic_score + 0.4 * keyword_score, 1)

    return {
        "overall_score": overall_score,
        "semantic_score": round(semantic_score, 1),
        "keyword_score": round(keyword_score, 1),
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "additional_skills": additional_skills,
    }