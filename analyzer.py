import re
import pdfplumber
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Load dense sentence transformer
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Domain taxonomy categorized by discipline
CATEGORIZED_SKILLS = {
    "ML & Deep Learning": [
        "python", "scikit-learn", "tensorflow", "pytorch", "keras", 
        "xgboost", "nlp", "computer vision", "transformers", "deep learning", "machine learning"
    ],
    "Data & Cloud Infrastructure": [
        "sql", "aws", "azure", "gcp", "docker", "kubernetes", "spark", "airflow", "hadoop", "etl"
    ],
    "Analytics & Visualization": [
        "r", "pandas", "numpy", "tableau", "power bi", "plotly", "matplotlib", "seaborn", "data analysis", "statistics"
    ],
    "Software & Development": [
        "git", "github", "linux", "rest api", "fastapi", "flask", "ci/cd", "c++", "java"
    ],
}

ALL_SKILLS = set(skill for sublist in CATEGORIZED_SKILLS.values() for skill in sublist)

def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from multi-page PDFs cleanly."""
    extracted = []
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                extracted.append(t)
    return "\n".join(extracted)

def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def extract_skills_categorized(text: str):
    """Categorize matched skills across key domains."""
    text_lower = text.lower()
    profile = {}
    total_found = set()
    
    for category, skills in CATEGORIZED_SKILLS.items():
        found = set()
        for s in skills:
            if re.search(r"\b" + re.escape(s) + r"\b", text_lower):
                found.add(s.title())
                total_found.add(s.title())
        profile[category] = found

    return profile, total_found

def run_ats_linting(text: str):
    """Perform health checks on ATS parseability and resume quality."""
    checks = []
    
    # 1. Check for contact details
    has_email = bool(re.search(r"[\w\.-]+@[\w\.-]+\.\w+", text))
    has_phone = bool(re.search(r"(\+\d{1,3}[- ]?)?\(?\d{3}\)?[- ]?\d{3}[- ]?\d{4}", text))
    has_linkedin = "linkedin.com" in text.lower() or "github.com" in text.lower()
    
    checks.append({
        "check": "Contact & Links", 
        "passed": has_email and (has_phone or has_linkedin),
        "detail": "Email and LinkedIn/GitHub or phone detected." if has_email else "Missing email or profile link."
    })

    # 2. Measurable Metrics Check (numbers, percentages, currencies)
    metrics_count = len(re.findall(r"\b\d+%\b|\$\d+[\d,]*|\b\d+\s*(?:users|clients|projects|models|x)\b", text, re.IGNORECASE))
    checks.append({
        "check": "Quantifiable Impact",
        "passed": metrics_count >= 3,
        "detail": f"Found {metrics_count} quantifiable data points (metrics, %, or $). Aim for 3+ in bullet points."
    })

    # 3. Action Verb Health Check
    action_verbs = ["developed", "deployed", "built", "spearheaded", "engineered", "optimized", "orchestrated", "reduced", "increased"]
    found_verbs = [v for v in action_verbs if re.search(r"\b" + v + r"\b", text.lower())]
    checks.append({
        "check": "Strong Action Verbs",
        "passed": len(found_verbs) >= 3,
        "detail": f"Detected strong verbs: {', '.join(found_verbs[:4])}" if found_verbs else "Low frequency of strong action verbs."
    })

    return checks

def generate_rewrite_suggestions(missing_skills: list):
    """Generate high-impact bullet point templates using Google's XYZ formula."""
    templates = []
    for skill in missing_skills[:3]:
        templates.append({
            "skill": skill,
            "formula": f"**Accomplished [X] by doing [Y] resulting in [Z]**",
            "example": f"Designed and deployed a scalable pipeline using **{skill}**, reducing inference latency by 35% across 50,000+ daily requests."
        })
    return templates

def run_gap_analysis(resume_text: str, jd_text: str):
    resume_cat, resume_skills = extract_skills_categorized(resume_text)
    jd_cat, jd_skills = extract_skills_categorized(jd_text)

    # Cosine Similarity on dense embeddings
    embeddings = embedder.encode([resume_text, jd_text])
    semantic_sim = float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]) * 100
    semantic_sim = max(0.0, min(100.0, semantic_sim))

    # Matched vs missing skills
    matched = sorted(list(resume_skills.intersection(jd_skills)))
    missing = sorted(list(jd_skills - resume_skills))

    # Radar breakdown per domain
    radar_data = []
    for cat in CATEGORIZED_SKILLS.keys():
        r_count = len(resume_cat[cat])
        j_count = len(jd_cat[cat])
        score = (r_count / j_count * 100) if j_count > 0 else (100.0 if r_count > 0 else 50.0)
        radar_data.append({"Category": cat, "Match": min(score, 100.0)})

    coverage_score = (len(matched) / len(jd_skills) * 100) if jd_skills else semantic_sim
    overall_score = round(0.55 * semantic_sim + 0.45 * coverage_score, 1)

    ats_checks = run_ats_linting(resume_text)
    rewrites = generate_rewrite_suggestions(missing)

    return {
        "overall_score": overall_score,
        "semantic_score": round(semantic_sim, 1),
        "coverage_score": round(coverage_score, 1),
        "matched_skills": matched,
        "missing_skills": missing,
        "radar_data": pd.DataFrame(radar_data),
        "ats_checks": ats_checks,
        "rewrites": rewrites
    }