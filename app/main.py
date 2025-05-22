from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify
import os
import asyncio
from uuid import uuid4
from app.headhunter_core import langgraph_app

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join('app', 'uploads')
SCREENSHOT_FOLDER = os.path.join('app', 'screenshots')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    job_title = request.form.get('job_title')
    job_pdf = request.files.get('job_pdf')

    if not job_title or not job_pdf:
        return "Missing job title or file", 400

    pdf_filename = f"{uuid4().hex}_{job_pdf.filename}"
    pdf_path = os.path.join(UPLOAD_FOLDER, pdf_filename)
    job_pdf.save(pdf_path)

    initial_state = {
        "original_description": job_title,
        "job_title": job_title,
        "job_description_pdf": pdf_path,
    }

    final_state = asyncio.run(langgraph_app.ainvoke(initial_state))

    return render_template(
        'result.html',
        job_description=final_state.get("job_description"),
        requirements=final_state.get("requirements", []),
        search_query=final_state.get("search_query"),
        raw_profiles=final_state.get("raw_profiles", []),
        candidates=final_state.get("filtered_candidates", [])
    )

@app.route('/screenshot/<path:filename>')
def screenshot(filename):
    return send_file(os.path.join(SCREENSHOT_FOLDER, filename), mimetype='image/png')

@app.route('/api/results', methods=['POST'])
def api_results():
    data = request.json
    job_title = data.get("job_title")
    job_pdf_path = data.get("job_pdf_path")

    if not job_title or not job_pdf_path:
        return jsonify({"error": "Missing job_title or job_pdf_path"}), 400

    initial_state = {
        "original_description": job_title,
        "job_title": job_title,
        "job_description_pdf": job_pdf_path,
    }

    final_state = asyncio.run(langgraph_app.ainvoke(initial_state))
    return jsonify(final_state)

if __name__ == '__main__':
    app.run(debug=True)
