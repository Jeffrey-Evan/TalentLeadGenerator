import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
LINKEDIN_USER = os.getenv("LINKEDIN_USER")
LINKEDIN_PASS = os.getenv("LINKEDIN_PASS")

genai.configure(api_key=GEMINI_API_KEY)

import re
import urllib.parse
import asyncio
import random
from playwright.async_api import async_playwright

class LinkedInScraper:
    def __init__(self, username, password):
        self.username = username
        self.password = password
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None

    async def launch(self):
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=True)
        self.context = await self.browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36" # Fixed: Removed line breaks and extra spaces
        )
        self.page = await self.context.new_page()
        await self._apply_stealth()

    async def _apply_stealth(self):
        await self.page.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', { get: () => false });
            window.chrome = { runtime: {} };
            const originalQuery = window.navigator.permissions.query;
            window.navigator.permissions.query = (parameters) =>
                parameters.name === 'notifications'
                    ? Promise.resolve({ state: Notification.permission })
                    : originalQuery(parameters);
            Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
            Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3] });
        """)

    async def login(self):
        await self.page.goto("https://www.linkedin.com/login")
        await self.page.fill('input#username', self.username)
        await self.page.fill('input#password', self.password)
        await self.page.click('button[type="submit"]')
        await self.page.wait_for_url(re.compile(r'.*linkedin\.com/feed.*'))

    async def _simulate_human_scrolling(self):
        for _ in range(2):
            await self.page.mouse.wheel(0, 500)
            await asyncio.sleep(random.uniform(1.0, 2.0))

    async def search_profiles(self, query, max_results=20):
        encoded_query = urllib.parse.quote_plus(f'site:linkedin.com/in indonesia {query}')
        await self.page.goto(f"https://search.yahoo.com/search?p={encoded_query}")
        await self.page.wait_for_timeout(3000)
        await self._simulate_human_scrolling()
        html = await self.page.content()
        return self.parse_results(html, max_results)

    def parse_results(self, html, max_results):
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')
        cards = soup.select('a[href*="linkedin.com/in/"]')
        seen = set()
        results = []
        for link in cards:
            url = link.get('href')
            clean_url = urllib.parse.unquote(url)
            if clean_url not in seen:
                seen.add(clean_url)
                results.append({'name': 'Unknown', 'position': 'Unknown', 'url': clean_url})
            if len(results) >= max_results:
                break
        return results

    async def close(self):
        await self.context.close()
        await self.browser.close()
        await self.playwright.stop()

import re
import json
import requests
from PIL import Image
import base64

class ResumeParser:

    @staticmethod
    def extract_text(image_path):
        # Convert image to base64
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

        prompt = "Extract all readable text from the following resume image."

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
        headers = {"Content-Type": "application/json"}
        data = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {
                        "inlineData": {
                            "mimeType": "image/png",  # Change if your image is JPEG, etc.
                            "data": encoded_image
                        }
                    }
                ]
            }]
        }

        response = requests.post(url, headers=headers, json=data)
        if response.ok:
            try:
                return response.json()['candidates'][0]['content']['parts'][0]['text']
            except (KeyError, IndexError):
                print("Failed to extract text from Gemini response.")
                return ""
        else:
            print("Gemini API request failed:", response.text)
            return ""

    @staticmethod
    def call_gemini(prompt_text):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
        headers = {"Content-Type": "application/json"}
        data = {"contents": [{"parts": [{"text": prompt_text}]}]}
        resp = requests.post(url, headers=headers, json=data)
        if resp.ok:
            try:
                return resp.json()['candidates'][0]['content']['parts'][0]['text']
            except (KeyError, IndexError):
                return None
        return None

    @staticmethod
    def build_prompt(resume_text):
        return f"""
Extract JSON from this resume:

- name
- skills
- education
- certifications
- experience
- experience_years

Text:
\"\"\"
{resume_text}
\"\"\"
        """.strip()

    @staticmethod
    def parse_from_image(image_path):
        text = ResumeParser.extract_text(image_path)
        prompt = ResumeParser.build_prompt(text)
        response = ResumeParser.call_gemini(prompt)
        if response:
            response = re.sub(r'^```json|```$', '', response.strip(), flags=re.MULTILINE).strip()
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                print("Failed to parse JSON")
        return None

import re
from ratelimit import limits, sleep_and_retry
import google.generativeai as genai

class RateLimitedMatcher:
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-1.5-flash')

    @sleep_and_retry
    @limits(calls=10, period=60)
    def match(self, profile, requirements):
        prompt = (
            f"Requirements:\n{chr(10).join('- ' + r for r in requirements)}\n\n"
            f"Candidate:\nName: {profile.get('name')}\nPosition: {profile.get('position')}\n"
            f"Skills: {', '.join(profile.get('skills', []))}\nExperience: {profile.get('experience_years', 'N/A')} years\n\n"
            f"Rate the match (0–10) and explain."
        )
        result = self.model.generate_content(prompt)
        text = result.text.strip()
        match = re.search(r'\d+(\.\d+)?', text)
        score = float(match.group(0)) if match else 0.0
        return {"profile": profile, "score": round(score, 2), "explanation": text}

from typing import List, Dict, TypedDict

class State(TypedDict, total=False):
    original_description: str
    job_title: str
    job_description_pdf: str
    job_description: str
    requirements: List[str]
    raw_profiles: List[Dict]
    filtered_candidates: List[Dict]
    search_query: str

import os
from uuid import uuid4
import concurrent.futures
from PyPDF2 import PdfReader
import google.generativeai as genai

cache = {}

def job_description_agent(state):
    if state['job_description_pdf'] in cache:
        print("🔁 Using cached job description...")
        return {"job_description": cache[state['job_description_pdf']]['job_description']}

    print("📄 Reading and summarizing the job description from PDF...")
    reader = PdfReader(state['job_description_pdf'])
    text = "\n".join(p.extract_text() for p in reader.pages if p.extract_text())
    model = genai.GenerativeModel("gemini-1.5-flash")
    resp = model.generate_content(f"Summarize this job description:\n\n{text}")
    result = {"job_description": resp.text.strip()}
    cache[state['job_description_pdf']] = result
    return result

def requirements_agent(state):
    print("✅ Extracting key requirements from the job description...")
    model = genai.GenerativeModel('gemini-1.5-flash')
    resp = model.generate_content(f"List bullet-point requirements from:\n\n{state['job_description']}")
    lines = [line.strip('-• ') for line in resp.text.strip().split('\n') if line.strip()]
    return {"requirements": lines}

def headhunter_agent(state):
    print("🔍 Generating a search query for potential candidates based on the original job description...")
    return {"search_query": state["original_description"]}

async def scraper_agent(state):
    print("🌐 Launching LinkedIn scraper and logging in...")
    scraper = LinkedInScraper(LINKEDIN_USER, LINKEDIN_PASS)
    await scraper.launch()
    await scraper.login()

    print(f"🕵️ Searching LinkedIn for profiles matching: '{state['search_query']}'")
    search_results = await scraper.search_profiles(state["search_query"])
    enriched_profiles = []

    os.makedirs("screenshots", exist_ok=True)

    for result in search_results:
        try:
            print(f"📸 Capturing profile screenshot: {result['url']}")
            await scraper.page.goto(result["url"])
            await scraper.page.wait_for_timeout(3000)  # Let page load
            filename = f"screenshots/{uuid4().hex}.png"
            await scraper.page.screenshot(path=filename, full_page=True)
            result["screenshot_path"] = filename
            enriched_profiles.append(result)
        except Exception as e:
            print(f"⚠️ Failed to capture screenshot for {result['url']}: {e}")

    await scraper.close()
    print(f"✅ Finished scraping. {len(enriched_profiles)} profiles captured.")
    return {"raw_profiles": enriched_profiles}

def matching_agent(state):
    print("🧠 Matching candidates to job requirements...")
    matcher = RateLimitedMatcher()
    top_matches = []

    for profile in state["raw_profiles"]:
        try:
            parsed = ResumeParser.parse_from_image(profile["screenshot_path"])
            if not parsed:
                continue

            score_data = matcher.match(parsed, state["requirements"])
            parsed.update({
                "match_score": score_data["score"],
                "explanation": score_data["explanation"]
            })
            top_matches.append(parsed)
        except Exception as e:
            print(f"⚠️ Error parsing or matching: {e}")

    top_sorted = sorted(top_matches, key=lambda x: x["match_score"], reverse=True)[:5]
    print(f"🏁 Matching complete. Top {len(top_sorted)} candidates selected.")
    return {"filtered_candidates": top_sorted}

def print_results(state):
    print("\n=== 🧾 JOB REQUIREMENTS ===")
    for req in state['requirements']:
        print(f"- {req}")

    print("\n=== 👤 TOP MATCHED CANDIDATES ===")
    for c in state['filtered_candidates']:
        print(f"{c['name']} | {c['match_score']} ⭐")
        print(f"{c['explanation']}\n")
    return state

import asyncio
import nest_asyncio
from langgraph.graph import StateGraph, END

nest_asyncio.apply()

workflow = StateGraph(State)
workflow.add_node("job_desc", job_description_agent)
workflow.add_node("extract_reqs", requirements_agent)
workflow.add_node("headhunt", headhunter_agent)
workflow.add_node("scrape", scraper_agent)
workflow.add_node("match", matching_agent)
workflow.add_node("print_results", print_results)

workflow.set_entry_point("job_desc")
workflow.add_edge("job_desc", "extract_reqs")
workflow.add_edge("extract_reqs", "headhunt")
workflow.add_edge("headhunt", "scrape")
workflow.add_edge("scrape", "match")
workflow.add_edge("match", "print_results")
workflow.add_edge("print_results", END)

langgraph_app = workflow.compile()