import os
import pandas as pd
import speech_recognition as sr
from fastapi import FastAPI, UploadFile, File, Request, Response
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
from gtts import gTTS
from pydub import AudioSegment
import io

# --- CONFIGURATION ---
# ⚠️ PASTE YOUR NEW API KEY HERE
GOOGLE_API_KEY = "YOUR_GEMINI_API_KEY_HERE"
genai.configure(api_key=GOOGLE_API_KEY)

app = FastAPI()

if not os.path.exists("static"):
    os.makedirs("static")
app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- LOAD DATA ---
try:
    df = pd.read_csv("List of GIPSA Hospitals - Sheet1.csv")
    df.columns = [c.strip() for c in df.columns]
    df['Search_Text'] = (df['HOSPITAL NAME'].fillna('') + " " + 
                         df['Address'].fillna('') + " " + 
                         df['CITY'].fillna('')).str.lower()
    print(f"Data Loaded! {len(df)} hospitals ready.")
except Exception as e:
    print(f"Error loading CSV: {e}")
    df = pd.DataFrame() 

# --- SEARCH TOOL ---
def search_hospitals(location: str = None, keywords: str = None, **kwargs):
    print(f"Tool Triggered: Location={location}, Keywords={keywords}")
    if df.empty: return "Database not loaded."
    results = df.copy()

    if location:
        loc_lower = location.lower().strip()
        if "bangalore" in loc_lower or "bengal" in loc_lower: 
            loc_lower = "bengaluru"
        results = results[results['CITY'].str.lower().str.contains(loc_lower, na=False)]

    if keywords:
        terms = keywords.lower().split()
        for term in terms:
            results = results[results['Search_Text'].str.contains(term, na=False)]

    if results.empty: return "No hospitals found."
    
    top_results = results.head(5)
    output = "Here are the hospitals I found:\n"
    for _, row in top_results.iterrows():
        output += f"- Name: {row['HOSPITAL NAME']}, City: {row['CITY']}\n"
    return output

# --- GEMINI SETUP ---
tools_list = [search_hospitals]
model = genai.GenerativeModel(
    model_name='gemini-flash-latest', 
    tools=tools_list,
    system_instruction="""
    You are Loop AI.
    1. If user says Hello, introduce yourself.
    2. Use 'search_hospitals' tool.
    3. SPEECH FIX: "Bangla/Bengal" -> Bangalore. "Money" -> Manipal. "Sir" -> Sarjapur.
    4. Keep answers VERY short (1 sentence).
    """
)
chat = model.start_chat(enable_automatic_function_calling=True)

# --- WEB HELPERS ---
def speech_to_text(audio_bytes):
    try:
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
        audio.export("temp_rec.wav", format="wav")
        r = sr.Recognizer()
        with sr.AudioFile("temp_rec.wav") as source:
            return r.recognize_google(r.record(source))
    except: return ""

def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    mp3_fp = io.BytesIO()
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)
    return mp3_fp

@app.post("/chat-audio")
async def chat_audio(file: UploadFile = File(...)):
    user_text = speech_to_text(await file.read())
    if not user_text: return StreamingResponse(text_to_speech("I didn't hear you."), media_type="audio/mpeg")
    try:
        response = chat.send_message(user_text)
        ai_text = response.text
    except: ai_text = "Connection error."
    return StreamingResponse(text_to_speech(ai_text), media_type="audio/mpeg")

# --- TWILIO ENDPOINTS (FIXED EMPTY TEXT BUG) ---

@app.post("/voice")
async def twilio_start(request: Request):
    # Using Standard Aditi voice (Safer than Neural for trials)
    xml_content = """
    <Response>
        <Say voice="Polly.Aditi">Hello! I am Loop AI. How can I help you?</Say>
        <Gather input="speech" action="/twilio-process" speechTimeout="auto" language="en-IN"/>
    </Response>
    """.strip()
    return Response(content=xml_content, media_type="application/xml")

@app.post("/twilio-process")
async def twilio_process(request: Request):
    try:
        form_data = await request.form()
        user_text = form_data.get("SpeechResult", "")
        print(f"Phone User said: {user_text}")

        if not user_text:
            return Response(content="<Response><Say>I didn't hear anything.</Say><Redirect>/voice</Redirect></Response>", media_type="application/xml")

        try:
            response = chat.send_message(user_text)
            ai_text = response.text
        except Exception as e:
            print(f"❌ GEMINI ERROR: {e}") 
            ai_text = "I am having trouble connecting."

        # ⚠️ CRITICAL FIX: Handle Empty Responses
        if not ai_text or not ai_text.strip():
            ai_text = "I heard you, but I don't have an answer for that."
            print("⚠️ WARNING: Gemini returned empty text. Using fallback.")

        clean_text = ai_text.replace("&", "and").replace("<", "").replace(">", "").replace('"', "'")
        
        xml_content = f"""
        <Response>
            <Say voice="Polly.Aditi">{clean_text}</Say>
            <Gather input="speech" action="/twilio-process" speechTimeout="auto" language="en-IN"/>
        </Response>
        """.strip()
        return Response(content=xml_content, media_type="application/xml")

    except Exception as e:
        print(f"🔥 SERVER CRASH: {e}")
        return Response(content="<Response><Say>System error.</Say></Response>", media_type="application/xml")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)