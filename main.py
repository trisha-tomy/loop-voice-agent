import os
import pandas as pd
import speech_recognition as sr
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
from gtts import gTTS
from pydub import AudioSegment
import io

# --- CONFIGURATION ---
# PASTE YOUR GEMINI API KEY BELOW
GOOGLE_API_KEY = "YOUR_GEMINI_API_KEY_HERE"
genai.configure(api_key=GOOGLE_API_KEY)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- STEP 1: LOAD DATA ---
print("Loading Hospital Data...")
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

# --- STEP 2: DEFINE TOOLS ---

def search_hospitals(location: str = None, keywords: str = None, **kwargs):
    print(f"Tool Triggered: Location={location}, Keywords={keywords}, Extras={kwargs}")
    
    if df.empty:
        return "Database not loaded."

    results = df.copy()

    # 1. Filter by Location (if provided)
    if location:
        loc_lower = location.lower().strip()
        if "bangalore" in loc_lower:
            loc_lower = "bengaluru"
        
        results = results[results['CITY'].str.lower().str.contains(loc_lower, na=False)]

    # 2. Filter by Keywords (SMARTER LOGIC)
    if keywords:
        key_lower = keywords.lower().strip()
        
        # Split keywords into separate words (e.g. "Manipal Sarjapur" -> ["manipal", "sarjapur"])
        search_terms = key_lower.split()
        
        # Filter results that contain ALL the terms
        for term in search_terms:
            results = results[results['Search_Text'].str.contains(term, na=False)]

    if results.empty:
        return "No hospitals found matching those details."
    
    top_results = results.head(5)
    output_text = "Here are the hospitals I found:\n"
    for _, row in top_results.iterrows():
        output_text += f"- Name: {row['HOSPITAL NAME']}\n  Address: {row['Address']}\n  City: {row['CITY']}\n"
    
    return output_text

tools_list = [search_hospitals]
model = genai.GenerativeModel(
    model_name='gemini-flash-latest',
    tools=tools_list,
    system_instruction="""
    You are Loop AI, a helpful voice assistant for a hospital network.
    1. If the user says 'Hello', introduce yourself as Loop AI.
    2. Use the 'search_hospitals' function to find real data.
    3. If the user asks about non-medical topics (like cooking, coding, weather), politely refuse:
       "I'm sorry, I can't help with that. I am forwarding this to a human agent." and stop.
    4. Keep your answers short and conversational.
    """
)
chat = model.start_chat(enable_automatic_function_calling=True)

# --- STEP 3: AUDIO PROCESSING (FIXED) ---

def speech_to_text(audio_bytes):
    try:
        # FIX: Load file with pydub (handles WebM/browser formats)
        # and export as a clean, standard WAV file
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
        audio.export("temp_rec.wav", format="wav")
        
        # Now read the clean WAV file
        r = sr.Recognizer()
        with sr.AudioFile("temp_rec.wav") as source:
            audio_data = r.record(source)
            text = r.recognize_google(audio_data)
            return text
    except Exception as e:
        print(f"Transcription Error: {e}")
        return ""

def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    mp3_fp = io.BytesIO()
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)
    return mp3_fp

# --- STEP 4: API ENDPOINT ---

@app.post("/chat-audio")
async def chat_audio(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    
    user_text = speech_to_text(audio_bytes)
    print(f"User Said: {user_text}")
    
    if not user_text:
        return StreamingResponse(
            text_to_speech("I didn't catch that. Could you please say it again?"),
            media_type="audio/mpeg"
        )

    try:
        response = chat.send_message(user_text)
        ai_text = response.text
    except Exception as e:
        print(f"Gemini Error: {e}")
        ai_text = "I'm having trouble connecting to my brain right now."

    print(f"AI Response: {ai_text}")
    
    return StreamingResponse(text_to_speech(ai_text), media_type="audio/mpeg")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)