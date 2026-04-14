from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key or api_key == "paste_your_groq_api_key_here":
    print("❌ ERROR: Please paste your real Groq API key in the .env file first!")
    print("   Get one free at: https://console.groq.com/keys")
    exit()

client = Groq(api_key=api_key)

print("⏳ Testing Groq + Llama 3.3 70B…")
response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role": "user", "content": "এই বাংলা টেক্সট থেকে ১টি প্রশ্ন-উত্তর তৈরি করো: বাংলাদেশের রাজধানী ঢাকা।"}],
    temperature=0.7,
    max_tokens=512,
)
print("✅ SUCCESS! Llama replied:")
print(response.choices[0].message.content)
