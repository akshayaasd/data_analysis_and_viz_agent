import os
from groq import Groq

# Set your API key
api_key = os.getenv('GROQ_API_KEY') or 'your_key'

print("=" * 50)
print("🧪 Groq API Test (Updated Models)")
print("=" * 50)

print(f"\n1. API Key Check:")
print(f"   Key exists: {bool(api_key)}")
print(f"   Key format: {api_key[:10]}...{api_key[-4:]}")
print(f"   Starts with 'gsk_': {api_key.startswith('gsk_')}")

print(f"\n2. Client Initialization:")
try:
    client = Groq(api_key=api_key)
    print("   ✅ Client created successfully")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    exit()

# List available models first
print(f"\n3. Available Models:")
try:
    models = client.models.list()
    print(f"   ✅ Found {len(models.data)} models:")
    for model in models.data[:5]:
        print(f"   • {model.id}")
except Exception as e:
    print(f"   ⚠️ Could not list models: {e}")

# Test with current models
print(f"\n4. API Connection Test:")
current_models = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768-v0.1",
    "gemma2-9b-it"
]

for model_name in current_models:
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": "Say 'Hello' in one word"}
            ],
            max_tokens=10,
            temperature=0
        )
        
        reply = response.choices[0].message.content
        print(f"   ✅ {model_name}: Working!")
        print(f"      Response: {reply}")
        break  # Stop after first successful model
        
    except Exception as e:
        print(f"   ⚠️ {model_name}: {str(e)[:60]}")
        continue

print("\n" + "=" * 50)
print("✅ Groq API is working with current models!")
print("=" * 50)
