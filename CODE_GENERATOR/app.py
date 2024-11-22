from flask import Flask, render_template, request
from dotenv import load_dotenv
import os
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Configure Google Generative AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_gemini_response(question):
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(question)
    return response.text

def convert_code(code, target_language):
    model = genai.GenerativeModel('gemini-pro')
    prompt = f"Convert the following Python code to {target_language}:\n\n{code}\n"

    response = model.generate_content(prompt)
    return response.text

@app.route('/', methods=['GET', 'POST'])
def index():
    response = ""
    converted_codes = {}
    
    if request.method == 'POST':
        input_text = request.form.get('input')
        action = request.form.get('action')

        if action == 'ask':
            # Get the Python output from Gemini
            response = get_gemini_response(input_text)
            # Convert code to all desired languages dynamically
            languages = ["Java", "C", "C++", "JavaScript"]
            for lang in languages:
                converted_codes[lang.lower()] = convert_code(response, lang)

    return render_template('index.html', response=response, converted_codes=converted_codes)


if __name__ == '__main__':
    app.run(debug=True)
