import os
import gc
from flask import Flask, request, jsonify
from flask_cors import CORS

# Set environment variables for memory optimization
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TRANSFORMERS_CACHE'] = '/tmp/transformers_cache'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

app = Flask(__name__)
print("Flask loaded")

# Load environment variables
from dotenv import load_dotenv
load_dotenv()
CORS(app)

# Initialize Gemini client
from google import genai
from google.genai import types
client = genai.Client(api_key=os.getenv("GEMINI_KEY"))
print("Gemini loaded")

# Global variables for lazy loading
tokenizer = None
model = None
classifier = None

label_map = {
    "LABEL_0": "Course Registration",
    "LABEL_1": "Documents & Certificates", 
    "LABEL_2": "General Inquiry",
    "LABEL_3": "Payment & Fees",
    "LABEL_4": "Scheduling & Attendance"
}

def load_models():
    """Lazy load models only when needed"""
    global tokenizer, model, classifier
    
    if classifier is None:
        print("Loading AI models...")
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            from transformers.pipelines import pipeline
            
            model_path = "hshkoukani/bolt"
            
            # Load with memory optimizations
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                low_cpu_mem_usage=True
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                low_cpu_mem_usage=True,
                torch_dtype="auto"
            )
            classifier = pipeline(
                "text-classification", 
                model=model, 
                tokenizer=tokenizer,
                device=-1  # Force CPU usage to save memory
            )
            print("AI Model loaded successfully")
            
            # Force garbage collection after loading
            gc.collect()
            
        except Exception as e:
            print(f"Error loading models: {e}")
            raise e

def generate_response(subject, body, label):
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            config=types.GenerateContentConfig(
                system_instruction=(
                    "You are replying to emails that I receive.\n"
                    "You will be provided with the subject, body, and label of an incoming email.\n"
                    "\n"
                    "Instructions:\n"
                    "- You are the **recipient** of the original email. Write a reply accordingly.\n"
                    "- If sender and recipient names are provided, **flip their roles** in your reply.\n"
                    "- If either name is missing, **do not invent or use a placeholder like [Sender Name]**. Just leave the greeting out unless necessary.\n"
                    "- Strictly output only the body of the response. Do not include the subject, sender, recipient, greeting, or signature unless it's contextually appropriate within the reply body.\n"
                    "- Match your tone to the given label (e.g., Complaint, Request, etc.).\n"
                )
            ),
            contents=f"Subject: {subject}\nBody: {body}\nLabel: {label}"
        )
        return response.text
    except Exception as e:
        print(f"Error generating response: {e}")
        return f"Error generating response: {str(e)}"

@app.route('/analyze-label', methods=['POST']) 
def analyze_label():
    try:
        # Load models on first request
        load_models()
        
        data = request.get_json()
        
        if not data or 'body' not in data:
            return jsonify({"Error": "No text provided"}), 400
        
        text = data.get('subject', '') + ' ' + data.get('body', '')
        
        # Classify the text
        result = classifier(text)
        result[0]['label'] = label_map.get(result[0]['label'], result[0]['label'])
        
        # Generate response
        gemini_response = generate_response(data.get('subject', ''), data.get('body', ''), result[0]['label'])
        
        return jsonify({
            "result": result,
            "output": gemini_response
        })
    
    except Exception as e:
        print(f"Error in analyze_label: {e}")
        return jsonify({"Error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for Render"""
    return jsonify({"status": "healthy"}), 200

@app.route('/', methods=['GET'])
def root():
    """Root endpoint"""
    return jsonify({"message": "Bolt API is running"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))