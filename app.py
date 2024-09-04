import openai
from transformers import ViltProcessor, ViltForQuestionAnswering
from PIL import Image
import io
from flask import Flask, render_template, request


app = Flask(__name__)

# Set your OpenAI API key 
openai.api_key = "key here"


processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
vqa_model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

def get_image_features(image, question):
    """Extract image features using the ViLT model."""
    try:
       
        img = Image.open(io.BytesIO(image)).convert("RGB")
        encoding = processor(img, question, return_tensors="pt")
        outputs = vqa_model(**encoding)
        logits = outputs.logits
        idx = logits.argmax(-1).item()
        vqa_answer = vqa_model.config.id2label[idx]

        return vqa_answer

    except Exception as e:
        return str(e)

def generate_answer_with_chatgpt(image_features, question):
    """Generate a detailed answer using OpenAI's ChatGPT."""
    try:
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", 
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"The image shows {image_features}. Based on this, {question}"}
            ],
            max_tokens=150
        )
        detailed_answer = response['choices'][0]['message']['content'].strip()
        return detailed_answer

    except Exception as e:
        return str(e)


def handle_complex_question(image, question):
    """Combine VQA and ChatGPT to handle complex questions."""
    # Step 1: Get features or initial answer from the VQA model
    image_features = get_image_features(image, question)

    # Step 2: Use ChatGPT to generate a more detailed or descriptive answer
    detailed_answer = generate_answer_with_chatgpt(image_features, question)

    return detailed_answer

@app.route("/", methods=["GET", "POST"])
def index():
    answer = None
    if request.method == "POST":
        if "image" not in request.files or "question" not in request.form:
            answer = "Please provide an image and a question."
        else:
            image = request.files["image"]
            question = request.form["question"]

            image_bytes = image.read()

            answer = handle_complex_question(image_bytes, question)

    return render_template("index.html", answer=answer)

if __name__ == "__main__":
    app.run(port=8000, debug=True)
