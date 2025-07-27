from transformers import pipeline
import json

user_prompt = input("Lets design! describe your idea:\n> ")

labels = {
    "category": ["fitness", "fashion", "food", "education", "tech", "real estate", "dental", "health care"],
    "style": ["modern", "grunge", "minimal", "vintage", "retro"],
    "platform": ["Instagram", "Facebook", "YouTube", "Print"],
    "tone": ["motivational", "professional", "casual", "formal"]
}

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")



def classify_field(prompt, label_list):
    output = classifier(prompt, label_list)
    return output['labels'][0]

result = {
    "project_type": "flyer",
    "goal": "promotion"
}

for field, options in labels.items():
    result[field] = classify_field(user_prompt, options)

# Save result as JSON
from datetime import datetime
filename = f"context_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
with open(filename, "w") as f:
    json.dump(result, f, indent=2)

print("\nStructured context saved to", filename)
print(json.dumps(result, indent=2))

