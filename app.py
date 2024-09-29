import os
from pathlib import Path
from byaldi import RAGMultiModalModel
from pdf2image import convert_from_path
import base64
from together import Together
import requests, os

model = RAGMultiModalModel.from_pretrained("vidore/colpali-v1.2")

index_name = "MoA_index"
model.index(input_path=Path("/current/CybersecutiyAuditReport.pdf"),
    index_name=index_name,
    store_collection_with_index=False,
    overwrite=True
)

#Indexing

query = "What's the conclusion of the report?"
results = model.search(query, k=5)

print(f"Search results for '{query}':")
for result in results:
    print(f"Doc ID: {result.doc_id}, Page: {result.page_num}, Score: {result.score}")

print("Test completed successfully!")


images = convert_from_path("/content/CSA.pdf")

images[4].save("retrieved_page.jpg")
images[4]


# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

#encode the image
retrieved_page_b64 = encode_image("retrieved_page.jpg")

# os.environ['TOGETHER_DEV_API_KEY'] = ""

# api_key = ''

client = Together(api_key=os.environ.get("TOGETHER_DEV_API_KEY"))

data = {
    "model": "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
    "max_tokens": 200,
    "messages":[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's the conclusion of the report?"}, #query
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{retrieved_page_b64}" #retrieved page image
                    }
                }
            ]
        }
    ],
    "stream": False,
    "logprobs": False
}

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

response = requests.post("https://api.together.xyz/v1/chat/completions", headers=headers, json=data)

response.json()['choices'][0]['message']['content']


# from groq import Groq

# client = Groq(
#     api_key="",
# )

# chat_completion = client.chat.completions.create(
#     messages=[
#     {
#             "role": "user",
#             "content": [
#                 {"type": "text", "text": "What's the conclusion of the report?"}, #query
#                 {
#                     "type": "image_url",
#                     "image_url": {
#                         "url": f"data:image/jpeg;base64,{retrieved_page_b64}" #retrieved page image
#                     }
#                 }
#             ]
#         }
#     ],
#     model="llama3-8b-8192",
# )

# print(chat_completion.choices[0].message.content)

