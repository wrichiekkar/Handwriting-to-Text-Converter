import io
import os
from google.cloud import vision_v1
from google.cloud.vision_v1 import types
import pandas as pd


#Set the os GCP APP Variable
os.environ['GOOGLE_APPLICATION_CREDENTIALS']=r'ENTER JSON FILE NAME FROM GOOGLE CREDENTIALS'

#client for image annotate vision
client = vision_v1.ImageAnnotatorClient()

file_name = os.path.abspath('ENTER IMAGE NAME')

with io.open(file_name, 'rb') as image_file:
    content = image_file.read()

# construct an iamge instance
image = vision_v1.types.Image(content=content)

"""
# or send  the image url
image = vision.types.Image()
image.source.image_uri = 'https://edu.pngfacts.com/uploads/1/1/3/2/11320972/grade-10-english_orig.png'
"""

# annotate Image Response
response = client.text_detection(image=image)  # returns TextAnnotation
df = pd.DataFrame(columns=['locale', 'description'])

texts = response.text_annotations
for text in texts:
    df = df.append(
        dict(
            locale=text.locale,
            description=text.description
        ),
        ignore_index=True
    )
## Output convnersion here
print(df['description'][0])

'''
with io.open(file_name, 'rb') as image_file:
    content = image_file.read()

image = types.Image(content=content)

#response = client.label_detection(image=image)

response = client.text_detection(image=image)

labels = response.label_annotations

for label in labels:
    print(label)
'''