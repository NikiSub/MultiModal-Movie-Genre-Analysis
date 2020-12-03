'''
	Main app for vislang.ai
'''
import random, io, time
import requests as http_requests

from flask import Flask, request, redirect, flash, url_for
from flask import render_template
import validators


from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge

from pagination import Pagination
from utils import resize_image, center_crop_image, image2string, rotate_image_if_needed

# If `entrypoint` is not defined in app.yaml, App Engine will look for an app
# called `app` in `main.py`.
app = Flask(__name__)

@app.route('/', methods=['GET'])
def main():
	return render_template('index.html')



# Simple Demo demo.
import imghdr, json
from PIL import Image

# Restrict filetypes allowed.  
# Important: This doesn't just check for file extensions.
ALLOWED_IMAGE_TYPES = ['jpeg' , 'png']

# Configure app to allow maximum  of 15 MB image file sizes.
# Note: This will send a 403 HTTP error page. 
# So we will need to validate file size on the client side.
# Client side could have a stricter requirement size.
app.config['MAX_CONTENT_LENGTH'] = 15 * 1024 * 1024


####################################################

from PIL import Image
import torchvision
from torchvision import transforms
import torch
from transformers import BertForSequenceClassification, BertConfig


print("loading ResNet18")
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Loading BERT
categories = ['Action', 'Adult', 'Adventure', 'Animation', 'Biography', 
                           'Comedy', 'Crime', 'Documentary', 'Drama', 
                           'Family', 'Fantasy', 'Film-Noir', 'History', 
                           'Horror', 'Music', 'Musical', 'Mystery', 'News', 
                           'Reality-TV', 'Romance', 'Sci-Fi', 'Short', 
                           'Sport', 'Talk-Show', 'Thriller', 'War', 'Western']
genreClasses = {id: category for (id, category) in enumerate(categories)}

num_categories = len(categories) 

text_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels = num_categories, output_attentions = False, output_hidden_states = False)

#text_model.load_state_dict(torch.load('best_model_bert.pth'))

text_model.eval()

# Loading ResNet-18
img_model = torchvision.models.resnet18(pretrained=True)
num_ftrs = img_model.fc.in_features
img_model.fc = torch.nn.Linear(num_ftrs, num_categories)

img_model.load_state_dict(torch.load('best_img_model.pth', map_location=torch.device('cpu')))

img_model.eval()

from gmu_model import LinearClassifier, LinearCombine, Gated_MultiModal_Unit

gmu_model = Gated_MultiModal_Unit(img_model, text_model)
gmu_model.load_state_dict(torch.load('best_model_mmu.pth', map_location=torch.device('cpu')))




# preload ResNet-50.
# model = torchvision.models.resnet50(pretrained = True)
# model.eval()
print('ResNet18 was loaded!!!!!')

# Load imagenet class names.
# imagenetClasses = {int(idx): entry[1] for (idx, entry) in 
#                    json.load(open('imagenet_class_index.json')).items()}

########################################################



@app.route('/simple-demo', methods = ["GET", "POST"])
def simple_demo():

	# If the request is GET then only render template.
	if request.method == "GET":
		return render_template('simple-demo.html')

	# If the request is POST then handle the upload.
	image_url = request.form.get('image_url')
	if image_url and validators.url(image_url):
		# TODO: Handle valid url and exceptions.
		response = http_requests.get(image_url)
		file = io.BytesIO(response.content)
		filename = image_url
	else:
		# Get file information just for display purposes.
		file = request.files.get('image')
		if not file:
			return {"error": "file-invalid", 
					"message": "The uploaded file is invalid"}
		filename = file.filename


	if len(filename) > 22: filename = filename[:10] + "..." + filename[-10:]

	# Verify if this is a valid image type.
	filestream = file.read()
	image_type = imghdr.what("", h = filestream)

	if image_type not in ALLOWED_IMAGE_TYPES:
		return {"error": "file-type-not-allowed",
				"message": "The uploaded file must be a JPG or PNG image"}

	# Read the image directly from the file stream.
	file.seek(0)  # Reset file stream pointer.
	img = Image.open(file).convert('RGB')

	# If the image is uploaded from a mobile device.
	# this avoids having the image rotated.
	img = rotate_image_if_needed(img)

	# Resize and crop the input image.
	img = resize_image(img, 256)
	img = center_crop_image(img, 224)

	# Encode the cropped  input image as a string for display.
	input_image_str = image2string(img)

	# Now process the image.
	from PIL import ImageFilter
	start_time = time.time()
	output_img = img.filter(ImageFilter.GaussianBlur(radius = 10))
	debug_str = 'process took %.2f seconds' % (time.time() - start_time)

	# Now let's try some pytorch.
	input_tensor = preprocess(img) # Convert to tensor.
	print(input_tensor.shape)  # Print shape.
	print('\n\n\n')
	
	input_batch = input_tensor.unsqueeze(0)  # Add batch dim.
	#print(model) # Print the model.
	
	outputs = model(input_batch)
	print(outputs.shape)

	# # Apply softmax and sort the values.
	probs, indices = (-outputs.softmax(dim = 1).data).sort()
	# # Pick the top-5 scoring ones.
	probs = (-probs).numpy()[0][:5]; indices = indices.numpy()[0][:10]
	# # Concat the top-5 scoring indices with the class names.
	preds = ['P[\"' + genreClasses[idx] + '\"] = ' + ('%.6f' % prob) \
         for (prob, idx) in zip(probs, indices)]

	# # Print top classes predicted.
	print('\n'.join(preds))
	my_str = '\n'.join(preds)

	# Encode the output image as a string for display.
	output_image_str = image2string(output_img)

	return {'filename': filename, 
			'input_image': input_image_str, 
			'output_image': output_image_str,
			'debug_str': my_str}


if __name__ == '__main__':
    # Used when running locally only. When deploying to Google App
    # Engine, a webserver process such as Gunicorn will serve the app. This
    # can be configured by adding an `entrypoint` to app.yaml.
    app.run(host='localhost', port=8080, debug=True)
