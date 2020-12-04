'''
	Main app for vislang.ai
'''
import sys
import os, shutil
import random, io, time
import requests as http_requests

from flask import Flask, request, redirect, flash, url_for
from flask import render_template
import validators


from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge

from pagination import Pagination
from utils_a import resize_image, center_crop_image, image2string, rotate_image_if_needed

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
from transformers import BertTokenizer


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

print("Loading BERT")
text_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels = num_categories, output_attentions = False, output_hidden_states = False)

text_model.load_state_dict(torch.load('best_bert_model.pth', map_location=torch.device('cpu')))

text_model.eval()
print("BERT loaded successfully!")

# Loading ResNet-18
print("Loading ResNet-18")
img_model = torchvision.models.resnet18(pretrained=True)
num_ftrs = img_model.fc.in_features
img_model.fc = torch.nn.Linear(num_ftrs, num_categories)

# 

img_model.load_state_dict(torch.load('best_img_model.pth', map_location=torch.device('cpu')))

img_model.eval()

print("ResNet-18 loaded successfully!")

sys.path.insert(1, '../')
from gmu_model import LinearClassifier, LinearCombine, Gated_MultiModal_Unit
from text_extractor import TextExtractor
gmu_model = Gated_MultiModal_Unit(img_model, text_model)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

#predicted = gmu_model.forward('eval', imgs, text, text_mask, label)

print("Loading GMU")

checkpoint = torch.load('best_gmu.pth', map_location=torch.device('cpu'))
gmu_model.hv_gate.load_state_dict(checkpoint['hv_gate_state_dict'])
gmu_model.ht_gate.load_state_dict(checkpoint['ht_gate_state_dict'])
gmu_model.z_gate.load_state_dict(checkpoint['z_gate_state_dict'])

print("GMU loaded successfully!")


print("Preparing text extractor")

if not os.path.isdir("./demo_images/"):
	os.mkdir("./demo_images/")

##Clear out folder	
for filename in os.listdir("./demo_images/"):
    file_path = os.path.join("./demo_images/", filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))
text_extractor = TextExtractor("./demo_images/","demo_text_extract_output.txt","demo")

print("Text extractor loaded successfully!")


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
	#Clear Demo folder
	for filename in os.listdir("./demo_images/"):
		file_path = os.path.join("./demo_images/", filename)
		try:
			if os.path.isfile(file_path) or os.path.islink(file_path):
				os.unlink(file_path)
			elif os.path.isdir(file_path):
				shutil.rmtree(file_path)
		except Exception as e:
			print('Failed to delete %s. Reason: %s' % (file_path, e))
	if os.path.isdir("./demo_intermediate_result/"):
		for filename in os.listdir("./demo_intermediate_result/"):
			file_path = os.path.join("./demo_intermediate_result/", filename)
			try:
				if os.path.isfile(file_path) or os.path.islink(file_path):
					os.unlink(file_path)
				elif os.path.isdir(file_path):
					shutil.rmtree(file_path)
			except Exception as e:
				print('Failed to delete %s. Reason: %s' % (file_path, e))
	#print("trying to save image")
	img.save("./demo_images/0001.jpeg")
	#print("saved image, starting text extraction")
	text_extractor.extract_text()
	text_from_file = text_extractor.get_item(0) #self.metadata[i][1]['plot'][0]
	encoded_text = tokenizer.encode_plus(
	text_from_file, add_special_tokens = True, truncation = True, 
		max_length = 256, padding = 'max_length',
		return_attention_mask = True,
		return_tensors = 'pt')
	text = encoded_text['input_ids'][0]
	text_mask = encoded_text['attention_mask'][0]
	print(tokenizer.convert_ids_to_tokens(text.numpy().tolist()))
	
	#print("Finished text extraction")
	# If the image is uploaded from a mobile device.
	# this avoids having the image rotated.
	#img = rotate_image_if_needed(img)

	# Resize and crop the input image.
	#img = resize_image(img, 256)
	#img = center_crop_image(img, 224)

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
	
	outputs = img_model(input_batch)
	print(outputs.shape)

	print(f"IMAGE CLASSIFIER OUTPUT: {outputs}")

	outputs = outputs.flatten()
	m = torch.nn.Sigmoid()
	output = m(outputs)

	values, indices = output.topk(5)

	values = values.tolist()
	indices = indices.tolist()

	preds = ['<b>' + genreClasses[idx] + ' score</b>: ' + ('%.4f' % val) + '<br>' for (val, idx) in zip(values, indices)] 

	# preds = ''
	# for val, idx in zip(values, indices):
	# 	preds += genreClasses[idx] + ' score: ' + str(val) + '\n'

	my_str = '\n'.join(preds)

	# # # Apply softmax and sort the values.
	# probs, indices = (-outputs.softmax(dim = 1).data).sort()
	# # # Pick the top-5 scoring ones.
	# probs = (-probs).numpy()[0][:5]; indices = indices.numpy()[0][:10]
	# # # Concat the top-5 scoring indices with the class names.
	# preds = ['P[\"' + genreClasses[idx] + '\"] = ' + ('%.6f' % prob) \
    #      for (prob, idx) in zip(probs, indices)]

	# # # Print top classes predicted.
	# print('\n'.join(preds))
	# my_str = '\n'.join(preds)

	# Encode the output image as a string for display.
	output_image_str = image2string(output_img)

	print(f"TEXT SHAPE: {text.shape}")
	print(f"TEXT_MASK SHAPE: {text_mask.shape}")
	
	text = text.unsqueeze(0)
	text_mask = text_mask.unsqueeze(0)

	predicted = gmu_model.forward('eval', input_batch, text, text_mask)

	outputs = predicted.flatten()

	values, indices = outputs.topk(5)

	values = values.tolist()
	indices = indices.tolist()

	preds = ['<b>' + genreClasses[idx] + ' score</b>: ' + ('%.4f' % val) + '<br>' for (val, idx) in zip(values, indices)]

	my_str2 = '\n'.join(preds)
	
	print(f'GMU PREDICTION: {predicted}')

	print(f'my_str2: {my_str2}')

	print(f'text_from_file: {text_from_file}')

	my_str3 = '<b>Extracted text: </b>' + text_from_file




	

	return {'filename': filename, 
			'input_image': input_image_str, 
			'output_image': output_image_str,
			'debug_str': my_str, 
			'debug_str2': my_str2,
			'debug_str3': my_str3}


if __name__ == '__main__':
    # Used when running locally only. When deploying to Google App
    # Engine, a webserver process such as Gunicorn will serve the app. This
    # can be configured by adding an `entrypoint` to app.yaml.
    app.run(host='localhost', port=8080, debug=True)
