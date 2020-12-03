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

# model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels = num_categories, output_attentions = False, output_hidden_states = False)

# model.load_state_dict(torch.load('./best_BERT_model.pth'))

# model.eval()

# Loading ResNet-18
model = torchvision.models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, num_categories)

model.load_state_dict(torch.load('best_img_model.pth', map_location=torch.device('cpu')))

model.eval() 


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

# COCO Captions Explorer.
@app.route('/coco-explorer', methods = ["GET"])
def coco_search():

	# Obtain the query string.
	query_str = request.args.get("query", "dog playing with ball")
	page_num = request.args.get("page_num", 1, type = int)
	page_len = request.args.get("page_len", 20, type = int)
	split = request.args.get("split", "train")

	# Location for the whoosh index to be queried.
	coco_index_path = 'static/whoosh/cococaptions-indexdir-%s' % split
	# Pre-load whoosh index to query coco-captions.
	cococaptions_index = index.open_dir(coco_index_path)

	# Return results and do any pre-formatting before sending to view.
	with cococaptions_index.searcher() as searcher:
		query = QueryParser("caption", cococaptions_index.schema).parse(query_str)
		results = searcher.search_page(query, page_num, pagelen = page_len)

		result_set = list()
		for result in results:
			result_set.append({"image_id": result["image_url"],
							   "caption": result["caption"].split("<S>")})

	# Create pagination navigation if needed.
	pagination = Pagination(query_str, len(results), page_num, page_len, other_arguments = {'split': split})

	# Render results template.
	return render_template('coco-search.html', 
            results = result_set, query = query_str, split = split, pagination = pagination)

if __name__ == '__main__':
    # Used when running locally only. When deploying to Google App
    # Engine, a webserver process such as Gunicorn will serve the app. This
    # can be configured by adding an `entrypoint` to app.yaml.
    app.run(host='localhost', port=8080, debug=True)
