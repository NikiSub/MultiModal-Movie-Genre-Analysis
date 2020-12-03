'''

    Image utilities to avoid pytorch transforms.
    Additional image transformations for webapps. e.g. image2string.

'''

import base64, numbers
from PIL import Image
from io import BytesIO
from PIL import ImageOps


def rotate_image_if_needed(img):
    img = ImageOps.exif_transpose(img)
    return img

def image2string(img, format = 'jpeg'):
    temporary_stream = BytesIO()
    img.save(temporary_stream, format)
    image_b64 = base64.b64encode(temporary_stream.getvalue())
    return str(image_b64, 'utf-8')

# From torchvision functional resize.
def resize_image(img, size, interpolation = Image.BILINEAR):
    w, h = img.size
    if (w <= h and w == size) or (h <= w and h == size):
        return img
    if w < h:
        ow = size
        oh = int(size * h / w)
        return img.resize((ow, oh), interpolation)
    else:
        oh = size
        ow = int(size * w / h)
        return img.resize((ow, oh), interpolation)

# From torchvision functional crop.
def crop_image(img, top, left, height, width):
    return img.crop((left, top, left + width, top + height))

# From torchvision functional center_crop.
def center_crop_image(img, output_size):
    if isinstance(output_size, numbers.Number):
        output_size = (int(output_size), int(output_size))
    image_width, image_height = img.size
    crop_height, crop_width = output_size
    crop_top = int(round((image_height - crop_height) / 2.))
    crop_left = int(round((image_width - crop_width) / 2.))
    return crop_image(img, crop_top, crop_left, crop_height, crop_width)

def top5_accuracy(predicted, labels):
    sorted_vals, sorted_ids = \
        predicted.data.sigmoid().sort(dim = 1, descending = True)
    pred_vals = sorted_vals[:, :5] > 0.5 # Anything with sigmoid > 0.5 is 1.
    true_vals = labels.data.gather(1, sorted_ids[:, :5]) # Find true values.
    return (pred_vals == true_vals).sum(dim = 1) / 5.0

def f_score(predicted, labels):
    #print(predicted)
    #print(predicted.data.sigmoid())
    pred_vals = predicted.data.sigmoid() > 0.5
    #print(pred_vals)
    #print(labels)
    true_positive = ((pred_vals==1) & (pred_vals == labels)).sum(dim = 1)
    false_positive = ((pred_vals==1) & (pred_vals != labels)).sum(dim = 1)
    false_negative = ((pred_vals==0) & (pred_vals != labels)).sum(dim = 1)
    f_score = true_positive/(true_positive+(false_positive+false_negative)/2)
    return f_score