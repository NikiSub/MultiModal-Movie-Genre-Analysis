from PIL import Image
import pytesseract
a=Image.open('ocr_test_3.jpg')
text = pytesseract.image_to_string(a)
print(text.split())