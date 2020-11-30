import torch, os, json
import matplotlib.pyplot as plt
from transformers import BertTokenizer
from PIL import Image
from text_extractor import TextExtractor

class MovieDataset(torch.utils.data.Dataset):
    def __init__(self, folder = 'data/mmimdb-256/dataset-resized-256max', split = 'dev',
                 image_transform = None):
        self.json_dir = os.path.join(folder, split, 'metadata')
        self.image_dir = os.path.join(folder, split, 'images')
        self.image_transform = image_transform
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.text_extractor = TextExtractor(folder+"/"+split+"/images/",split+"_"+"dataset_text_extract_output.txt",split)
        #insantiate a model to extract text

        # Category definitions of movies.
        self.categories = ['Action', 'Adult', 'Adventure', 'Animation', 'Biography', 
                           'Comedy', 'Crime', 'Documentary', 'Drama', 
                           'Family', 'Fantasy', 'Film-Noir', 'History', 
                           'Horror', 'Music', 'Musical', 'Mystery', 'News', 
                           'Reality-TV', 'Romance', 'Sci-Fi', 'Short', 
                           'Sport', 'Talk-Show', 'Thriller', 'War', 'Western']
        self.categories2ids = {category: id for (id, category) 
                               in enumerate(self.categories)}

        # Load JSON files.
        #print('Loading %s ...' % self.json_dir, end = '')
        print("extracting text and getting metadata")
        self.fdir = os.listdir(self.json_dir)
        self.metadata = [(fname[:-5], json.load(open(os.path.join(self.json_dir, fname)))) 
                     for fname in sorted(self.fdir) if not fname.startswith('.')]
        print(len(self.metadata))
        self.text_extractor.extract_text()
        
        print(' finished')

        # Pre-tokenizing all sentences.
        
        print('Tokenizing...', end = '')
        self.tokenized_plots = list()
        for i in range(0, len(self.metadata)):
            text = self.text_extractor.get_item(i) #self.metadata[i][1]['plot'][0]
            encoded_text = self.tokenizer.encode_plus(
                text, add_special_tokens = True, truncation = True, 
                max_length = 256, padding = 'max_length',
                return_attention_mask = True,
                return_tensors = 'pt')
            self.tokenized_plots.append(encoded_text)
        print(' finished')
        
            
    def __getitem__(self, index: int):
        # Load images on the fly.
        filename, movie_data = self.metadata[index]
        img_path = os.path.join(self.image_dir, filename + '.jpeg')
        image = Image.open(img_path).convert('RGB')
        #TODO: ADD cacheing
        text = self.tokenized_plots[index]['input_ids'][0]
        text_mask = self.tokenized_plots[index]['attention_mask'][0]
        genres = movie_data['genres']

        if self.image_transform: image = self.image_transform(image)

        # Encode labels in a binary vector.
        label_vector = torch.zeros((len(self.categories)))
        label_ids = [self.categories2ids[cat] for cat in genres]
        label_vector[label_ids] = 1

        return image, text, text_mask, label_vector

    def load_image_only(self, index: int):
        filename, movie_data = self.metadata[index]
        img_path = os.path.join(self.image_dir, filename + '.jpeg')
        image = Image.open(img_path).convert('RGB')
        return image


    def __len__(self):
        return len(self.metadata)

val_data = MovieDataset(split = 'train')
print('Data size: %d samples' % len(val_data))

sample_movieID = 2
img, text, text_mask, labels = val_data[sample_movieID]
#print(text)
#print(text_mask)

print(val_data.tokenizer.convert_ids_to_tokens(text.numpy().tolist()))
#print(val_data.tokenizer.convert_ids_to_tokens([100,0,1,2,3,4,5,6,7,8,9,10,101,102,103,104,105,106,107]))

#labels = labels.numpy()
# Is there a better way to do this?
#print([val_data.categories[ind] for ind, val in enumerate(labels == 1) if val == 1])

plt.imshow(img)
plt.show()