#%% [markdown]
# ## Load data
# 
# Let's read in the FUNSD dataset, which is stored on HuggingFace's hub.

# %%
from datasets import load_dataset 

datasets = load_dataset("nielsr/funsd")

# %% [markdown]
# As we can see, the dataset consists of a train and test split. Each example consists of an ID, words, corresponding bounding boxes, ner tags and an image path.

# %%
datasets

# %% [markdown]
# ## Preprocess data
# 
# First, let's store the labels in a list, and create dictionaries that let us map from labels to integer indices and vice versa. The latter will be useful when evaluating the model.

# %%
labels = datasets['train'].features['ner_tags'].feature.names
print(labels)

# %%
id2label = {v: k for v, k in enumerate(labels)}
label2id = {k: v for v, k in enumerate(labels)}
label2id

# %% [markdown]
# Next, let's use `LayoutLMv2Processor` to prepare the data for the model.

# %%
from PIL import Image
from transformers import LayoutLMv2Processor
from datasets import Features, Sequence, ClassLabel, Value, Array2D, Array3D

processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased", revision="no_ocr")

# we need to define custom features
features = Features({
    'image': Array3D(dtype="int64", shape=(3, 224, 224)),
    'input_ids': Sequence(feature=Value(dtype='int64')),
    'attention_mask': Sequence(Value(dtype='int64')),
    'token_type_ids': Sequence(Value(dtype='int64')),
    'bbox': Array2D(dtype="int64", shape=(512, 4)),
    'labels': Sequence(ClassLabel(names=labels)),
})

def preprocess_data(examples):
  images = [Image.open(path).convert("RGB") for path in examples['image_path']]
  words = examples['words']
  boxes = examples['bboxes']
  word_labels = examples['ner_tags']
  
  encoded_inputs = processor(images, words, boxes=boxes, word_labels=word_labels,
                             padding="max_length", truncation=True)
  
  return encoded_inputs

train_dataset = datasets['train'].map(preprocess_data, batched=True, remove_columns=datasets['train'].column_names,
                                      features=features)
test_dataset = datasets['test'].map(preprocess_data, batched=True, remove_columns=datasets['test'].column_names,
                                      features=features)

# %%
datasets['train'].bboxes

# %%
train_dataset[0]

# %%
processor.tokenizer.decode(train_dataset['input_ids'][0])

# %%
print(train_dataset['labels'][0])

# %% [markdown]
# Finally, let's set the format to PyTorch, and place everything on the GPU:

# %%
train_dataset.set_format(type="torch", device="cuda")
test_dataset.set_format(type="torch", device="cuda")

# %%
train_dataset.features.keys()

# %% [markdown]
# Next, we create corresponding dataloaders.

# %%
from torch.utils.data import DataLoader

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=2)

# %% [markdown]
# Let's verify a batch:

# %%
batch = next(iter(train_dataloader))

for k,v in batch.items():
  print(k, v.shape)

# %% [markdown]
# ## Train the model
# 
# Here we train the model in native PyTorch. We use the AdamW optimizer.

# %%
from transformers import LayoutLMv2ForTokenClassification, AdamW
import torch
from tqdm.notebook import tqdm

model = LayoutLMv2ForTokenClassification.from_pretrained('microsoft/layoutlmv2-base-uncased',
                                                          num_labels=len(labels))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
optimizer = AdamW(model.parameters(), lr=5e-5)

global_step = 0
num_train_epochs = 6
t_total = len(train_dataloader) * num_train_epochs # total number of training steps 

#put the model in training mode
model.train() 
for epoch in range(num_train_epochs):  
   print("Epoch:", epoch)
   for batch in tqdm(train_dataloader):
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(**batch) 
        loss = outputs.loss
        
        # print loss every 100 steps
        if global_step % 100 == 0:
          print(f"Loss after {global_step} steps: {loss.item()}")

        loss.backward()
        optimizer.step()
        global_step += 1

# %% [markdown]
# ## Evaluation
# 
# Next, let's evaluate the model on the test set.

# %%
from datasets import load_metric

metric = load_metric("seqeval")

# put model in evaluation mode
model.eval()
for batch in tqdm(test_dataloader, desc="Evaluating"):
    with torch.no_grad():
        input_ids = batch['input_ids'].to(device)
        bbox = batch['bbox'].to(device)
        image = batch['image'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels = batch['labels'].to(device)

        # forward pass
        outputs = model(input_ids=input_ids, bbox=bbox, image=image, attention_mask=attention_mask, 
                        token_type_ids=token_type_ids, labels=labels)
        
        # predictions
        predictions = outputs.logits.argmax(dim=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [id2label[p.item()] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [id2label[l.item()] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        metric.add_batch(predictions=true_predictions, references=true_labels)

final_score = metric.compute()
print(final_score)

# %% [markdown]
# The results I was getting are:
# 
# `{'ANSWER': {'precision': 0.7622324159021406, 'recall': 0.8609671848013817, 'f1': 0.8085969180859691, 'number': 1158}, 'HEADER': {'precision': 0.5104166666666666, 'recall': 0.34265734265734266, 'f1': 0.4100418410041841, 'number': 143}, 'QUESTION': {'precision': 0.8499327052489906, 'recall': 0.8627049180327869, 'f1': 0.856271186440678, 'number': 1464}, 'overall_precision': 0.798961937716263, 'overall_recall': 0.8350813743218807, 'overall_f1': 0.8166224580017684, 'overall_accuracy': 0.8195385460538362}`
# 
# This is consistent with the results of the paper, they report an overall F1 of 0.8276 with LayoutLMv2-base. Of course, better results will be obtained when using LayoutLMv2-large.

# %% [markdown]
# ## Inference
# 
# Let's test the trained model on the first image of the test set:

# %%
example = datasets["test"][0]
print(example.keys())

# %%
from PIL import Image, ImageDraw, ImageFont

image = Image.open(example['image_path'])
image = image.convert("RGB")
image

# %% [markdown]
# We prepare it for the model using `LayoutLMv2Processor`.

# %%
encoded_inputs = processor(image, example['words'], boxes=example['bboxes'], word_labels=example['ner_tags'],
                           padding="max_length", truncation=True, return_tensors="pt")

# %%
labels = encoded_inputs.pop('labels').squeeze().tolist()
for k,v in encoded_inputs.items():
  encoded_inputs[k] = v.to(device)

# %%
# forward pass
outputs = model(**encoded_inputs)

# %%
outputs.logits.shape

# %%
def unnormalize_box(bbox, width, height):
     return [
         width * (bbox[0] / 1000),
         height * (bbox[1] / 1000),
         width * (bbox[2] / 1000),
         height * (bbox[3] / 1000),
     ]

predictions = outputs.logits.argmax(-1).squeeze().tolist()
token_boxes = encoded_inputs.bbox.squeeze().tolist()

width, height = image.size

true_predictions = [id2label[prediction] for prediction, label in zip(predictions, labels) if label != -100]
true_labels = [id2label[label] for prediction, label in zip(predictions, labels) if label != -100]
true_boxes = [unnormalize_box(box, width, height) for box, label in zip(token_boxes, labels) if label != -100]

# %%
print(true_predictions)

# %%
print(true_labels)

# %%
print(true_boxes)

# %% [markdown]
# Let's visualize the result!

# %%
from PIL import ImageDraw

draw = ImageDraw.Draw(image)

font = ImageFont.load_default()

def iob_to_label(label):
    label = label[2:]
    if not label:
      return 'other'
    return label

label2color = {'question':'blue', 'answer':'green', 'header':'orange', 'other':'violet'}

for prediction, box in zip(true_predictions, true_boxes):
    predicted_label = iob_to_label(prediction).lower()
    draw.rectangle(box, outline=label2color[predicted_label])
    draw.text((box[0] + 10, box[1] - 10), text=predicted_label, fill=label2color[predicted_label], font=font)

image

# %% [markdown]
# Compare this to the ground truth:

# %%
example.keys()

# %%
image = Image.open(example['image_path'])
image = image.convert("RGB")

draw = ImageDraw.Draw(image)

for word, box, label in zip(example['words'], example['bboxes'], example['ner_tags']):
  actual_label = iob_to_label(id2label[label]).lower()
  box = unnormalize_box(box, width, height)
  draw.rectangle(box, outline=label2color[actual_label], width=2)
  draw.text((box[0] + 10, box[1] - 10), actual_label, fill=label2color[actual_label], font=font)

image

# %%



