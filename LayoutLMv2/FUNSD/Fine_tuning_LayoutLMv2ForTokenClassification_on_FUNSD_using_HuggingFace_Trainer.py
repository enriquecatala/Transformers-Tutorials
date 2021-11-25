
#%%
from datasets import load_dataset

datasets = load_dataset("nielsr/funsd")
labels = datasets['train'].features['ner_tags'].feature.names

id2label = {v: k for v, k in enumerate(labels)}
label2id = {k: v for v, k in enumerate(labels)}
label2id

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

#%%
from datasets import Dataset 

tmp =  preprocess_data(datasets['train'])

train_dataset = datasets['train'].map(preprocess_data, batched=True, remove_columns=datasets['train'].column_names,
                                      features=features)
test_dataset = datasets['test'].map(preprocess_data, batched=True, remove_columns=datasets['test'].column_names,
                                      features=features)

processor.tokenizer.decode(train_dataset['input_ids'][0])
print(train_dataset['labels'][0])
train_dataset.set_format(type="torch")
test_dataset.set_format(type="torch")
train_dataset.features.keys()

from torch.utils.data import DataLoader

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=2)

batch = next(iter(train_dataloader))

for k,v in batch.items():
  print(k, v.shape)

#%%
from transformers import LayoutLMv2ForTokenClassification, TrainingArguments, Trainer
from datasets import load_metric
import numpy as np

model = LayoutLMv2ForTokenClassification.from_pretrained('microsoft/layoutlmv2-base-uncased',
                                                                      num_labels=len(label2id))

# Set id2label and label2id 
model.config.id2label = id2label
model.config.label2id = label2id

# Metrics
metric = load_metric("seqeval")
return_entity_level_metrics = True

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    if return_entity_level_metrics:
        # Unpack nested dictionaries
        final_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                for n, v in value.items():
                    final_results[f"{key}_{n}"] = v
            else:
                final_results[key] = value
        return final_results
    else:
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

class FunsdTrainer(Trainer):
    def get_train_dataloader(self):
      return train_dataloader

    def get_test_dataloader(self, test_dataset):
      return test_dataloader

args = TrainingArguments(
    output_dir="layoutlmv2-finetuned-funsd-v2", # name of directory to store the checkpoints
    max_steps=1000, # we train for a maximum of 1,000 batches
    warmup_ratio=0.1, # we warmup a bit
    fp16=True, # we use mixed precision (less memory consumption)
    push_to_hub=True, # after training, we'd like to push our model to the hub
    push_to_hub_model_id=f"layoutlmv2-finetuned-funsd-test", # this is the name we'll use for our model on the hub
)

# Initialize our Trainer
trainer = FunsdTrainer(
    model=model,
    args=args,
    compute_metrics=compute_metrics,
)

#%%
trainer.train()