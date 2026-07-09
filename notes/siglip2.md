# SigLIP2

[SigLIP2](https://huggingface.co/papers/2502.14786) is a family of multilingual vision-language encoders that builds on the [SigLIP](./siglip) training recipe. It includes decoder-based pretraining, self-distillation, and masked prediction to improve dense prediction tasks (segmentation, depth estimation, etc.). This model is available in two variants:

- NaFlex supports different resolutions and maintains the native image aspect ratio
- FixRes supports fixed resolutions and is backwards compatible with [SigLIP](./siglip)

You can find all the original SigLIP2 checkpoints under the [SigLIP2](https://huggingface.co/collections/google/siglip2-67b5dcef38c175486e240107) collection.

## What SigLIP2 is for

SigLIP2 is not a chat model and it does not generate text. It is an encoder model, designed to turn images and text into vectors that can be compared. The usual workflow is:

1. Encode an image with the vision tower.
2. Encode one or more text strings with the text tower.
3. Project both outputs into the same embedding space.
4. Compare image and text embeddings with a dot product or cosine similarity.

If the image embedding is close to the text embedding for `"a photo of a cat"`, the model is saying that the image and that text are a good semantic match. This is why SigLIP2 is useful for zero-shot image classification, image-text retrieval, caption matching, dataset filtering, and as a vision encoder inside larger multimodal systems.

The "LIP" part of the name comes from contrastive language-image pretraining (CLIP). The model learns by seeing many image-text pairs and adjusting the encoders so matching pairs become close and non-matching pairs become far apart.

SigLIP's main change from CLIP is the loss function. CLIP commonly uses a softmax-style contrastive loss over a batch. SigLIP uses independent sigmoid binary classifications over image-text pairs. Practically, this means each pair can be treated as its own "does this match?" decision rather than only as one choice among all candidates in the current batch.

## Mental model

Think of SigLIP2 as two coordinated encoders:

- The image encoder splits the image into patches, processes those patches with a vision transformer, and produces an image embedding.
- The text encoder tokenizes the text, processes the tokens with a transformer, and produces a text embedding.
- A projection layer maps both embeddings into the same comparison space.
- A learned scale and bias are applied to the image-text similarity score before the sigmoid.

The score from `outputs.logits_per_image` is not a probability by itself. It is a logit, meaning a value before the sigmoid function. Applying `torch.sigmoid(logits_per_image)` converts each image-text pair into an independent match confidence. This is why the example code uses `sigmoid` instead of `softmax`.

The independence matters when reading the output. Several labels can receive high scores if several descriptions plausibly fit the image, and all labels can receive low scores if none of them fit. For single-label classification you can still choose the highest-scoring label, but the model itself is scoring compatibility rather than enforcing "exactly one answer".

## How to read the model names

Model names such as `google/siglip2-base-patch16-224` encode several important choices:

- `base`, `large`, `so400m`, etc. describe the rough model size.
- `patch16` or `patch14` describes the patch size used by the vision encoder. Smaller patches create more image tokens for the same resolution and can preserve more detail, but cost more compute.
- `224`, `384`, and similar suffixes are fixed input resolutions for FixRes models.
- `naflex` means the model can keep variable aspect ratios and operate under a patch budget instead of forcing every image into one fixed square resolution.

For deployment work, the patch and resolution choices are not just accuracy details. They affect tensor shapes, memory use, RKNN conversion constraints, and the number of image tokens that must be passed into any downstream language model.

## FixRes versus NaFlex

FixRes models are easier to reason about because the input image size is fixed. For example, a `patch16-224` image encoder sees a 224 by 224 image divided into 16 by 16 patches. That produces a 14 by 14 patch grid, or 196 image patches, before any implementation-specific pooling or projection.

NaFlex models are more flexible. They resize the image so its dimensions are compatible with the patch size, preserve the original aspect ratio as much as possible, and then pad the patch sequence up to `max_num_patches`. This is often better for documents, screenshots, OCR-style inputs, and other images where stretching to a square can damage important spatial structure.

The tradeoff is that NaFlex introduces more metadata. In addition to `pixel_values`, the model may need a `pixel_attention_mask` and
`spatial_shapes` so it can tell real patches from padded patches and recover the layout of the resized image. If exporting or converting the model, check these inputs carefully rather than assuming a single fixed image tensor is enough.

## Why preprocessing matters

SigLIP2 is sensitive to preprocessing because the encoders were trained with specific text and image normalization rules. Small-looking differences can move embeddings enough to hurt retrieval or classification quality.

For text, the important rules are:

- Lowercase text before encoding.
- Pad and truncate to the training length, typically `max_length=64`.
- Use the same prompt wording when comparing against example outputs. The pipeline template is `"This is a photo of {label}."`.

For images, the important rules are:

- Use the image processor that belongs to the checkpoint when working in Transformers.
- Match the expected patch size and resolution or patch budget.
- Preserve the expected normalization, channel order, and tensor layout when porting the model outside Transformers.

When converting to RKNN, the key question is whether preprocessing is baked into the exported graph or must be reproduced by the host application. This repo should treat that as a model-profile detail, not as a universal image resize rule shared by all vision encoders.

## Example Code

The examples below focus on the parts that are most useful when learning or porting SigLIP2:

- Zero-shot classification produces label scores for an image.
- Manual image-text scoring exposes the logits and probabilities directly.
- Embedding extraction produces vectors that can be stored and compared.
- Processor inspection shows the tensors that must be reproduced for export or deployment.

SigLIP2 does not generate captions or answers. It generates embeddings and image-text compatibility scores.

### Zero-shot classification with pipeline

Use the pipeline when you want the shortest path from an image plus labels to ranked scores.

```python
from transformers import pipeline

image = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
candidate_labels = ["a Pallas cat", "a lion", "a Siberian tiger"]

classifier = pipeline(
    task="zero-shot-image-classification",
    model="google/siglip2-base-patch16-224",
    device=0,
)

scores = classifier(image, candidate_labels=candidate_labels)
print(scores)
```

This generates a list of dictionaries, usually ordered from best match to worst match:

```python
[
    {"score": 0.99, "label": "a Pallas cat"},
    {"score": 0.01, "label": "a lion"},
    {"score": 0.00, "label": "a Siberian tiger"},
]
```

The exact numbers depend on the model, library version, hardware precision, and image. The important point is the shape of the result: each label gets its own match score. These scores are convenient for demos, but the pipeline hides the tensors and preprocessing details.

### Manual image-text scoring

Use `AutoModel` and `AutoProcessor` when you need to see the model inputs and outputs directly. This is the better starting point for export, conversion, or debugging.

```python
import requests
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

model_id = "google/siglip2-base-patch16-224"
model = AutoModel.from_pretrained(model_id).eval()
processor = AutoProcessor.from_pretrained(model_id)

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = Image.open(requests.get(url, stream=True).raw)

candidate_labels = ["a Pallas cat", "a lion", "a Siberian tiger"]
texts = [f"This is a photo of {label}." for label in candidate_labels]

inputs = processor(
    text=texts,
    images=image,
    padding="max_length",
    max_length=64,
    return_tensors="pt",
)

with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits_per_image
probs = torch.sigmoid(logits)
best_label = candidate_labels[probs[0].argmax().item()]

print(logits.shape)
print(probs)
print(best_label)
```

This generates:

- `inputs`, a dictionary of tensors such as `input_ids`, `attention_mask`, and `pixel_values`.
- `outputs.logits_per_image`, a tensor shaped `(num_images, num_texts)`.
- `probs`, the sigmoid of those logits, with one independent match confidence per image-text pair.
- `best_label`, the highest-scoring label if you want a single classification result.

For one image and three candidate labels, `logits_per_image` has shape `(1, 3)`. Use `torch.sigmoid`, not `softmax`, when you want SigLIP-style independent match probabilities.

### NaFlex image-text scoring

Use a NaFlex checkpoint when preserving aspect ratio matters or when images vary widely in shape and detail.

```python
import requests
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

model_id = "google/siglip2-base-patch16-naflex"
model = AutoModel.from_pretrained(model_id).eval()
processor = AutoProcessor.from_pretrained(model_id)

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = Image.open(requests.get(url, stream=True).raw)

texts = [
    "This is a photo of a Pallas cat.",
    "This is a photo of a lion.",
    "This is a photo of a Siberian tiger.",
]

inputs = processor(
    text=texts,
    images=image,
    padding="max_length",
    max_num_patches=256,
    return_tensors="pt",
)

with torch.no_grad():
    outputs = model(**inputs)

probs = torch.sigmoid(outputs.logits_per_image)
print(probs)
```

This generates the same kind of image-text scores as the fixed-resolution example, but the image preprocessing is different. Instead of forcing the image to a fixed square input, the processor resizes to a patch budget and may include extra tensors such as `pixel_attention_mask` and `spatial_shapes`.

`max_num_patches` is the main quality and cost knob. A larger value lets the model see more image detail, but it also increases memory and compute. For RKNN export, these extra tensors and variable shapes are the details most likely to matter.

### Text embeddings for retrieval

Use `get_text_features` when building a text index or checking how text strings compare in SigLIP2's embedding space.

```python
import torch
from transformers import AutoModel, AutoProcessor

model_id = "google/siglip2-so400m-patch14-384"
model = AutoModel.from_pretrained(model_id).eval()
processor = AutoProcessor.from_pretrained(model_id)

texts = [
    "transparent digital bathroom scale",
    "digital personal weight scale",
    "stainless steel kitchen thermometer",
]

inputs = processor(text=texts, return_tensors="pt")

with torch.no_grad():
    text_features = model.get_text_features(**inputs)

text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
print(text_features.shape)
```

This generates one embedding vector per input string. The printed shape is `(num_texts, embedding_dim)`. The exact embedding dimension depends on the checkpoint.

After normalization, dot product is equivalent to cosine similarity. For text retrieval, store these vectors in a vector index. For image retrieval, compute image embeddings with `get_image_features`, normalize them in the same way, and compare query vectors against stored vectors.

### Image embeddings for retrieval

Use `get_image_features` when you need image vectors rather than image-text scores.

```python
import requests
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

model_id = "google/siglip2-base-patch16-224"
model = AutoModel.from_pretrained(model_id).eval()
processor = AutoProcessor.from_pretrained(model_id)

urls = [
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
]

images = [Image.open(requests.get(url, stream=True).raw) for url in urls]
inputs = processor(images=images, return_tensors="pt")

with torch.no_grad():
    image_features = model.get_image_features(**inputs)

image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
print(image_features.shape)
```

This generates one embedding vector per image. The printed shape is `(num_images, embedding_dim)`. These vectors can be compared with normalized text embeddings from the same checkpoint.

Do not compare embeddings produced by different SigLIP2 checkpoints unless you have validated that combination. Each checkpoint defines its own embedding space.

### Inspecting processor outputs

Use this when preparing an export path, because it shows which tensors the PyTorch model receives after preprocessing.

```python
import requests
from PIL import Image
from transformers import AutoProcessor

model_id = "google/siglip2-base-patch16-naflex"
processor = AutoProcessor.from_pretrained(model_id)

image = Image.open(
    requests.get(
        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
        stream=True,
    ).raw
)

inputs = processor(
    text=["This is a photo of a cat."],
    images=image,
    padding="max_length",
    max_num_patches=256,
    return_tensors="pt",
)

for name, tensor in inputs.items():
    print(name, tuple(tensor.shape), tensor.dtype)
```

This generates a list of input tensor names, shapes, and dtypes. For a fixed-resolution model you should expect text tensors and image pixels. For a NaFlex model, expect additional shape or mask metadata when the checkpoint requires it.

These printed tensors are a useful checklist for RKNN conversion:

- Which inputs must the exported graph accept?
- Are pixels already normalized in the graph, or must the host normalize them?
- Is the image layout channels-first or channels-last?
- Are `pixel_attention_mask` or `spatial_shapes` required?
- Which dimensions are static, and which are allowed to vary?

## Practical notes

- SigLIP2 text should be lowercased. `AutoProcessor` and `Siglip2Tokenizer` handle this for normal Transformers usage.
- Keep `padding="max_length"` and `max_length=64` for text unless you know the checkpoint was trained differently.
- Use the prompt template `"This is a photo of {label}."` when trying to match pipeline-style zero-shot classification behavior.
- Toggle `attn_implementation` to `"sdpa"` or `"flash_attention_2"` only for PyTorch inference performance. It does not change the conceptual inputs or outputs.
- Quantization can reduce memory use for PyTorch experiments, but RKNN/RKLLM deployment has its own conversion and quantization path.

## Relevance to this repo

For this RKNN/RKLLM project, SigLIP2 is most relevant as a possible vision encoder rather than as a complete multimodal assistant. A converted SigLIP2 vision tower could produce image features, but a separate language model or runtime path still has to consume those features.

When evaluating whether a SigLIP2 checkpoint can be used here, confirm:

- The exact exported RKNN input tensor layout, dtype, normalization, and resize policy.
- Whether the model is FixRes or NaFlex, since NaFlex may require masks and spatial-shape metadata.
- The output tensor shape and whether it is a pooled image embedding, per-patch features, or another intermediate representation.
- The embedding width and number of image tokens expected by the downstream RKLLM model.
- Whether text encoding is needed on-device, or only image encoding.

The safest implementation pattern is to capture these details in a model profile. Avoid assuming that the current Qwen-VL preprocessing path is correct for SigLIP2 or for a SigLIP-derived encoder used by another VLM.
