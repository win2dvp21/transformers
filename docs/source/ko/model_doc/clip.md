<!--Copyright 2021 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# CLIP [[clip]]

## 개요 [[overview]]

CLIP 모델은 Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever에 의해 제안된 [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) 논문에서 제시되었습니다. CLIP(Contrastive Language-Image Pre-Training)는 다양한 (이미지, 텍스트) 쌍으로 학습된 신경망입니다. 이 모델은 자연어로 이미지를 주어 가장 관련성 높은 텍스트 스니펫을 예측하도록 지시할 수 있으며, 이를 위한 작업을 직접 최적화하지 않고도 GPT-2와 3의 zero-shot 기능과 유사한 방식으로 작동합니다.

논문의 초록은 다음과 같습니다:

*최첨단 컴퓨터 비전 시스템은 고정된 사전 결정된 객체 카테고리를 예측하도록 학습됩니다. 이 제한된 형태의 감독은 이들의 일반성과 사용성을 제한하며, 다른 시각적 개념을 지정하려면 추가로 라벨링된 데이터가 필요합니다. 이미지를 설명하는 텍스트로부터 직접 학습하는 것은 훨씬 더 폭넓은 감독 소스를 활용하는 유망한 대안입니다. 우리는 4억 개의 (이미지, 텍스트) 쌍으로 구성된 데이터셋에서 캡션과 이미지가 일치하는지를 예측하는 단순한 사전 학습 작업이 최첨단(SOTA) 이미지 표현을 처음부터 효율적이고 확장 가능하게 학습할 수 있는 방법임을 입증합니다. 사전 학습 후에는 자연어를 사용하여 학습된 시각적 개념을 참조하거나 새로운 개념을 설명할 수 있으며, 이를 통해 모델을 후속 작업으로 zero-shot 전환할 수 있습니다. 우리는 OCR, 비디오 내 동작 인식, 지리적 위치 지정, 세밀한 객체 분류 등의 다양한 컴퓨터 비전 데이터셋을 포함한 30개 이상의 기존 데이터셋에서 이 접근법의 성능을 연구합니다. 이 모델은 대부분의 작업에서 비지도 학습된 베이스라인과 경쟁할 수 있으며, 데이터셋별 학습이 필요하지 않습니다. 예를 들어, 우리는 ResNet-50의 원래 ImageNet zero-shot 정확도를 1.28백만 개의 학습 예제 없이도 맞춥니다. 우리는 이 코드와 사전 학습된 모델 가중치를 공개합니다.* 

이 모델은 [valhalla](https://huggingface.co/valhalla)에 의해 제공되었습니다. 원본 코드는 [여기](https://github.com/openai/CLIP)에서 확인할 수 있습니다.

## 사용 팁 및 예시 [[usage-tips-and-example]]

CLIP은 다중 모달 비전 및 언어 모델입니다. 이미지-텍스트 유사도 계산 및 zero-shot 이미지 분류에 사용할 수 있습니다. CLIP은 ViT 유사한 트랜스포머를 사용하여 시각적 특징을 얻고, 인과적 언어 모델을 사용하여 텍스트 특징을 얻습니다. 그 후 텍스트와 시각적 특징 모두 동일한 차원의 잠재 공간에 투영됩니다. 투영된 이미지와 텍스트 특징 간의 내적 곱(dot product)을 사용하여 유사 점수를 계산합니다.

트랜스포머 인코더에 이미지를 입력하려면, 각 이미지를 고정된 크기의 겹치지 않는 패치로 나누고 이를 선형적으로 임베딩합니다. [CLS] 토큰이 전체 이미지를 나타내는 역할을 하도록 추가됩니다. 저자들은 또한 절대 위치 임베딩을 추가하고, 그 결과 벡터 시퀀스를 표준 트랜스포머 인코더에 입력합니다. [`CLIPImageProcessor`]는 모델에 이미지를 크기 조정(또는 스케일링)하고 정규화하는 데 사용할 수 있습니다.

[`CLIPTokenizer`]는 텍스트를 인코딩하는 데 사용됩니다. [`CLIPProcessor`]는 [`CLIPImageProcessor`]와 [`CLIPTokenizer`]를 단일 인스턴스로 감싸 텍스트를 인코딩하고 이미지를 준비할 수 있습니다. 다음 예시는 [`CLIPProcessor`]와 [`CLIPModel`]을 사용하여 이미지-텍스트 유사도 점수를 얻는 방법을 보여줍니다.


```python
>>> from PIL import Image
>>> import requests

>>> from transformers import CLIPProcessor, CLIPModel

>>> model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
>>> processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)

>>> outputs = model(**inputs)
>>> logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
>>> probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
```


### CLIP과 Flash Attention 2의 결합 [[combining-clip-and-flash-attention-2]]

먼저, 최신 버전의 Flash Attention 2를 설치해야 합니다.

```bash
pip install -U flash-attn --no-build-isolation
```

또한 Flash-Attention 2와 호환되는 하드웨어를 사용하는지 확인하세요. flash-attn 리포지토리의 공식 문서에서 이에 대해 더 자세히 알아볼 수 있습니다. 모델을 하프 프리시전(예: `torch.float16`)으로 로드하는 것도 잊지 마세요.

<Tip warning={true}>

작은 배치 크기의 경우 Flash Attention을 사용할 때 모델이 느려질 수 있습니다. 아래의 [Flash Attention과 SDPA를 사용한 예상 속도 향상](#Expected-speedups-with-Flash-Attention-and-SDPA) 섹션을 참고하여 적절한 어텐션 구현을 선택하세요.

</Tip>

Flash Attention 2를 사용하여 모델을 로드하고 실행하는 방법은 아래 코드를 참조하세요:

```python
>>> import torch
>>> import requests
>>> from PIL import Image

>>> from transformers import CLIPProcessor, CLIPModel

>>> device = "cuda"
>>> torch_dtype = torch.float16

>>> model = CLIPModel.from_pretrained(
...     "openai/clip-vit-base-patch32",
...     attn_implementation="flash_attention_2",
...     device_map=device,
...     torch_dtype=torch_dtype,
... )
>>> processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)
>>> inputs.to(device)

>>> with torch.no_grad():
...     with torch.autocast(device):
...         outputs = model(**inputs)

>>> logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
>>> probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
>>> print(probs)
tensor([[0.9946, 0.0052]], device='cuda:0', dtype=torch.float16)
```


### Scaled Dot Product Attention(SDPA)의 사용 [[using-scaled-dot-product-attention-sdpa]]

PyTorch는 `torch.nn.functional`의 일부로 기본 Scaled Dot-Product Attention(SDPA) 연산자를 포함하고 있습니다. 이 함수는 입력과 사용하는 하드웨어에 따라 여러 구현을 적용할 수 있습니다. 자세한 내용은 [공식 문서](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) 또는 [GPU 추론](https://huggingface.co/docs/transformers/main/en/perf_infer_gpu_one#pytorch-scaled-dot-product-attention) 페이지를 참조하세요.

SDPA는 `torch>=2.1.1`에서 구현이 가능한 경우 기본적으로 사용되지만, `attn_implementation="sdpa"`를 `from_pretrained()`에 설정하여 SDPA 사용을 명시적으로 요청할 수도 있습니다.

```python
from transformers import CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", torch_dtype=torch.float16, attn_implementation="sdpa")
```

최상의 속도 향상을 위해 모델을 하프 프리시전(예: `torch.float16` 또는 `torch.bfloat16`)으로 로드하는 것을 권장합니다.

### Flash Attention과 SDPA로 인한 예상 속도 향상 [[expected-speedups-with-flash-attention-and-sdpa]]

로컬 벤치마크(NVIDIA A10G, PyTorch 2.3.1+cu121)에서 `float16`을 사용하여 `"openai/clip-vit-large-patch14"` 체크포인트에 대한 추론 시 다음과 같은 속도 향상이 있었습니다([코드](https://gist.github.com/qubvel/ac691a54e54f9fae8144275f866a7ff8)):

#### CLIPTextModel [[cliptextmodel]]

|   Num text labels |   Eager (s/iter) |   FA2 (s/iter) |   FA2 speedup |   SDPA (s/iter) |   SDPA speedup |
|------------------:|-----------------:|---------------:|--------------:|----------------:|---------------:|
|                 4 |            0.009 |          0.012 |         0.737 |           0.007 |          1.269 |
|                16 |            0.009 |          0.014 |         0.659 |           0.008 |          1.187 |
|                32 |            0.018 |          0.021 |         0.862 |           0.016 |          1.142 |
|                64 |            0.034 |          0.034 |         1.001 |           0.03  |          1.163 |
|               128 |            0.063 |          0.058 |         1.09  |           0.054 |          1.174 |

![clip_text_model_viz_3](https://github.com/user-attachments/assets/e9826b43-4e66-4f4c-952b-af4d90bd38eb)

#### CLIPVisionModel [[clipvisionmodel]]

|   Image batch size |   Eager (s/iter) |   FA2 (s/iter) |   FA2 speedup |   SDPA (s/iter) |   SDPA speedup |
|-------------------:|-----------------:|---------------:|--------------:|----------------:|---------------:|
|                  1 |            0.016 |          0.013 |         1.247 |           0.012 |          1.318 |
|                  4 |            0.025 |          0.021 |         1.198 |           0.021 |          1.202 |
|                 16 |            0.093 |          0.075 |         1.234 |           0.075 |          1.24  |
|                 32 |            0.181 |          0.147 |         1.237 |           0.146 |          1.241 |

![clip_image_model_viz_3](https://github.com/user-attachments/assets/50a36206-e3b9-4adc-ac8e-926b8b071d63)

#### CLIPModel [[clipmodel]]

|   Image batch size |   Num text labels |   Eager (s/iter) |   FA2 (s/iter) |   FA2 speedup |   SDPA (s/iter) |   SDPA speedup |
|-------------------:|------------------:|-----------------:|---------------:|--------------:|----------------:|---------------:|
|                  1 |                 4 |            0.025 |          0.026 |         0.954 |           0.02  |          1.217 |
|                  1 |                16 |            0.026 |          0.028 |         0.918 |           0.02  |          1.287 |
|                  1 |                64 |            0.042 |          0.046 |         0.906 |           0.036 |          1.167 |
|                  4 |                 4 |            0.028 |          0.033 |         0.849 |           0.024 |          1.189 |
|                  4 |                16 |            0.034 |          0.035 |         0.955 |           0.029 |          1.169 |
|                  4 |                64 |            0.059 |          0.055 |         1.072 |           0.05  |          1.179 |
|                 16 |                 4 |            0.096 |          0.088 |         1.091 |           0.078 |          1.234 |
|                 16 |                16 |            0.102 |          0.09  |         1.129 |           0.083 |          1.224 |
|                 16 |                64 |            0.127 |          0.11  |         1.157 |           0.105 |          1.218 |
|                 32 |                 4 |            0.185 |          0.159 |         1.157 |           0.149 |          1.238 |
|                 32 |                16 |            0.19  |          0.162 |         1.177 |           0.154 |          1.233 |
|                 32 |                64 |            0.216 |          0.181 |         1.19  |           0.176 |          1.228 |

## 리소스 [[resources]]

🤗 및 커뮤니티(🌎로 표시) 리소스 목록은 CLIP 사용을 시작하는 데 도움을 줄 것입니다.

- [원격 감지(위성) 이미지와 캡션으로 CLIP 미세 조정하기](https://huggingface.co/blog/fine-tune-clip-rsicd), RSICD 데이터셋과 데이터 증강으로 인한 성능 변화를 비교한 CLIP 미세 조정에 대한 블로그 포스트입니다.
- 이 [예시 스크립트](https://github.com/huggingface/transformers/tree/main/examples/pytorch/contrastive-image-text)는 미리 학습된 비전 및 텍스트 인코더를 사용하여 CLIP 유사한 비전-텍스트 이중 인코더 모델을 훈련하는 방법을 보여줍니다 [COCO 데이터셋](https://cocodataset.org/#home).

<PipelineTag pipeline="image-to-text"/>

- 사전 학습된 CLIP을 사용하여 빔 서치로 이미지 캡셔닝 추론을 수행하는 방법에 대한 [노트북](https://colab.research.google.com/drive/1tuoAC5F4sC7qid56Z0ap-stR3rwdk0ZV?usp=sharing) 🌎

**이미지 검색**

- 사전 학습된 CLIP을 사용한 이미지 검색 및 MRR(Mean Reciprocal Rank) 점수 계산에 관한 [노트북](https://colab.research.google.com/drive/1bLVwVKpAndpEDHqjzxVPr_9nGrSbuOQd?usp=sharing) 🌎
- 이미지 검색 및 유사도 점수를 보여주는 [노트북](https://colab.research.google.com/github/deep-diver/image_search_with_natural_language/blob/main/notebooks/Image_Search_CLIP.ipynb) 🌎
- 다국어 CLIP을 사용하여 이미지와 텍스트를 동일한 벡터 공간으로 매핑하는 방법에 관한 [노트북](https://colab.research.google.com/drive/1xO-wC_m_GNzgjIBQ4a4znvQkvDoZJvH4?usp=sharing) 🌎
- [Unsplash](https://unsplash.com) 및 [TMDB](https://www.themoviedb.org/) 데이터셋을 사용하여 CLIP을 실행하는 방법에 관한 [노트북](https://colab.research.google.com/github/vivien000/clip-demo/blob/master/clip.ipynb#scrollTo=uzdFhRGqiWkR) 🌎

**설명 가능성**

- 입력 토큰과 이미지 세그먼트 간의 유사성을 시각화하는 방법에 관한 [노트북](https://colab.research.google.com/github/hila-chefer/Transformer-MM-Explainability/blob/main/CLIP_explainability.ipynb) 🌎

여기에 포함될 리소스를 제출하고 싶다면, 자유롭게 Pull Request를 열어 주시면 검토하겠습니다. 리소스는 기존 리소스를 중복하지 않고 새로운 무언가를 시연하는 것이 이상적입니다.

## CLIPConfig [[clipconfig]]

[[autodoc]] CLIPConfig
    - from_text_vision_configs

## CLIPTextConfig [[cliptextconfig]]

[[autodoc]] CLIPTextConfig

## CLIPVisionConfig [[clipvisionconfig]]

[[autodoc]] CLIPVisionConfig

## CLIPTokenizer [[cliptokenizer]]

[[autodoc]] CLIPTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## CLIPTokenizerFast [[cliptokenizerfast]]

[[autodoc]] CLIPTokenizerFast

## CLIPImageProcessor [[clipimageprocessor]]

[[autodoc]] CLIPImageProcessor
    - preprocess

## CLIPFeatureExtractor [[clipfeatureextractor]]

[[autodoc]] CLIPFeatureExtractor

## CLIPProcessor [[clipprocessor]]

[[autodoc]] CLIPProcessor

<frameworkcontent>
<pt>

## CLIPModel [[clipmodel]]

[[autodoc]] CLIPModel
    - forward
    - get_text_features
    - get_image_features

## CLIPTextModel [[cliptextmodel]]

[[autodoc]] CLIPTextModel
    - forward

## CLIPTextModelWithProjection [[cliptextmodelwithprojection]]

[[autodoc]] CLIPTextModelWithProjection
    - forward

## CLIPVisionModelWithProjection [[clipvisionmodelwithprojection]]

[[autodoc]] CLIPVisionModelWithProjection
    - forward

## CLIPVisionModel [[clipvisionmodel]]

[[autodoc]] CLIPVisionModel
    - forward

## CLIPForImageClassification [[clipforimageclassification]]

[[autodoc]] CLIPForImageClassification
    - forward

</pt>
<tf>

## TFCLIPModel [[tfclipmodel]]

[[autodoc]] TFCLIPModel
    - call
    - get_text_features
    - get_image_features

## TFCLIPTextModel [[tfcliptextmodel]]

[[autodoc]] TFCLIPTextModel
    - call

## TFCLIPVisionModel [[tfclipvisionmodel]]

[[autodoc]] TFCLIPVisionModel
    - call

</tf>
<jax>

## FlaxCLIPModel [[flaxclipmodel]]

[[autodoc]] FlaxCLIPModel
    - __call__
    - get_text_features
    - get_image_features

## FlaxCLIPTextModel [[flaxcliptextmodel]]

[[autodoc]] FlaxCLIPTextModel
    - __call__

## FlaxCLIPTextModelWithProjection [[flaxcliptextmodelwithprojection]]

[[autodoc]] FlaxCLIPTextModelWithProjection
    - __call__

## FlaxCLIPVisionModel [[flaxclipvisionmodel]]

[[autodoc]] FlaxCLIPVisionModel
    - __call__

</jax>
</frameworkcontent>
