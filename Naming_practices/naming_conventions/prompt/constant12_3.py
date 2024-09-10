BACKGROUND = \
    "Background: You are an assistant to categorize pre-trained model names. Parse the given neural network model name into one of the following categories: Implementation, Application and Task, Implementation with Application and Task, Other.\n\
    Definitions:\n\
        - Implementation: refers to the detailed configuration and design aspects of a model that shape its structural and operational framework, encompassing:\n\
            - Architecture [A]: The foundational structure or deisng of the deep learning model.\n\
            - Model Size [S]: The scale of the model, often in terms of the number of parameters or architectural unit.\n\
            - Dataset [D]: The specific dataset used for training or evaluation of the model.\n\
            - Model Versioning [V]: The release version of the model.\n\
        - Application and Task: Encompasses models designed for real-world applications and specific tasks they are built to perform. This inclues the following component:\n\
            - Language [L]: Specifies the natural language(s) for which the model is specifically trained, such as English(en or eng), Chinese(ch), or Arabic(ar), which indicates the linguistic capability and applicability of the model.\n\
            - Task [T]: The particular problem or application the model is designed to address.\n\
        - Implementation with Application and Task: For models that incorporate specific designs and are tailored for particular tasks or applications.\n\
        - Other: For model names that don't fit the above categories.\n\
    Here are the example names that fall under specific category:\n\
        - Implementation: whisper-large-v2-pt-v3 [A: Whisper, S: large, V: v2, D: PT dataset, L: v3], cross-encoder-umberto-stsb [A: Umberto (variant of BERT), S: cross-encoder size, V: unspecified, D: STSb dataset, L: unspecified], DIAL-BART0 [A: BART, S: initial version, V: 0, D: DIAL dataset, L: unspecified].\n\
        - Application and Task: garbage-classification [A: grabage, S: unspecified, V: unspecified, D: unspecified, L: unspecified, T: Classification], autonlp-swahili-sentiment-615517563 [A: autonlp, S: unspecified, V: unspecified, D: unspecified, L: Swahili, T: Sentiment Analysis], anime-ai-detect [A: unspecified, S: unspecified, V: unspecified, D: unspecified, L: unspecified, T: Object Detection]\n\
        - Implementation with Application and Task: s2t-small-mustc-en-de-st [A: Speech-to-Text architecture, S: small, D: MuST-C dataset, L: English-German, Task: Speech Translation], roberta-base-indonesian-1.5G-sentiment-analysis-smsa [A: RoBERTa, S: base, D: 1.5G Indonesian dataset, V: unspecified, Task: Sentiment Analysis], Llama-2-7b-chat-hf [A: LLAMA, S: 7 billion parameters, Task: Conversational AI], dfm-sentence-encoder-large-2 [A: dfm, S: large, Task: Encode Sentence]\n\
        - Other: indian-foods [Subject: Cuisine, Classification: Non-technical], potat1 [Subject: Unclear, Classification: Non-technical], CS182-DreamBooth-2-Object [Subject: Course-related or Project, Classification: Non-technical or Other]\n\
    Categorization Rules:\n\
        - If architecture [A] is unspecified, categorize as Application and Task or Other\n\
        - Non-Technical Terms: Not categorized as Implementation.\n\
        - Model/Architecture Names Only: Categorized as Implementation.\n\
        - Model/Architecture with Tasks/Applications: Categorized as Implementation with Application and Task.\n\
        - Task names (e.g. img2img, text2img, text-to-speech) are not Implementation\n\
        - If no technical terms (e.g. cafe-aesthetic, mit-indoor-scenes), classify as Other\n\
        - Categorized as Application and Task or Implementation with Application and Task if the model name consists of the following listed terms: Image-Text-to-Text, Visual Question Answering, Document Question Answering, Depth Estimation, Image Classification, Object Detection, Image Segmentation, Text-to-Image,  Image-to-Text, Image-to-Image, Image-to-Video, Unconditional Image Generation, Video Classification, Text-to-Video, Zero-Shot Image Classification, Mask Generation, Zero-Shot Object Detection, Text-to-3D, Image-to-3D, Image Feature Extraction, Text Classification,  Token Classification, Table Question Answering, Question Answering, Zero-Shot Classification, Translation, Summarization, Feature Extraction, Text Generation, Text2Text Generation, Fill-Mask, Sentence Similarity, Text-to-Speech, Text-to-Audio, Speech Recognition, Audio-to-Audio, Audio Classification, Voice Activity Detection, Zeroshot.\n\
        - Task Abbreviations: Recognized (e.g. automatic speech recognition (asr), text-to-image (text2img), visual question answering (vqa), document question answering (dqa or doc-qa), image-to-text (img2text or i2t), text-to-video (text2vid), text-to-speech (tts), and voice activity detection (vad), etc.).\n\
    Without additional text, Output from one of the categories: Implementation, Application and Task, Implementation with Application and Task, Other.\n\
    Example of good naming:\n\
    Input: promptcap-coco-vqa\n\
    Output: Implementation with Application and Task\n\
    "

'''
Unmatching Model Names
data2vec-vision-base
Prediction:  Implementation
Actual value:  Implementation with Application and Task
Output token 1: Implementation, logprobs: -4.3226137e-05, linear prob: 100.0
Output token 2: Application, logprobs: -10.062543, linear prob: 0.0

deformable-detr-with-box-refine
Prediction:  Implementation
Actual value:  Implementation with Application and Task
Output token 1: Implementation, logprobs: -9.74638e-05, linear prob: 99.99
Output token 2: Application, logprobs: -9.312597, linear prob: 0.01

clip-vit-base-patch32-ko
Prediction:  Implementation
Actual value:  Implementation with Application and Task
Output token 1: Implementation, logprobs: -1.735894e-05, linear prob: 100.0
Output token 2: Application, logprobs: -11.000017, linear prob: 0.0

data2vec-audio-base-960h
Prediction:  Implementation
Actual value:  Implementation with Application and Task
Output token 1: Implementation, logprobs: -2.3438328e-05, linear prob: 100.0
Output token 2: Application, logprobs: -10.687524, linear prob: 0.0

whisper-medium-ko-zeroth
Prediction:  Implementation
Actual value:  Implementation with Application and Task

whisper-small-zh
Prediction:  Implementation
Actual value:  Implementation with Application and Task
Output token 1: Implementation, logprobs: -0.00027051452, linear prob: 99.97
Output token 2: Application, logprobs: -8.219021, linear prob: 0.03


whisper-large-icelandic-62640-steps-967h
Prediction:  Implementation
Actual value:  Implementation with Application and Task
Output token 1: Implementation, logprobs: -0.004643723, linear prob: 99.54
Output token 2: Application, logprobs: -5.379644, linear prob: 0.46

text2img_vision_2.0
Prediction:  Implementation with Application and Task
Actual value:  Application and Task
Output token 1: Implementation, logprobs: -0.016045395, linear prob: 98.41
Output token 2: Application, logprobs: -4.1410456, linear prob: 1.59

damo-image-to-video
Prediction:  Application and Task
Actual value:  Implementation with Application and Task
Output token 1: Application, logprobs: -0.03365181, linear prob: 96.69
Output token 2: Implementation, logprobs: -3.4086518, linear prob: 3.31
'''