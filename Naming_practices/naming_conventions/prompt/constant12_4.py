BACKGROUND = \
    "Background: You are an assistant to categorize pre-trained model names. Parse the given neural network model name into one of the following categories: Implementation, Application and Task, Implementation with Application and Task, Other.\n\
    Definitions:\n\
        - Implementation: refers to the detailed configuration and design aspects of a model that shape its structural and operational framework, encompassing:\n\
            - Architecture [A]: The foundational structure or deisng of the deep learning model.\n\
            - Model Size [S]: The scale of the model, often in terms of the number of parameters or architectural unit.\n\
            - Dataset [D]: The specific dataset used for training or evaluation of the model.\n\
            - Model Versioning [V]: The release version of the model.\n\
        - Application and Task: Encompasses models designed for real-world applications and specific tasks they are built to perform. This inclues the following component:\n\
            - Application [AP]: The real-world use the model is designed for.\n\
            - Language [L]: Specifies the natural language(s) for which the model is specifically trained, such as English(en or eng), Chinese(ch), or Arabic(ar), which indicates the linguistic capability and applicability of the model.\n\
            - Task [T]: The particular problem or application the model is designed to address.\n\
        - Implementation with Application and Task: For models that incorporate specific designs and are tailored for particular tasks or applications.\n\
        - Other: For model names that don't fit the above categories.\n\
    Here are the example names that fall under specific category:\n\
        - Implementation: text2img_vision_2.0 [A: unspecified, S: unspecified, V: 2.0, D: unspecified, L: unspecified], whisper-large-v2-pt-v3 [A: Whisper, S: large, V: v2, D: PT dataset, L: unspecified], cross-encoder-umberto-stsb [A: Umberto (variant of BERT), S: unspecified, V: unspecified, D: STSb dataset, L: unspecified], DIAL-BART0 [A: BART, S: unspecified, V: 0, D: DIAL dataset, L: unspecified].\n\
        - Application and Task: text2img_vision_2.0 [AP: Image Generation, T: Text-to-Image], garbage-classification [AP: Waste Managemetn, T: Classification], autonlp-swahili-sentiment-615517563 [AP: NLP, L: Swahili, T: Sentiment Analysis], anime-ai-detect [AP: Media, L: unspecified, T: Object Detection], querygenerator [AP: Query Generation, L: unspecified, T: Text Generation]\n\
        - Implementation with Application and Task: GODEL-v1_1-large-seq2seq [A: GODEL, S: large, V: v1_1, Task: Sequence to Sequence], s2t-small-mustc-en-de-st [A: Speech-to-Text architecture, S: small, D: MuST-C dataset, L: English-German, Task: Speech Translation], roberta-base-indonesian-1.5G-sentiment-analysis-smsa [A: RoBERTa, S: base, D: unspecified, V: unspecified, Task: Sentiment Analysis, L: Indonesian], Llama-2-7b-chat-hf [A: LLAMA, S: 7 billion parameters, Task: Conversational AI], dfm-sentence-encoder-large-2 [A: dfm, S: large, Task: Encode Sentence]\n\
        - Other: indian-foods [Subject: Cuisine, Classification: Non-technical], potat1 [Subject: Unclear, Classification: Non-technical], CS182-DreamBooth-2-Object [Subject: Course-related or Project, Classification: Non-technical or Other]\n\
    Categorization Rules:\n\
        - Models primarily identified by language [L] without detailed architectural specifics are categorized as Application and Task.\n\
        - Unspecified architecture [A]: categorized as Application and Task, or Other\n\
        - Non-Technical Terms: Not categorized as Implementation.\n\
        - Model/Architecture Names Only: Categorized as Implementation.\n\
        - Models with names indicating both an architectural framework and a task-specific orientation should be considered under Implementation with Application and Task, especially when such tasks imply a direct application.\n\
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
There are a lot of wrong predictions (I and AT) although it's supposed to predict as IAT.
'''