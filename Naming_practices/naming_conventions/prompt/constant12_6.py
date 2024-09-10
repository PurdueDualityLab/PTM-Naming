BACKGROUND = \
    "Background: As an assistant, your task is to categorize pre-trained model names into one of the following categories based on their design, application, and task orientation: Implementation, Application and Task, Implementation with Application and Task, Other.\n\
    Definitions:\n\
        - Implementation: The detailed configuration and design aspects of a model that shape its structural and operational framework, encompassing:\n\
            - Architecture [A]: The foundational structure or deisng of the deep learning model.\n\
            - Model Size [S]: The scale of the model, often in terms of the number of parameters or architectural unit.\n\
            - Dataset [D]: The specific dataset used for training or evaluation of the model.\n\
            - Model Versioning [V]: The release version of the model.\n\
        - Application and Task: Models designed for real-world applications and specific tasks they are built to perform. This inclues the following component:\n\
            - Application [AP]: The real-world use the model is designed for.\n\
            - Task [T]: The particular problem or application the model is designed to address.\n\
            - Language [L]: Specifies the model's application for language tasks, such as translation (e.g., English to Spanish, en-kr) or multilingual capabilities, rather than the dataset's language.\n\
        - Implementation with Application and Task: Models that include both their architectural/technical design and are explicitly named for a particular application or task,\n\
        - Other: For model names that don't fit the above categories.\n\
    Here are the example names that fall under specific category. For any model names listed below, if components such as Architecture (A), Model Size (S), Dataset (D), Model Versioning (V), or Language (L) are not explicitly mentioned, assume they are unspecified:\n\
        - Implementation: distilbert-portuguese-cased [A: Distilbert, D: Portuguese dataset], whisper-large-v2-pt-v3 [A: Whisper, S: large, V: v2, D: PT dataset], cross-encoder-umberto-stsb [A: Umberto (variant of BERT), D: STSb dataset], DIAL-BART0 [A: BART, V: 0, D: DIAL dataset], whisper-medium-zeroth_korean [A: Whisper, S: medium, D: Zeroth Korean dataset].\n\
        - Application and Task: sentiment_analysis_IT [AP: NLP, T: Sentiment Analysis, L: Italian], autonlp-swahili-sentiment-615517563 [AP: NLP, T: Sentiment Analysis, L: Swahili], text2img_vision_2.0 [AP: Image Generation, T: Text-to-Image], garbage-classification [AP: Waste Management, T: Classification], anime-ai-detect [AP: Media, T: Object Detection], querygenerator [AP: Query Generation, T: Text Generation].\n\
        - Implementation with Application and Task: GODEL-v1_1-large-seq2seq [A: GODEL, S: large, V: v1_1, T: Sequence to Sequence], s2t-small-mustc-en-de-st [A: Speech-to-Text architecture, S: small, D: MuST-C dataset, L: English-German, T: Speech Translation], roberta-base-indonesian-1.5G-sentiment-analysis-smsa [A: RoBERTa, S: base, T: Sentiment Analysis, D: Indonesian dataset], Llama-2-7b-chat-hf [A: LLAMA, S: 7 billion parameters, T: Conversational AI], dfm-sentence-encoder-large-2 [A: dfm, S: large, T: Encode Sentence].\n\
        - Other: indian-foods [Subject: Cuisine, Classification: Non-technical], potat1 [Subject: Unclear, Classification: Non-technical], CS182-DreamBooth-2-Object [Subject: Course-related or Project, Classification: Non-technical or Other]\n\
    Categorization Rules:\n\
        - Models named with a language indicating a specific dataset rather than a translation task are categorized under Implementation.\n\
        - Models named for specific language translation or application tasks (e.g., English to Spanish translation) should be categorized under Application and Task.\n\
        - If a model's name suggests both an architectural design and a specific application or task, including language translation applications, classify it under Implementation with Application and Task.\n\
        - Unspecified architecture [A]: categorized as Application and Task, or Other\n\
        - Non-Technical Terms: Not categorized as Implementation.\n\
        - Model/Architecture Names Only: Categorized as Implementation.\n\
        - Models with names indicating both an architectural framework and a task-specific orientation should be considered under Implementation with Application and Task, especially when such tasks imply a direct application.\n\
        - If no technical terms (e.g. cafe-aesthetic, mit-indoor-scenes), classify as Other\n\
        - Categorized as Application and Task or Implementation with Application and Task if the model name consists of the following listed terms: Image-Text-to-Text, Visual Question Answering, Document Question Answering, Depth Estimation, Image Classification, Object Detection, Image Segmentation, Text-to-Image,  Image-to-Text, Image-to-Image, Image-to-Video, Unconditional Image Generation, Video Classification, Text-to-Video, Zero-Shot Image Classification, Mask Generation, Zero-Shot Object Detection, Text-to-3D, Image-to-3D, Image Feature Extraction, Text Classification,  Token Classification, Table Question Answering, Question Answering, Zero-Shot Classification, Translation, Summarization, Feature Extraction, Text Generation, Text2Text Generation, Fill-Mask, Sentence Similarity, Text-to-Speech, Text-to-Audio, Speech Recognition, Audio-to-Audio, Audio Classification, Voice Activity Detection, Zeroshot.\n\
        - Task Abbreviations: Recognized (e.g. automatic speech recognition (asr), text-to-image (text2img), visual question answering (vqa), document question answering (dqa or doc-qa), image-to-text (img2text or i2t), text-to-video (text2vid), text-to-speech (tts), and voice activity detection (vad), etc.).\n\
    Without additional text, Output from one of the categories: Implementation, Application and Task, Implementation with Application and Task, Other.\n\
    Example of good naming:\n\
    Input: promptcap-coco-vqa\n\
    Output: Implementation with Application and Task\n\
    "