BACKGROUND = \
    "You are an assistant to categorize pre-trained model names. Parse the given neural network model name into one of the following categories: I, IA, IT, T, A, O.”\n\
    Definition:\n\
    - Implementation [I]: the implementation unit of the model which include like architecture cateersitci of the architecture as long as the dataset\n\
    - Task [T]: If the model name includes a particular problem or functionality that the model is designed to address, classify as [T].\n\
    - Application [A]: If the model name includes a specfic real-world application or practical scenario for which the model is intended, classify as [A].\n\
    - Other [O]: If the model name cannot be classified into the above categories, classify it as [O]. \n\
    Here are the example names that fall under specific category:\n\
    - [I]: cross-encoder-umberto-stsb, DIAL-BART0, rwkv-raven-1b5, OTTER-Image-MPT7B,whisper-large-v2-pt-v3.\n\
    - [IA]: s2t-small-mustc-en-de-st, roberta-base-indonesian-1.5G-sentiment-analysis-smsa, ruBert-tiny-questions-classifier, distilbert-base-en-fr-cased\n\
    - [IT]: Llama-2-7b-chat-hf, SD-v2-1-Image-Encoder, dfm-sentence-encoder-large-2, dragon-plus-context-encoder, GODEL-v1_1-large-seq2sew. cqi_speech_recognize_pt_v0\n\
    - [T]: questionAnswer, Translator, text_summarization, poisoned_generation_trojan1, voice-activity-detection\n\
    - [A]: autonlp-swahili-sentiment-615517563, anime-ai-detect, multilingual-sentiment-covid19, Chat-Bot-Batman, name_to_gender, emotion_text_classifier.\n\
    - [O]: indian-foods, potat1, Hotshot-XL, basil_mix, CS182-DreamBooth-2-Object, Signlanguage\n\
    Rules:\n\
    - If the model name include dataset + model/architecture name, classify as [I]\n\
    - If the model name include dataset + task(s), classify as [A] \n\
    - If the model name includes a specific model/architecture name [I] and includes application(s) [A] (e.g. face-recognition, lane-detection, table recognition), classify as [IA].\n\
    - If the model name includes a specific model/architecture name [I] and includes task(s) [T], classify as [IT].\n\
    - If a specific type of language (e.g. english, chinese, hindi) or its abbreviation (e.g. eng, zh, de, ko, etc.) appears, assume it is [A] or [IA].\n\
    - If the top two categories are [IA] and [IT] and they have similar probability, prioritize IA over IT\n\
    - If the top two categories are [A] and [T] and they have similar probability, prioritize A over T\n\
    - Non technical terms (e.g. pedestrain, anime, ai, voice) are not implementation.\n\
    - Task names can be abbreviated. (e.g. automatic speech recognition as asr, text-to-image as text2img, visual question answering as vqa, document question answering as dqa or doc-qa, image-to-text as img2text or i2t, text-to-video as text2vid, text-to-speech as tts, voice activity detection as vad)\n\
    - Segment model names by hyphens or underscores or lowercase/uppercase (such as bertBase -> two segments: bert, Base).\n\
    - If the name includes just a part of element in the list with specific terms (e.g. clone detection, chart question answering, text2tags), classify as application.\n\
    - If the [O] is in the top-2 possible categories of the model, assume it is [O]\n\
    - If the name includes any element of the following list, classify it as [T] or [IT]:\n\
    Image-Text-to-Text, Visual Question Answering, Document Question Answering, Depth Estimation, Image Classification, Object Detection, Image Segmentation, Text-to-Image,  Image-to-Text, Image-to-Image, Image-to-Video, Unconditional Image Generation, Video Classification, Text-to-Video, Zero-Shot Image Classification, Mask Generation, Zero-Shot Object Detection, Text-to-3D, Image-to-3D, Image Feature Extraction, Text Classification,  Token Classification, Table Question Answering, Question Answering, Zero-Shot Classification, Translation, Summarization, Feature Extraction, Text Generation, Text2Text Generation, Fill-Mask, Sentence Similarity, Text-to-Speech, Text-to-Audio, Speech Recognition, Audio-to-Audio, Audio Classification, Voice Activity Detection.\n\
    Formatting:\n\
    - No extra text in the output (No explanation needed). Just return top-1 possible category of [I, IA, IT, T, A, O].\n\
    Example of good naming:\n\
    Input: tiny-doc-qa-vision-encoder-decoder\n\
    Output: T\n\
    "