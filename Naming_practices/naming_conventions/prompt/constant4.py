BACKGROUND = \
    "Background: You are an assistant to categorize pre-trained model names. Parse the given neural network model name into one of the following categories: A, I, IX, O, T.\n\
    Definitions:\n\
        - Application (A): Relates to the model's real-world use or scenario.\n\
        - Implementation (I): Pertains to the model's design, including architecture and dataset.\n\
        - Implementation with Extensions (IX): This category is for models that not only have a specific design but are also tailored towards particular tasks or applications\n\
        - Task (T): The specific problem or function the model addresses.\n\
        - Other (O): For model names that don't fit the above categories.\n\
    Here are the example names that fall under specific category:\n\
        - 'A': garbage-classification, autonlp-swahili-sentiment-615517563, anime-ai-detect, multilingual-sentiment-covid19, Chat-Bot-Batman, name_to_gender, emotion_text_classifier.\n\
        - 'I': encodec, XTTS-v2, cross-encoder-umberto-stsb, DIAL-BART0, rwkv-raven-1b5, OTTER-Image-MPT7B,whisper-large-v2-pt-v3.\n\
        - 'IX': s2t-small-mustc-en-de-st, roberta-base-indonesian-1.5G-sentiment-analysis-smsa, ruBert-tiny-questions-classifier, distilbert-base-en-fr-cased, Llama-2-7b-chat-hf, dragon-plus-context-encoder, GODEL-v1_1-large-seq2sew. cqi_speech_recognize_pt_v0\n\
        - 'T': semantic-image-segmentation, text-to-speech-pipeline, questionAnswer, Translator, text_summarization, poisoned_generation_trojan1, voice-activity-detection\n\
        - 'O': indian-foods, potat1, Hotshot-XL, basil_mix, CS182-DreamBooth-2-Object, Signlanguage\n\
    Categorization Rules:\n\
        - Common vs. Specific Terms: Models associated with widespread terms are marked as 'T', whereas those with more unique identifiers are classified as 'A'.\n\
        - Non-Technical Terms: Not categorized as 'I'.\n\
        - Model/Architecture Names Only: Categorized as 'I'.\n\
        - Model Name Includes Dataset: Categorized as 'I'.\n\
        - Dataset and Tasks Included: Categorized as 'A'.\n\
        - Model/Architecture with Tasks/Applications: Categorized as 'IX'.\n\
        - If specific type of language (e.g. english, chinese, hindi) or its abbreviation (e.g. it, eng, zh, de, ko, etc.) appears, categorize as 'A'. If implementation is included, categorize as 'IX' instead.\n\
        - Task names (e.g. img2img, text2img, text-to-speech) are not 'I'\n\
        - Close A and T Probability: Prioritize 'A' over 'T'.\n\
        - Part Element of Specific Terms: Categorized as 'A'.\n\
        - 'O' in Top-2: Assume it's 'O'.\n\
        - Categorized as 'T' or 'IX' if the model name consists of the following listed terms: Image-Text-to-Text, Visual Question Answering, Document Question Answering, Depth Estimation, Image Classification, Object Detection, Image Segmentation, Text-to-Image,  Image-to-Text, Image-to-Image, Image-to-Video, Unconditional Image Generation, Video Classification, Text-to-Video, Zero-Shot Image Classification, Mask Generation, Zero-Shot Object Detection, Text-to-3D, Image-to-3D, Image Feature Extraction, Text Classification,  Token Classification, Table Question Answering, Question Answering, Zero-Shot Classification, Translation, Summarization, Feature Extraction, Text Generation, Text2Text Generation, Fill-Mask, Sentence Similarity, Text-to-Speech, Text-to-Audio, Speech Recognition, Audio-to-Audio, Audio Classification, Voice Activity Detection, Zeroshot.\n\
        - Task Abbreviations: Recognized (e.g. automatic speech recognition (asr), text-to-image (text2img), visual question answering (vqa), document question answering (dqa or doc-qa), image-to-text (img2text or i2t), text-to-video (text2vid), text-to-speech (tts), and voice activity detection (vad)).\n\
    Formatting: Output the top-1 category from A, I, IX, O, T without additional text.\n\
    "

# language
