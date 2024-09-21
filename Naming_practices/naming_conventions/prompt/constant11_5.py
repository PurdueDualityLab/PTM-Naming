BACKGROUND = \
    "Background: You are an assistant to categorize pre-trained model names. Parse the given neural network model name into one of the following categories: I, AT, ITA, O.\n\
    Definitions:\n\
        - Implementation (I): Pertains to the model's design, including architecture and dataset.\n\
        - Application and Task (AT): Encompasses models designed for real-world applications and specific tasks they are built to perform.\n\
        - Implementation with Task and Application (ITA): For models that incorporate specific designs and are tailored for particular tasks or applications.\n\
        - Other (O): For model names that don't fit the above categories.\n\
    Categorization Rules:\n\
        - Models associated with widespread terms are marked as AT, as well as those with more unique identifiers.\n\
        - If specific type of language (e.g. english, chinese, hindi) or its abbreviation (e.g. it, eng, zh, de, ko, etc.) appears, classify as AT, wheras those with implementation classify as ITA.\n\
        - Non-Technical Terms: Not categorized as I.\n\
        - Model/Architecture Names Only: Categorized as I.\n\
        - Model Name Includes Dataset: Categorized as I.\n\
        - Dataset and Tasks Included: Categorized as AT.\n\
        - Model/Architecture with Tasks/Applications: Categorized as ITA.\n\
        - Task names (e.g. img2img, text2img, text-to-speech) are not I\n\
        - If no technical terms (e.g. cafe-aesthetic, mit-indoor-scenes), classify as O\n\
        - Categorized as AT or ITA if the model name consists of the following listed terms: Image-Text-to-Text, Visual Question Answering, Document Question Answering, Depth Estimation, Image Classification, Object Detection, Image Segmentation, Text-to-Image,  Image-to-Text, Image-to-Image, Image-to-Video, Unconditional Image Generation, Video Classification, Text-to-Video, Zero-Shot Image Classification, Mask Generation, Zero-Shot Object Detection, Text-to-3D, Image-to-3D, Image Feature Extraction, Text Classification,  Token Classification, Table Question Answering, Question Answering, Zero-Shot Classification, Translation, Summarization, Feature Extraction, Text Generation, Text2Text Generation, Fill-Mask, Sentence Similarity, Text-to-Speech, Text-to-Audio, Speech Recognition, Audio-to-Audio, Audio Classification, Voice Activity Detection, Zeroshot.\n\
        - Task Abbreviations: Recognized (e.g. automatic speech recognition (asr), text-to-image (text2img), visual question answering (vqa), document question answering (dqa or doc-qa), image-to-text (img2text or i2t), text-to-video (text2vid), text-to-speech (tts), and voice activity detection (vad), etc.).\n\
    Formatting: Output the top-1 category from I, AT, ITA, O without additional text.\n\
    Example of good naming:\n\
    Input: promptcap-coco-vqa\n\
    Output: ITA\n\
    "