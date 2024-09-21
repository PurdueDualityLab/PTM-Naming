"""
This file contains the prompt for GPT.
"""

BACKGROUND = \
    "Parse the given neural network model name into the following categories:\n \
    - Architecture [A]: e.g, bert, albert, resnet\n \
    - Model size [S]: e.g, 50, 101, base, large, xxlarge\n \
    - Dataset [D]: e.g, squad, imagenet\n \
    - Dataset characteristic [C]: e.g. case, uncased, 1024-1024, 224\n \
    - Model version [V]: e.g, v1, v2, v1.1, 1.2, 1-1, v1-1, v2_1, V2-2\n \
    - Reuse method [F]: e.g, finetune, distill, fewshot, lora\n \
    - Language [L]: e.g, en, english, chinese, arabic\n \
    - Task or Application Goal [T]: e.g, qa, cls, fill-mask, \
    image-segmentation fake-news-detector, face-recognition, multilingual-sentiment-covid19, Chat-Bot-Batman, name_to_gender, emotion_text_classifier\n \
    - Training process [R]: e.g, pretrain, sparse\n \
    - Numer of layers [N]: e.g, L-12, L-24\n \
    - Number of heads [H]: e.g, H-12, H-256\n \
    - Number of parameters [P]: e.g, 100M, 8B\n \
    - Other [O]: If a portion of the model name cannot be classified into the above categories, classify it as other. For example, test, demo\n\
    Formatting Rules:\n\
    - For multiple inputs, give line-by-line outputs.\n\
    - No extra text in the output (No repeating inputs, No repeating inputs please!! No explanation needed).\n\
    - Don't put the input in your output!!\n\
    - Strictly follow the formatting style (may also use the examples as formatting reference).\n\
    Formatting Style Example:\n\
    Input: albert-base-v2\n\
    Output: albert-base-v2: A, S, V\n\
    Input: opus-mt-it-en\n\
    Output: opus-mt-it-en: A, D, L\n\
    - The order of the categories does not matter.\n\
    - The format is <model name> <category 1> <category 2> <category 3>...\n\
    - Your output should be a set of category that is contained in the model name.\n\
    "
