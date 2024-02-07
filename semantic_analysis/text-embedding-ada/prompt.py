BACKGROUND = \
    "Parse the given neural network model name into the following categories:\n \
    - Architecture [A]: e.g, bert, albert, resnet\n \
    - Model size [S]: e.g, 50, 101, base, large, xxlarge\n \
    - Dataset [D]: e.g, squad, imagenet\n \
    - Dataset characteristic [C]: e.g. case, uncased, 1024-1024, 224\n \
    - Model version [V]: e.g, v1, v2\n \
    - Reuse method [F]: e.g, finetune, distill, fewshot\n \
    - Language [L]: e.g, en, english, chinese, arabic\n \
    - Task [T]: e.g, qa\n \
    - Training process [R]: e.g, pretrain, sparse\n \
    - Numer of layers [N]: e.g, L-12, L-24\n \
    - Number of heads [H]: e.g, H-12, H-256\n \
    - Number of parameters [P]: e.g, 100M, 8B\n \
    - Other [O]: If a portion of the model name cannot be classified into the above categories, classify it as other. For example, test, demo\n\
    Formatting Rules:\n\
    - Segment model names by hyphens or underscores or lowercase/uppercase (such as bertBase -> two segments: bert, Base).\n\
    - Output list length must match the number of segments in the name.\n\
    - For each segment, provide the top-3 possible categories with their corresponding confidence values.\n\
    - For multiple inputs, give line-by-line outputs.\n\
    - Output format: albert:(A,1.0)(O,0.1)(L,0.1),...\n\
    - No extra text in the output (No repeating inputs, No repeating inputs please!! No explanation needed).\n\
    - Don't put the input in your output!!\n\
    - Strictly follow the formatting style (may also use the examples as formatting reference).\n\
    Confidence Guidelines:\n\
    - 0.0: You think there is less than 1 percent this could be true\n\
    - 0.1 to 0.3: You can guess what the string means in the context of model name but is extremely unsure\n\
    - 0.3 to 0.6: You have some idea in what the string means but still uncertain about the exact meaning of it\n\
    - 0.6 to 0.9: You are confident in what the string means\n\
    - 1.0: You are over 99 percent sure of the meaning\n\
    Example of good naming:\n\
    Input: albert-base-v2\n\
    Output: albert:(A,1.0)(O,0.1)(L,0.1),base:(S,0.9)(O,0.2)(V,0.2),v2:(V,0.9)(S,0.2)(O,0.2)\n\
    Example of bad naming:\n\
    Input: random-model\n\
    Output: random:(O,0.4)(M,0.4)(F,0.3),model:(O,0.3)(A,0.3)(T,0.1)\n\
    Important Notes:\n\
    - The examples are provided to help you input the data and you should not strictly follow the example output even if the names are the same.\n\
    - You should determine the category base not only on the segment itself but also the context of the whole model name.\n\
    - Do not include the full input name in the output, only include the segmented components!\n\
    "
