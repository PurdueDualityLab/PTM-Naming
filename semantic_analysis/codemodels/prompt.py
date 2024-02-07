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
    - Other [O]\n\
    Formatting Rules:\n\
    - Segment model names by hyphens or underscores or lowercase/uppercase (such as bertBase -> two segments: bert, Base).\n\
    - Output list length must match the number of segments in the name.\n\
    - For each segment, provide the top-3 possible categories with their corresponding confidence values.\n\
    - For multiple inputs, give line-by-line outputs.\n\
    - Output format: albert:(A,1.0)(O,0.1)(L,0.1),...\n\
    - No extra text in the output (No repeating inputs, No repeating inputs please!! No explanation needed).\n\
    - Don't put the input in your output!!\n\
    - Strictly follow the formatting style (may also use the examples as formatting reference).\n\
    Example of good naming:\n\
    Input: albert-base-v2\n\
    Output: albert:(A,1.0)(O,0.1)(L,0.1),base:(S,0.9)(O,0.2)(V,0.2),v2:(V,0.9)(S,0.2)(O,0.2)\n\
    Example of bad naming:\n\
    Input: random-model\n\
    Output: random:(O,0.4)(M,0.4)(F,0.3),model:(O,0.3)(A,0.3)(T,0.1)\n\
    "
