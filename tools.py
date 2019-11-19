def convert_emotion_index_to_txt(emotion_index):
    if emotion_index == 0:
        return "happy"
    if emotion_index == 1:
        return "neutral"
    if emotion_index == 2:
        return "sad"
    if emotion_index == 3:
        return "hate"
    if emotion_index == 4:
        return "anger"


# Grouping the emotions into 5 classes
def emotion_converg(emotion):
    if emotion == "happiness" or emotion == "love" or emotion == "surprise" or emotion == "fun" or emotion == "enthusiasm":
        return '0'
    if emotion == "empty" or emotion == "relief" or emotion == "neutral":
        return '1'
    if emotion == "worry" or emotion == "sadness":
        return '2'
    if emotion == "hate" or emotion == "boredom":
        return '3'
    if emotion == "anger":
        return '4'

# function for reading text files
def read_txt_file(path):
    x = []
    y = []
    with open(path, 'r', encoding='utf-8') as input_file:
        for line in input_file.readlines():
            [sentence, emotion] = line.split(';')
            if len(sentence) == 0:
                continue
            x.append(sentence)
            y.append(emotion)
    return x, y