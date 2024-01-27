import pickle
import nltk
import TranslaterLogic

translatorPic = open("trigramTrans.pickle", "rb")
translator = pickle.load(translatorPic)

def triGramTranslate(sentence):
    sentence_romanized=sentence.split(" ")
    translation = ""
    translated = translator.tag(nltk.word_tokenize(sentence.lower()))
    #print(translated)
    i=-1
    for word, trans in translated:
        i+=1
        if trans in ('NNN'):
            translation = translation + str(TranslaterLogic.convertText(str(sentence_romanized[i])) + " ")
        else:
            translation = translation + str(trans + " ")
    return translation

