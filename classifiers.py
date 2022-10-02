import pickle

filename_ENG = "E:/Tesis/BackEnd-Integration/src/calculator/predictUser/test_RandomForestClassifier_ENG.sav"

filename_ESP = "E:/Tesis/BackEnd-Integration/src/calculator/predictUser/test_RandomForestClassifier_ESP.sav"

load_ENGmodel = pickle.load(open(filename_ENG, 'rb'))

load_ESPmodel = pickle.load(open(filename_ESP, 'rb'))

