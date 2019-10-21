import argparse
import pickle

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--data", required=False,
                default=[2, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 1, 3, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2,
                         0, 18],
                help="path to input dataset")
args = vars(ap.parse_args())

with open('RNN-model.model', 'rb') as f:
    rf = pickle.load(f)


if(rf.predict([args["data"]])==1):
    print("Psoriasis")
elif(rf.predict([args["data"]])==2):
    print("seboreic dermatitis")
elif(rf.predict([args["data"]])==3):
    print("lichen planus")
elif(rf.predict([args["data"]])==4):
    print("pityriasis rosea")
elif(rf.predict([args["data"]])==5):
    print("cronic dermatitis")
elif(rf.predict([args["data"]])==6):
    print("pityriasis rubra pilaris")

# print("pred: " + str(rf.predict([args["data"]])))