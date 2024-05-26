import matplotlib.pyplot as plt
import pandas as pd

def main():
    data = pd.read_csv("data/final_results.csv")
    
    
    plt.plot(data["Accuracy"])
    plt.plot(data["Val_Accuracy"])
    plt.title('Model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(f"data/Data Exploration/Final Accuracy.jpg")
    plt.clf()
    # summarize history for loss
    plt.plot(data["Loss"])
    plt.plot(data["Val_Loss"])
    plt.title('CNN LSTM Model Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(f"data/Data Exploration/Final Loss.jpg")

if __name__ == "__main__":
    main()