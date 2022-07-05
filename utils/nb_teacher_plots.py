from sklearn.naive_bayes import MultinomialNB
from utils import make_x_y_data
import numpy as np
import matplotlib.pyplot as plt
import pickle

train_path = '/home/abennetot/dataset/OD-MonuMAI/MonuMAI_dataset/train.txt'
val_path = '/home/abennetot/dataset/OD-MonuMAI/MonuMAI_dataset/val.txt'
test_path = '/home/abennetot/dataset/OD-MonuMAI/MonuMAI_dataset/test.txt'

#Parse data
train_images_path, train_x, train_y = make_x_y_data(train_path)
val_images_path, val_x, val_y = make_x_y_data(val_path)
test_images_path, test_x, test_y = make_x_y_data(test_path)


for i in range(1, 50, 1):
    alpha = i
    teacher_nb = MultinomialNB(alpha=alpha, fit_prior=False)
    teacher_nb.fit(train_x, train_y)

    #Use the Naive Bayesian model on the test/total dataset. Predict the target
    pred_y_train_nb = teacher_nb.predict(train_x)
    pred_y_val_nb = teacher_nb.predict(val_x)
    pred_y_test_nb = teacher_nb.predict(test_x)

    #Use the Naive Bayesian model on the test/total dataset. Predict the proba
    pred_proba_y_train_nb = teacher_nb.predict_proba(train_x)
    pred_proba_y_val_nb = teacher_nb.predict_proba(val_x)
    pred_proba_y_test_nb = teacher_nb.predict_proba(test_x)

    #Print accuracy of the Bayesian model
    """
    print("Accuracy Train of NB Model", (np.sum(pred_y_train_nb==train_y)/len(train_y)))
    print("Accuracy Val of NB Model", (np.sum(pred_y_val_nb==val_y)/len(val_y)))
    print("Accuracy Test of NB Model", (np.sum(pred_y_test_nb==test_y)/len(test_y)))
    
    print("NO SOFTMAX")
    print(pred_proba_y_test_nb[0][0])
    print(pred_proba_y_test_nb[0][1])
    print(pred_proba_y_test_nb[0][2])
    print(pred_proba_y_test_nb[0][3])
    """
    #legend =
    top_proba = []
    acc_model = (np.sum(pred_y_train_nb==train_y)/len(train_y))
    for i in range (0, len(pred_proba_y_train_nb)):
        top_proba.append(pred_proba_y_train_nb[i][np.argmax(pred_proba_y_train_nb[i])])


    mean_top_proba = np.mean(top_proba)
    print("ALPHA IS", alpha)
    print("AVG CONFIDENCE IS", mean_top_proba)
    print("ACCURACY IS",acc_model)

    bins = [i/100 for i in range(0, 100, 1)]

    a = np.hstack(top_proba)
    plt.hist(a, bins=bins)
    plt.axvline(x=mean_top_proba, ymin=0, ymax=1, linestyle="--", color="black")
    plt.text(mean_top_proba-0.01, 25, 'Avg Confidence', ha='center', va='center',rotation='vertical', backgroundcolor='none')
    plt.axvline(x=acc_model, ymin=0, ymax=1, linestyle="--", color="red")
    plt.text(acc_model-0.01, 25, 'Accuracy', ha='center', va='center',rotation='vertical', backgroundcolor='none')
    plt.xticks(np.arange(0, 1, step=0.05))
    plt.title(("Alpha", alpha, "Avg Confidence", mean_top_proba, "Acc", acc_model))
    plt.get_current_fig_manager().full_screen_toggle()
    plt.show()

#Uncomment to save
#with open('teacher/nb_teacher_93_acc_on_gt.pkl', 'wb') as f:
#   pickle.dump(teacher_nb, f)
#   print("NB Classifier saved")