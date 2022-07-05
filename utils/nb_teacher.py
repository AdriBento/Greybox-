from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from utils import make_x_y_data
import numpy as np
import pickle

train_path = '/home/abennetot/dataset/OD-MonuMAI/MonuMAI_dataset/train.txt'
val_path = '/home/abennetot/dataset/OD-MonuMAI/MonuMAI_dataset/val.txt'
test_path = '/home/abennetot/dataset/OD-MonuMAI/MonuMAI_dataset/test.txt'

#Parse data
train_images_path, train_x, train_y = make_x_y_data(train_path)
val_images_path, val_x, val_y = make_x_y_data(val_path)
test_images_path, test_x, test_y = make_x_y_data(test_path)

alpha = 1

teacher_nb = MultinomialNB(alpha=alpha, fit_prior=False) #alpha=0.02 #False
teacher_nb.fit(train_x, train_y)

#Use the Naive Bayesian model on the test/total dataset. Predict the target
pred_y_train_nb = teacher_nb.predict(train_x)
pred_y_val_nb = teacher_nb.predict(val_x)
pred_y_test_nb = teacher_nb.predict(test_x)

#Use the Naive Bayesian model on the test/total dataset. Predict the proba
pred_proba_y_train_nb = teacher_nb.predict_proba(train_x)
pred_proba_y_val_nb = teacher_nb.predict_proba(val_x)
pred_proba_y_test_nb = teacher_nb.predict_proba(test_x)

max_prob_train, max_prob_val, max_prob_test = [], [], []

for i in range(0, len(pred_proba_y_train_nb)):
    max_prob_train.append(np.max(pred_proba_y_train_nb[i]))

for i in range(0, len(pred_proba_y_val_nb)):
    max_prob_val.append(np.max(pred_proba_y_val_nb[i]))

for i in range(0, len(pred_proba_y_test_nb)):
    max_prob_test.append(np.max(pred_proba_y_test_nb[i]))



#Print accuracy of the Bayesian model
print("Accuracy Train of NB Model with alpha = ", alpha, ":" , (np.sum(pred_y_train_nb==train_y)/len(train_y)))
print("Accuracy Val of NB Model with alpha = ", alpha, ":" , (np.sum(pred_y_val_nb==val_y)/len(val_y)))
print("Accuracy Test of NB Model with alpha = ", alpha, ":" , (np.sum(pred_y_test_nb==test_y)/len(test_y)))

print("Mean Train Confidence of NB Model with alpha = ", alpha, ":" , np.mean(max_prob_train))
print("Mean Val Confidence of NB Model with alpha = ", alpha, ":" , np.mean(max_prob_val))
print("Mean Test Confidence of NB Model with alpha = ", alpha, ":" , np.mean(max_prob_test))

print("###############################################################")

alpha = 32

teacher_nb = MultinomialNB(alpha=alpha, fit_prior=False) #alpha=0.02 #False
teacher_nb.fit(train_x, train_y)

#Use the Naive Bayesian model on the test/total dataset. Predict the target
pred_y_train_nb = teacher_nb.predict(train_x)
pred_y_val_nb = teacher_nb.predict(val_x)
pred_y_test_nb = teacher_nb.predict(test_x)

#Use the Naive Bayesian model on the test/total dataset. Predict the proba
pred_proba_y_train_nb = teacher_nb.predict_proba(train_x)
pred_proba_y_val_nb = teacher_nb.predict_proba(val_x)
pred_proba_y_test_nb = teacher_nb.predict_proba(test_x)

max_prob_train, max_prob_val, max_prob_test = [], [], []

for i in range(0, len(pred_proba_y_train_nb)):
    max_prob_train.append(np.max(pred_proba_y_train_nb[i]))

for i in range(0, len(pred_proba_y_val_nb)):
    max_prob_val.append(np.max(pred_proba_y_val_nb[i]))

for i in range(0, len(pred_proba_y_test_nb)):
    max_prob_test.append(np.max(pred_proba_y_test_nb[i]))



#Print accuracy of the Bayesian model
print("Accuracy Train of NB Model with alpha = ", alpha, ":" , (np.sum(pred_y_train_nb==train_y)/len(train_y)))
print("Accuracy Val of NB Model with alpha = ", alpha, ":" , (np.sum(pred_y_val_nb==val_y)/len(val_y)))
print("Accuracy Test of NB Model with alpha = ", alpha, ":" , (np.sum(pred_y_test_nb==test_y)/len(test_y)))

print("Mean Train Confidence of NB Model with alpha = ", alpha, ":" , np.mean(max_prob_train))
print("Mean Val Confidence of NB Model with alpha = ", alpha, ":" , np.mean(max_prob_val))
print("Mean Test Confidence of NB Model with alpha = ", alpha, ":" , np.mean(max_prob_test))

print("###############################################################")
"""
lin_reg = LinearRegression().fit(train_x, train_y)
#Use the regression model on the test/total dataset. Predict the target
pred_y_train_linr = lin_reg.predict(train_x)
pred_y_val_linr = lin_reg.predict(val_x)
pred_y_test_linr = lin_reg.predict(test_x)

print("Linear Regression score Train", lin_reg.score(train_x, train_y))
print("Linear Regression score Val", lin_reg.score(val_x, val_y))
print("Linear Regression score Test", lin_reg.score(test_x, test_y))

max_prob_train, max_prob_val, max_prob_test = [], [], []


print("###############################################################")
"""


log_reg = LogisticRegression().fit(train_x, train_y)
#Use the regression model on the test/total dataset. Predict the target
pred_y_train_logr = log_reg.predict(train_x)
pred_y_val_logr = log_reg.predict(val_x)
pred_y_test_logr = log_reg.predict(test_x)

print("Logistic Regression score Train", log_reg.score(train_x, train_y))
print("Logistic Regression score Val", log_reg.score(val_x, val_y))
print("Logistic Regression score Test", log_reg.score(test_x, test_y))

#Print accuracy of the Bayesian model
print("Accuracy Train of Logistic Regression = ", (np.sum(pred_y_train_logr==train_y)/len(train_y)))
print("Accuracy Val of Logistic Regression = " , (np.sum(pred_y_val_logr==val_y)/len(val_y)))
print("Accuracy Test of Logistic Regression = ", (np.sum(pred_y_test_logr==test_y)/len(test_y)))

pred_proba_y_train_logr = log_reg.predict_proba(train_x)
pred_proba_y_val_logr = log_reg.predict_proba(val_x)
pred_proba_y_test_logr = log_reg.predict_proba(test_x)

max_prob_train, max_prob_val, max_prob_test = [], [], []

for i in range(0, len(pred_proba_y_train_logr)):
    max_prob_train.append(np.max(pred_proba_y_train_logr[i]))

for i in range(0, len(pred_proba_y_val_logr)):
    max_prob_val.append(np.max(pred_proba_y_val_logr[i]))

for i in range(0, len(pred_proba_y_test_logr)):
    max_prob_test.append(np.max(pred_proba_y_test_logr[i]))

print("Mean Train Confidence of Logistic Reg Model = ", np.mean(max_prob_train))
print("Mean Val Confidence of Logistic Reg Model  = ", np.mean(max_prob_val))
print("Mean Test Confidence of Logistic Reg Model  = ", np.mean(max_prob_test))


#Uncomment to save
#with open('teacher/nb_teacher_alpha_31.pkl', 'wb') as f:
#   pickle.dump(teacher_nb, f)
#   print("NB Classifier saved")

#with open('teacher/logreg_teacher.pkl', 'wb') as f:
#   pickle.dump(log_reg, f)
#   print("Log Reg Classifier saved")