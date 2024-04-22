import pandas as pd

data = pd.read_csv('Social_Network_Ads.csv')

# identifying the target(y) and feature(x) variables
y = data['Purchased']
x = data[['Age','EstimatedSalary']]

# dividing 25% of data to test and 75% of data to train
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state = 42, test_size = .25)

# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier(n_estimators =20,max_depth = 20, min_samples_split = 3,criterion = 'entropy' )
rf_model = rf_clf.fit(x_train,y_train)

# pickle the trained model
import pickle
with open('model.pkl','wb') as model_file:
    pickle.dump(rf_clf,model_file)