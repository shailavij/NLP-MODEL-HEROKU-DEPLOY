
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB 
#from sklearn.externals import joblib
 
df= pd.read_csv("spam.csv",encoding="latin-1")
df.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1,inplace=True)

# features and columns
df['label']=df['class'].map({'ham': 0,'spam':1})
X=df['message']
y=df['label']

# Extract featuer with Countvectorizer

cv=CountVectorizer()
X=cv.fit_transform(X)  # Fit the data in countvectorise

with open('transform','wb') as f:
    pickle.dump(cv,f)



from sklearn.model_selection import train_test_split
X_train,x_test,Y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=42)

# Navies bayes classifier
clf= MultinomialNB()
clf.fit(X_train,Y_train)
clf.score(x_test,y_test)

with open('nlp_model','wb') as f:
    pickle.dump(clf,f)








 
