from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import os,pickle


def model_build(X_train,X_test,y_train,y_test):
    model = RandomForestClassifier()
    model.fit(X_train,y_train)
    
    y_pred = model.predict(X_test)
    
    report=classification_report(y_test,y_pred)
    
    os.makedirs('models',exist_ok=True)
    
    with open('models/model.pkl','wb') as f:
        pickle.dump(model,f)
    
    return report