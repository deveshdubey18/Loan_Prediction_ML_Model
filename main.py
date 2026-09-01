from src.loan_prediction_classification_model.data_ingestion import data_ingestion
from src.loan_prediction_classification_model.data_preprocessing import preprocessing
from src.loan_prediction_classification_model.model_building import model_build
from src.loan_prediction_classification_model.model_cluster import model_cluster

def main():
    df = data_ingestion()
    print(df.shape)
    
    X_train,X_test,y_train,y_test = preprocessing(df)
    
    report = model_build(X_train,X_test,y_train,y_test)
    print(report)

    kmeans= model_cluster(X_train,X_test)
    print(kmeans)

main()
