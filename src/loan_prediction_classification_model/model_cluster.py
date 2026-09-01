from sklearn.cluster import KMeans
import pandas as pd



def model_cluster(X_train, X_test):

    kmeans = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )

    kmeans.fit(X_train)

    train_clusters = kmeans.labels_
    test_clusters = kmeans.predict(X_test)

    print("========== KMEANS CLUSTERING ==========")
    print("Number of clusters:", kmeans.n_clusters) # type: ignore
    print("Iterations:", kmeans.n_iter_)
    print("Inertia:", kmeans.inertia_)

    print("\nTraining Cluster Distribution:")
    print(pd.Series(train_clusters).value_counts().sort_index())

    print("\nTesting Cluster Distribution:")
    print(pd.Series(test_clusters).value_counts().sort_index())

    return kmeans
