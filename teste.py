import importlib.util
import numpy as np
import ast

def find_first_class(file_path):

    with open(file_path, "r") as file:
        tree = ast.parse(file.read(), filename=file_path)
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                return node.name  # Return the name of the first class found
    return None

def load_class_from_file(file_path, class_name):
    spec = importlib.util.spec_from_file_location("module.name", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)

def main(file_path):
    class_name = find_first_class(file_path)
    user_model = load_class_from_file(file_path, class_name)
    
    # Define a toy dataset
    X = np.array([[1, 2], [2, 3], [3, 4], [6, 7], [7, 8], [8, 9]])
    y = np.array([0, 0, 0, 1, 1, 1])

    # Create an instance of the NaiveBayes classifier
    model = user_model()

    # Train the model with the dataset
    model.fit(X, y)

    # Define a test sample
    test_sample = np.array([[4, 5]])

    # Predict probabilities and class for the test sample
    predicted_probabilities = model.predict_proba(test_sample)
    predicted_class = model.predict(test_sample)

    # Output the results
    print("Predicted probabilities:", predicted_probabilities)
    print("Predicted class:", predicted_class)

if __name__ == "__main__":
    # Provide the file path to the NaiveBayes class
    file_path = r"C:\\Users\\Estudo\Documents\\Best-Classification-AI-testing\\model_example.py"
    main(file_path)