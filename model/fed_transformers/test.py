import pandas as pd 

from model.fed_transformers.classification import ClassificationModel, MultiLabelClassificationModel


# Test for binary classification

# Train and Evaluation data needs to be in a Pandas Dataframe of two columns.
# The first column is the text with type str, and the second column is the
# label with type int.
train_data = [
    ["Example sentence belonging to class 1", 1],
    ["Example sentence belonging to class 0", 0],
]
train_df = pd.DataFrame(train_data)

eval_data = [
    ["Example eval sentence belonging to class 1", 1],
    ["Example eval sentence belonging to class 0", 0],
]
eval_df = pd.DataFrame(eval_data)

model_type, model_name =   ("distilbert", "distilbert-base-uncased")
# Create a ClassificationModel
model = ClassificationModel(
    model_type,
    model_name,
    use_cuda=False,
    args={"no_save": True, "reprocess_input_data": True, "overwrite_output_dir": True},
)

# Train the model
model.train_model(train_df)

# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(eval_df)