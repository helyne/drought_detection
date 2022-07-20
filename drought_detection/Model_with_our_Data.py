from drought_detection.data_handling import load_dataset
from drought_detection.data_handling import prepare_for_training, initialize_model, train_model, evaluate_model
from drought_detection.tmp_helyne_model_save import evaluate_model

if __name__ == '__main__':
    # Load data
    train_ds, test_ds, valid_ds, num_examples, num_classes = load_dataset(train_n=1,
    val_n=1, test_n=1 )

    print("loading dataset done")

    batch_size = 64

    train_ds = prepare_for_training(train_ds, batch_size=batch_size)

    valid_ds = prepare_for_training(valid_ds, batch_size=batch_size)

    model = initialize_model(num_classes)

    history, model_path = train_model(model, num_examples)

    accuracy  = evaluate_model(model_path, all_ds)
