from src.data.load import load_raw_ibm_hr
from src.data.preprocess import clean_hr_data
from src.modeling.attrition import (
    prepare_attrition_data,
    save_attrition_model,
    select_best_model,
    split_attrition_data,
    train_attrition_models,
)


def main() -> None:
    raw_df = load_raw_ibm_hr()
    clean_df = clean_hr_data(raw_df)

    X, y = prepare_attrition_data(clean_df)
    X_train, X_test, y_train, y_test = split_attrition_data(X, y)

    fitted_models, comparison_df, detailed_metrics = train_attrition_models(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )

    best_model_name, best_model = select_best_model(
        fitted_models=fitted_models,
        comparison_df=comparison_df,
    )

    model_path, metadata_path = save_attrition_model(
        model=best_model,
        model_name=best_model_name,
        metrics=detailed_metrics[best_model_name],
    )

    print("\nModel comparison:")
    print(comparison_df.to_string(index=False))

    print(f"\nSelected model: {best_model_name}")
    print(f"Saved model: {model_path}")
    print(f"Saved metadata: {metadata_path}")


if __name__ == "__main__":
    main()