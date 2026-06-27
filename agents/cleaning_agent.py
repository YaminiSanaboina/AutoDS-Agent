import pandas as pd


def clean_dataset(df):
    """
    Automatically cleans the dataset
    """

    cleaned_df = df.copy()

    report = []

    # Remove duplicates
    duplicates = cleaned_df.duplicated().sum()

    if duplicates > 0:
        cleaned_df.drop_duplicates(inplace=True)
        report.append(
            f"Removed {duplicates} duplicate rows."
        )


    # Handle missing values
    for column in cleaned_df.columns:

        missing_count = cleaned_df[column].isnull().sum()

        if missing_count > 0:

            # Numerical columns
            if cleaned_df[column].dtype in [
                "int64",
                "float64"
            ]:

                value = cleaned_df[column].median()

                cleaned_df[column].fillna(
                    value,
                    inplace=True
                )

                report.append(
                    f"Filled {missing_count} missing values in {column} with median."
                )


            # Categorical columns
            else:

                value = cleaned_df[column].mode()[0]

                cleaned_df[column].fillna(
                    value,
                    inplace=True
                )

                report.append(
                    f"Filled {missing_count} missing values in {column} with mode."
                )


    if len(report) == 0:
        report.append("Dataset was already clean.")


    return cleaned_df, report