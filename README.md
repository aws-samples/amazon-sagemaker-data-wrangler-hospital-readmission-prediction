<div style="font-size:500%;font-weight:bold"> </div>

# Amazon SageMaker Data Wrangler - Diabetic Patient Readmission Prediction

Patient readmission to hospital after prior visits for the same disease results in additional burden on healthcare providers and health system. Being able to understand, and predict readmission allows providers to create a better treatment plans and care. Reduction in cost is another benefit of such predictive modeling. In this example, we show how we prepare a machine learning dataset and build a predictive model using a diabetic patient readmission dataset that captures 10 years (1999-2008) of clinical care at 130 US hospitals and integrated delivery networks in Amazon SageMaker Studio Data Wrangler.

[Amazon SageMaker Data Wrangler](https://aws.amazon.com/sagemaker/data-wrangler/) in Amazon SageMaker Studio is a tool designed to allow data scientist quickly and iteratively explore and transform data for machine learning use cases. This project showcases how you can build a machine learning data transformation pipeline without writing sophisticated coding and create a model training, feature store or a ML pipeline with reproducibility for a diabetic patient readmission prediction use case.

In this example, you will be running machine learning workflow with Amazon SageMaker Data Wrangler and Amazon SageMaker features using a HCLS dataset.

Here are the high-level activities:

1. [Load UCI Source Dataset into your S3 bucket](#1-source-dataset)
1. [Design your DataWrangler flow file](#2-design-your-dataWrangler-flow-file)
1. [Processing & Training Jobs for Model building](#3-processing-and-training-jobs-for-model-building)
1. [Host trained Model for real-time inference](#4-Host-trained-Model-for-real-time-inference)

## 1. Source Dataset

[UCI diabetic patient readmission dataset](https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008). The dataset represents 10 years (1999-2008) of clinical care at 130 US hospitals and integrated delivery networks. It includes over 50 features representing patient and hospital outcomes.

You will start by downloading the dataset and uploading it to a S3 bucket for you to run the example. Please review and execute the code in [datawrangler_workshop_pre_requisite.ipynb](datawrangler_workshop_pre_requisite.ipynb). The data will be available in `s3://sagemaker-${region}-${account_number}/sagemaker/demo-diabetic-datawrangler/` if you leave everything default.

## 2. Design your DataWrangler flow file

###  Data Wrangler flow overview and highlights

This project comes with a pre-built Data Wrangler flow file that can be customized with your `s3Uri` for reusability: [datawrangler_diabetes_readmission.flow](datawrangler_diabetes_readmission.flow).

![dw-flow](images/data_wrangler_flow.png)

It has multiple files from S3 loaded in: `diabetic_data_hospital_visits.csv`, `diabetic_data_demographic.csv` and `diabetic_data_labs.csv` for demonstration. It performs a inner join between the tables in `diabetic_data_hospital_visits.csv` and `diabetic_data_demographic.csv` by `encounter_id`. It has 22 transformation steps applied to process the data to meet the following requirements:

* no duplicate columns
* no duplicate entries
* no missing values (either fill the missing ones or remove columns that are largely missing)
* one hot encode the categorical features
* ordinally encode the age feature
* normalization (Standard scaler)
* Custom Transformation (Feature Store - EventTime needed)
* Analysis (Quick Model)
* ready for ML training (Export notebook steps)
* Hosting steps need to be added here
* Model monitoring
* Extract test-dataset after transformation

These are analyses created at different stage of the wrangling to serve as indication of the value these wrangling steps add. Most noticeably the Quick Model tells us that patient readmission prediction increases from X to Y after performing additional steps between 13 and 22 (in [datawrangler_diabetes_readmission.flow](datawrangler_diabetes_readmission.flow)).

In this lab, we will perform data preprocessing using a combination of transformations described below to demonstrate the capability of Amazon SageMaker Data Wrangler. We will then train a XGBoost model to show you the process after data wrangling. We will then be hosting a trained model to SageMaker Hosted Endpoint for real-time inferencing.

##  Walk through

### Create a new flow

Please click on the **SageMaker component and registry** tab and click **New flow**.

![create_flow](images/create_flow.png)

### Rename your DW dataflow file

Right click on the **untitled.flow** file tab to reveal below options.  Then choose Rename file to change the file name.

![import_data](images/rename-flow-options.png)

![import_data](images/rename-flow-dialog.png)

### Load the data from S3 into Data Wrangler

Select Amazon S3 as data source in **Import Data** view.

![import_data](images/import_data_s3.png)

*Note: You could also import data from Athena: [how databases and tables in Amazon Athena can be imported](https://docs.aws.amazon.com/sagemaker/latest/dg/data-wrangler-import.html#data-wrangler-import-athena).*

Select the csv files from the bucket: `s3://sagemaker-${region}-${account_number}/sagemaker/demo-diabetic-datawrangler/` one at a time.

![import_csv](images/import_1_csv.png)

### Join the CSV files

Click the + sign on the Data-types icon for `diabetic_data_hospital_visits.csv` Select **Join** and new panel is presented for configuring input dataset join.

![import_data](images/dataset_join_screen.png)

1) Select `diabetic_data_demographic.csv` dataset as Right dataset.

2) Click `Join` and new Preview panel is presented.

![import_data](images/join_layout_screen.png)

3) Give a name to the Join and choose join type and Left & Right columns for join condition

![import_data](images/join_configuration_screen.png)

4) Click Apply to preview the joined dataset

![import_data](images/joined_preview_screen.png)

5) Click Add for the join configuration to be added to the data-flow

![import_data](images/join_updated_dataflow.png)

### Transformations

We will use Data Wrangler built-in transforms to apply the listed transformations to our dataset.

###  No Duplicate Columns

1) Click + sign next to Join flow icon and choose `Add Transform`

![import_data](images/add_transform_screen.png)

2) Pick `Manage Columns` from the list of transforms on the right panel

![import_data](images/manage_columns_drop_column.png)

3) Choose `Drop Column` transform and select `encounter_id_1` column to drop

![import_data](images/manage_columns_to_drop.png)

4) Click `Preview` to preview the changes to the data set

![import_data](images/manage_columns_preview.png)

5) Click `Add` to add the change to the data flow

![import_data](images/manage_columns_add.png)

6) Repeat above steps 1 through 5 for `patient_nbr_1` column.

![import_data](images/manage_columns_drop2.png)


###  No Duplicate Rows/Observations

1) Click + sign next to Join flow icon and choose `Add Transform`

![import_data](images/add_transform_screen.png)

2) Pick `Custom Transform` from the list of transforms on the right panel

![import_data](images/custom_transform_options.png)

3) select `Python (Pandas)` and enter below line of code in the text box.  Then click `Preview` to view the results.

```
df.drop_duplicates(subset=['encounter_id_0', 'patient_nbr_0'],keep='first', inplace=True)
```

![import_data](images/custom_transform_preview.png)

4) Click `Add` to add the change to the data flow

![import_data](images/custom_transform_add.png)

###  Custom Transformation - Add new features to your dataset

Feature Store would need Event Time feature to be present to be able to store inside Feature Groups

1) Click + sign next to Join flow icon and choose `Add Transform`

![import_data](images/add_transform_screen.png)

2) Pick `Custom Transform` from the list of transforms on the right panel

![import_data](images/custom_transform_options.png)

3) select `Python (Pandas)` and enter below line of code in the text box.  Then click `Preview` to view the results.

```
import time
df['eventTime'] = time.time()
```

![import_data](images/custom_transform_add_feature.png)

4) Click `Add` to add the change to the data flow

![import_data](images/custom_transform_add_screen.png)

###  More Custom Transformations - impute fillers for undesirable values

1) Click + sign next to Join flow icon and choose `Add Transform`

![import_data](images/add_transform_screen.png)

2) Pick `Custom Transform` from the list of transforms on the right panel

![import_data](images/custom_transform_options.png)

3) select `Python (Pandas)` and enter below line of code in the text box.  Then click `Preview` to view the results.

```
# Table is available as variable `df`
df["race"]=df["race"].str.replace("?","Other")
df["weight"]=df["weight"].str.replace("?","0")
df["payer_code"]=df["payer_code"].str.replace("?","0")
df["medical_specialty"]=df["medical_specialty"].str.replace("?","Other")
```

![import_data](images/custom_transform_imputes.png)

4) Click `Add` to add the change to the data flow

![import_data](images/custom_transform_imputes_add.png)



###  Handle missing values

1) Click + sign next to Join flow icon and choose `Add Transform`

![import_data](images/add_transform_screen.png)

2) Pick `Handle missing` from the list of transforms on the right panel and choose `Impute` for Transform

![import_data](images/handle_missing_screen.png)

3) Choose `Column type` as `Numeric` and select `Input column` as `diag_1`.  Let's use `Mean` for `Imputing strategy`.  You can also provide optional Output column name.

![import_data](images/handle_missing_impute_diag1.png)

4) Click `Add` to add the change to the data flow

![import_data](images/handle_missing_impute_diag1_add.png)

5) Repeat above steps 1 through 4 for `diag_2` feature as shown below

![import_data](images/handle_missing_impute_diag2.png)

6) Repeat above steps 1 through 4 for `diag_3` feature as shown below

![import_data](images/handle_missing_impute.png)


![import_data](images/handle_missing_impute_add.png)

###  One-hot Encoding for categorical features

1) Click + sign next to Join flow icon and choose `Add Transform`

![import_data](images/add_transform_screen.png)

2) Pick `Encode categorical` from the list of transforms on the right panel. Select `One-hot encode` and `gender` for input column.  For `Output style`, choose `Columns`.  After filling the fields click `Preview`

![import_data](images/one_hot_encod_preview.png)

3) After review the transformation results, Click `Add` to add the change to the data flow

![import_data](images/one_hot_encod_add.png)

4) Repeat above steps 1 through 3 to encode `race` feature as shown below

![import_data](images/one_hot_encod_race.png)

###  Ordinal Encoding for categorical features

1) Click + sign next to Join flow icon and choose `Add Transform`

![import_data](images/add_transform_screen.png)

2) Pick `Encode categorical` from the list of transforms on the right panel. Select `Ordinal encode` and `age` for input column.  For `Invalid handling strategy` select `skip`.  After filling the fields click `Preview`

![import_data](images/ordinal_encod_screen.png)

3) After review the transformation results, Click `Add` to add the change to the data flow

![import_data](images/ordinal_encod_add.png)

###  Normalization with `standard scaler` using `Process numeric` Transform

1) Click + sign next to Join flow icon and choose `Add Transform`

![import_data](images/add_transform_screen.png)

2) Pick `Process numeric` from the list of transforms on the right panel. Select `Scale values` for Transform and `Standard scaler` for Scaler.  Provide `diag_1_imputed` as `Input column`. After filling the fields click `Preview`

![import_data](images/scaler_preview_diag1.png)

3) After review the transformation results, Click `Add` to add the change to the data flow

![import_data](images/scaler_add_diag1.png)

###  One-hot Encoding for target label feature

1) Click + sign next to Join flow icon and choose `Add Transform`

![import_data](images/add_transform_screen.png)

2) Pick `Encode categorical` from the list of transforms on the right panel. Select `One-hot encode` and `readmission` for input column.  For `Output style`, choose `Columns`.  After filling the fields click `Preview` followed by `Add`

![import_data](images/1hot_readmitted.png)

###  Position the target label as first column to utilize xgboost algorithm

We will be using xgboost built-in algorith to train the model using our dataset.  One of the pre-requisite for using xgboost algorithm is to have the target label as first column.  Let's do that.

1) Click + sign next to Join flow icon and choose `Add Transform`

![import_data](images/add_transform_screen.png)

2) Pick `Manage Column` from the list of transforms on the right panel. Select `Move Column` for `Transform` and select `Move to start` for `Move type`.  Provide a name to new column `readmitted_1hot_NO`. After filling the fields click `Preview`

![import_data](images/position_target_column_preview.png)

3) After review the transformation results, Click `Add` to add the change to the data flow

![import_data](images/position_target_column.png)

###  Drop redundant columns

1) Click + sign next to Join flow icon and choose `Add Transform`

![import_data](images/add_transform_screen.png)

2) Pick `Manage Columns` from the list of transforms on the right panel

![import_data](images/manage_columns_drop_column.png)

3) Choose `Drop Column` transform and select `age` column to drop

![import_data](images/drop_age.png)

4) Click `Preview` to preview the changes to the data set.  Then `Add` to add the changes to flow file.

5) Repeat above steps 1 through 4 for other redundant columns `race` `diag_1` `diag_2` `diag_3` `readmitted`, as shown below

![import_data](images/drop_race.png)


![import_data](images/drop_diag_3.png)


![import_data](images/drop_readmitted.png)


###  Quick Model

Now that we have transformed our initial dataset, Let's explore `Quick Model` transformation. This transform option allows you to visualize the target model accuracy by dynamically running the model on the fly with the transformed data.  This allows you to quickly see the feature relationship weightage on target prediction

1) Click + sign next to Join flow icon and choose `Add analysis`

![import_data](images/add_analysis_screen.png)

2) Select `Quick Model` from the list of Analysis types on the right panel.

![import_data](images/analysis_quick_model_option.png)

3) Give a name to your analysis and select the target label in `Label` field.

![import_data](images/analysis_quick_model_configuration.png)

4) Click `Preview` and wait for the model to be results to be displayed on the screen

![import_data](images/analysis_quick_model_result.png)

5) Click `Create` button to add the quick model analysis to the data flow.

###  Export Options

We are now ready to export dataflow for further processing.

1) Save the DW flow file as shown below

![import_data](images/dw_flow_save.png)

1) Click `Export` tab and select `Steps` icon to reveal all the DW flow steps.  Click the last step to mark it as check (as shown in figure below)

![import_data](images/export_step_check.png)

2) Click `Export step` to reveal the export options.  You currently have 4 export options

    a) `Save to S3` Save to S3 using a SageMaker Processing Job.
    b) `Pipeline` exports a Jupyter Notebook that creates a Pipeline with your data flow.
    c) `Python Code` exports your data flow to python code.
    d) `Feature Store` exports a Jupyter Notebook that creates a Feature Store feature group and adds features to an offline or online feature store.


![import_data](images/new_export_options.png)

3) Select `Pipeline` to generate Jupyter Notebook that creates a Pre-processing Job using your data flow file.

![import_data](images/sample-processing-notebook.png)

## 3. Processing and Training Jobs for Model building

###  Pre-Processing Job submission

1) We are now ready to submit a SageMaker Processing Job using the data flow file.  Run all the cells upto `Create Processing Job`.  This cell `Create Processing Job` will trigger a new SagaMaker processing job by provisioning managed infrastructure and running the required DataWrangler docker container on that infrastructure.

![import_data](images/preprocessing-job-submit.png)

2) You can check the status of the submitted processing job by running next cell `Job Status & S3 Output Location`

![import_data](images/preprocessing-job-status.png)


3) You can also check the status of the submitted processing job from Amazon SageMaker Console as shown below

![import_data](images/preprocessing-job-status.png)


###  Train a model with Amazon SageMaker

1) Now that the data has been processed, you may want to train a model using the data.  The same notebook has sample steps to train an XGBoost algorithm.  Since our use case is binary classification, we need to change the `objective` inside the sample training steps as shown below.


![import_data](images/change-objective.png)

2) All set.  Now we are ready to fire our training job using SageMaker managed infrastructure.  Run the cell below.

![import_data](images/submit-training-job.png)

3) You can monitor the status of submitted training job in SageMaker Console under `Training`/`Training jobs` tab on the left.

![import_data](images/jobs-monitor.png)

## 4. Host trained Model for real time inference

###  Deploy model for real-time inference

1) We will now use another notebook provided under project folder `hosting/Model_deployment_Steps.ipynb`.
This is a simple notebook with 2 cells - First cell has code for deploying your model to persistent endpoint.  Here you need to update `model_url` with your training job output `S3 model artifact`.  Here are image for reference.

![import_data](images/training-output.png)


![import_data](images/model-deployment.png)

2) The second cell in the notebook will run inference on sample test file `test_data_UCI_sample.csv`.

![import_data](images/inference-cell.png)

###  Conclusion

This concludes the example.  In this example you have learnt how to use SageMaker Data Wrangler capability to create data preprocessing, feature engineering steps using simple to use Data Wrangler GUI.  We then used the generated notebook to submit a SageMaker managed processing job to perform the data preparation using our data flow file.  Later we saw how to train a simple XBOOST algorithm using our processed dataset.  In the end we hosted our trained model and ran inferences against synthetic test data.
