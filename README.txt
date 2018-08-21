How to useRead the ML Engine manual: https://cloud.google.com/ml-engine/docs/tensorflow/getting-started-training-prediction======================================================================================TRAINING THE MODEL LOCALLY======================================================================================Define environmental variables:

MODEL_DIR=<folder_name> (e.g ./output)

Run the task locally:gcloud ml-engine local train \    --module-name trainer_tf.task \    --package-path trainer_tf \    --job-dir $MODEL_DIR======================================================================================TRAINING THE MODEL IN THE CLOUD======================================================================================Define environmental variables:JOB_NAME=<job_name>OUTPUT_PATH=gs://$BUCKET_NAME/$JOB_NAMERun the task in the Google Cloud (e.g. trainer_name = trainer_tf | trainer_keras | trainer_tf_with_LRD | trainer_tf_softmaxreg):gcloud ml-engine jobs submit training $JOB_NAME \    	
	--job-dir $OUTPUT_PATH \    	
	--runtime-version 1.8 \    	
	--module-name <trainer_name>.task \    	
	--package-path <trainer_name>/ \    	
	--region $REGION \    	
	--verbosity debug======================================================================================CREATING VERSION IN THE CLOUD======================================================================================gcloud ml-engine models create "<model_name>"DEPLOYMENT_SOURCE=gs://<BUCKET_NAME>/<JOB_NAME>/export/
gcloud ml-engine versions create "<version_name>"\    	
	--model "<model_name>" 
	--origin $DEPLOYMENT_SOURCE======================================================================================GETTING ONLINE PREDICTIONS======================================================================================MODEL_NAME="[MODEL-NAME]"INPUT_DATA_FILE=“input.json"VERSION_NAME="[VERSION-NAME]”gcloud ml-engine predict --model $MODEL_NAME  \                         --version $VERSION_NAME \                         --json-instances $INPUT_DATA_FILEAs an input.json you can use "image.json", "images_3.json" or "images_10.json"in this example