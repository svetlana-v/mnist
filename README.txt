How to use

MODEL_DIR=<folder_name> (e.g ./output)

Run the task locally:
	--job-dir $OUTPUT_PATH \    	
	--runtime-version 1.8 \    	
	--module-name <trainer_name>.task \    	
	--package-path <trainer_name>/ \    	
	--region $REGION \    	
	--verbosity debug
gcloud ml-engine versions create "<version_name>"\    	
	--model "<model_name>" 
	--origin $DEPLOYMENT_SOURCE