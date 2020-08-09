##############################################################################################
# This example is an adaptation from 
# https://rpubs.com/dalekube/XGBoost-Iris-Classification-Example-in-R
#
# This example will go through how to use sagemaker using R paws sdk using Iris data frame
##############################################################################################

###################################################
# Helper Functions
###################################################
# As paws works with the lower level api of aws similar to boto3
# We have to create some helper functions to get some of the python sagemaker functionlity

# Designed from python's sagemaker wrapper package
get_execution_role <- function(config = list()){
  assumed_role <- paws::sts(config = config)$get_caller_identity()$Arn
  if (grepl("AmazonSageMaker-ExecutionRole",assumed_role)){
    role <- gsub("^(.+)sts::(\\d+):assumed-role/(.+?)/.*$", "\\1iam::\\2:role/service-role/\\3", role)
    return(role)}
  
  role <- gsub("^(.+)sts::(\\d+):assumed-role/(.+?)/.*$", "\\1iam::\\2:role/\\3", assumed_role)
  
  # Call IAM to get the role's path
  role_name = gsub(".*/","", role)
  
  tryCatch({role = paws::iam(config = config)$get_role(RoleName = role_name)$Role$Arn},
           error = function(e) stop("Couldn't call 'get_role' to get Role ARN from role name ", role_name ," to get Role path."))
  
  return(role)    
}

# Waits for the training job/ creation of endpoint to be completed
sagemaker_waiter <- function(TrainingJobName = NULL, EndpointName = NULL){    
  if(!is.null(TrainingJobName)){
    while (TRUE){
      tryCatch(job_status <- paws::sagemaker()$describe_training_job(TrainingJobName=TrainingJobName)$TrainingJobStatus)
      if (job_status %in% c('Completed', 'Failed', 'Stopped')){return (job_status)} else {Sys.sleep(1)}
    }
  }
  
  if(!is.null(EndpointName)){
    while (TRUE){
      tryCatch(job_status <- paws::sagemaker()$describe_endpoint(EndpointName=job_name)$EndpointStatus)
      if (job_status != "Creating"){return (job_status)} else {Sys.sleep(1)}
    }
  }
}

###################################################
# Setting initial Parameters
###################################################
# get execution role for sagemaker
role <- get_execution_role()

# S3 parameter set up
bucket <- '<your bucket>'
prefix <- '<prefix where you want to save sagemaker build>'

# Get the region where you want to run sagemaker
my_region <- paws.common:::get_region() 

# Sagemaker uses pre-built docker images
# Each docker image is linked to a region
#
# AWS prebuilt docker images use a framework:
# <account id>.dkr.ecr.<region>.amazonaws.com/sagemaker-<model>:<version>-cpu-py<python version>'
# more information go to: https://docs.aws.amazon.com/sagemaker/latest/dg/pre-built-docker-containers-frameworks.html

# This is the docker image for xgboost on the eu-west-1 region
xgb_img <- '141502667606.dkr.ecr.eu-west-1.amazonaws.com/sagemaker-xgboost:1.0-1-cpu-py3'

###################################################
# Data Prep
###################################################
# get iris data frame
data(iris)

# Convert the Species factor to an integer class starting at 0
target_lookup <- data.frame(Species = iris$Species, label = as.integer(iris$Species)-1)
iris$Species <- target_lookup$label

# split train/test: 0.75/0.25
set.seed(42)
n <- nrow(iris)
train.index <- sample(n,floor(0.75*n))
train_data <- iris[train.index,]
test_data <- iris[-train.index,]

# Next send the data up to AWS S3
# This is so that sagemaker framework can pick it up
# note: column 5 is species
write.table(cbind(train_data[,5], train_data[,-5]), "train.csv",
            sep = ",",quote = F, row.names = F, col.names = F)


# Send train and validation to AWS S3
obj <- readBin("train.csv", "raw", n = file.size("train.csv"))
paws::s3()$put_object(Body = obj, Bucket = bucket, Key = file.path(prefix, 'train/train.csv'))

###################################################
# Training
###################################################
# Get sagemaker object from paws
sm <- paws::sagemaker()

# Python's sagemaker package is a wrapper of boto3 for AWS Sagemaker
# This means we will need build each component that python's sagemaker wraps for us

# Each Job needs to be given an unique name
job_name <- paste0('demo-xgboost-', format(Sys.time(),"%Y-%m-%d-%H-%M-%S",tz="GMT"))

# This lets AWS Sagemaker know what docker file we want to use
# and want format the data will be sent to it
alg_spec <- list(
  TrainingImage = xgb_img,
  TrainingInputMode = "File")

# This lets AWS Sagemaker know what AWS EC2 is required and how many
resource_config <- list(
  InstanceCount = 1,
  InstanceType = "ml.m4.xlarge",
  VolumeSizeInGB = 10)

# This lets AWS Sagemaker know all about the data it is going to get
# set up train input
s3_input_train <- 
  list(ChannelName = "train",
       DataSource=
         list(S3DataSource =  
                list(S3DataType = 'S3Prefix',
                     S3Uri = sprintf('s3://%s/%s/train', bucket, prefix),
                     S3DataDistributionType = 'FullyReplicated')),
       ContentType= 'csv',
       CompressionType = 'None',
       RecordWrapperType = 'None')

# set validation input
s3_input_val <- 
  list(ChannelName = "validation",
       DataSource=
         list(S3DataSource =  
                list(S3DataType = 'S3Prefix',
                     S3Uri = sprintf('s3://%s/%s/validation', bucket, prefix),
                     S3DataDistributionType = 'FullyReplicated')),
       ContentType= 'csv',
       CompressionType = 'None',
       RecordWrapperType = 'None')

# Combine train and validation so that they can be sent to AWS Sagemaker
input_config <- list(s3_input_train)

# Set xgboost Hyperparameters
# list of xgboost hyperparameters: https://xgboost.readthedocs.io/en/latest/parameter.html
num_class <- length(levels(target_lookup$Species))
hyper_param <- list(
  booster="gbtree",
  eta=0.001,
  max_depth=5,
  gamma=3,
  subsample=0.75,
  colsample_bytree=1,
  objective="multi:softprob",
  eval_metric="mlogloss",
  num_class=num_class
)

# NOTE: each model has their own hyperparameters

# Set location for output
s3_output <-  list(S3OutputPath = sprintf('s3://%s/%s/output',bucket, prefix))

# Set conditions for Sagemaker to stop if it runs too long
stopping_cond <- list(MaxRuntimeInSeconds = 60*60)

# Now everything has been set up we can run the AWS Sagemaker Job
sm$create_training_job(RoleArn = role,
                       TrainingJobName = job_name,
                       AlgorithmSpecification = alg_spec,
                       ResourceConfig = resource_config,
                       InputDataConfig = input_config,
                       OutputDataConfig = s3_output,
                       HyperParameters = hyper_param,
                       StoppingCondition = stopping_cond)

# Wait for the AWS Sagemaker job to run
# Note: Without this part we won't know when the job has been completed or failed
sagemaker_waiter(TrainingJobName = job_name)

###################################################
# Host
###################################################
# The next step is to host the model on a rest api

# Similar to before we need to build each component
# To build a rest api from a built jobn we need to do the following steps:
# - Create Model
# - Configure an endpoint
# - Invoke endpoint with model

## Creating AWS Sagemaker Model

# To create a model in AWS Sagemaker you require:
# - docker file used to run the job in the previous task
# - The location the model is stored
host_container <- list(Image = xgb_img,
                      ModelDataUrl = sm$describe_training_job(TrainingJobName=job_name)$ModelArtifacts$S3ModelArtifacts)

create_model_response <- sm$create_model(
  ModelName = job_name, # This doesn't need to be the same name as the job name, but I am lazy
  ExecutionRoleArn = role,
  PrimaryContainer = host_container)

## Configuring an endpoint

# This is to tell AWS Sagemaker want EC2 is required to run the model as a rest api
prod_var <- list(InstanceType = 'ml.m4.xlarge',
                InitialInstanceCount = 1,
                ModelName = job_name,
                VariantName = 'AllTraffic')

create_endpoint_config_response = sm$create_endpoint_config(
  EndpointConfigName = job_name, # This doesn't need to be the same name as the job name, but I am lazy
  ProductionVariants = list(prod_var))


## Invoke endpoint with model
# This builds the endpoint
create_endpoint_response <- sm$create_endpoint(
  EndpointName = job_name, 
  EndpointConfigName = job_name)

# Waiting for the endpoint to be built
sagemaker_waiter(EndpointName = job_name)

###################################################
# Evaluation
###################################################
# Get Sagemaker runtime object from paws
sm_run <- paws::sagemakerruntime()


# First we need to send the test data to the Model rest api to get the predictions
# To do this we need to convert our test data.frame into a raw object
write.table(test_data[,-5], "test.csv", sep = ",",
            quote = F, row.names = F, col.names = F)

obj <- readBin("test.csv", "raw", n = file.size("test.csv"))

# Next we need to send it to the endpoint and tell it we are sending data in csv format
resp <- sm_run$invoke_endpoint(EndpointName = job_name,
                              ContentType = 'text/csv',
                              Body = obj)

# get data from raw object
obj_str <- rawToChar(resp$Body)

# format initial response: "[0.232, 0.445, 0.86546],[..."
obj_group <- gsub("^\\[|\\]$", "", unlist(strsplit(obj_str, "\\],\\[")))

# There are several ways to split this into a useable data.frame
# Please refer to: https://stackoverflow.com/questions/4350440/split-data-frame-string-column-into-multiple-columns
# for the method that suits you. For this example I will use base R.
out <- strsplit(as.character(obj_group),',') 
pred <- as.data.frame(do.call(rbind, out))
colnames(pred) <- levels(target_lookup$Species)

# Use the predicted label with the highest probability
pred$prediction = apply(pred,1,function(x) colnames(pred)[which.max(x)])
pred$label = levels(target_lookup$Species)[test_data$Species+1]

# Calculate the final accuracy
result = sum(pred$prediction==pred$label)/nrow(pred)
print(paste("Final Accuracy =",sprintf("%1.2f%%", 100*result)))

###################################################
# (Optional) Clean-up
###################################################
# When you are done with your model you can delete it so that it doesn't continue incurring cost
sm$delete_endpoint(EndpointName = job_name)
