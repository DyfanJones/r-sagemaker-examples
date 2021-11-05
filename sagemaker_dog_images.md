# Disclaimers:

**Note:** This example is heavily inspired by https://github.com/robinsones/weratedogs presented in `SLC R Users Group Future Meetups`. To see the original code please go to https://github.com/robinsones/weratedogs and watch the meet up presentation video on youtube: https://youtu.be/EZXJmFg4MIg.

This code was ran on `AWS Sagemaker Notebooks`. 

`R version: 4.1.1`

# Set up

## Install packages

As `conda` is installed onto `AWS Sagemaker` I will use it to install some R packages.

```r
# helper function to install R packages on sagemaker using conda
conda_install = function(..., env="R", channel = "conda-forge"){
    pkg = list(...)
    cmd = c("install", "-n", env, "-c", channel, pkg, "-y")
    return(cat(processx::run("conda", as.character(cmd))$stdout))
}
```

Install `R` packages for image editing and [R AWS SDK paws](https://github.com/paws-r/paws)

```r
conda_install(
    "r-magick", "r-remotes", "r-rtweet",
    # R6sagemaker dependencies
    "r-paws", "r-lgr", "r-urltools", "r-zip"
)
```

To keep the secret tokens, passwords extra I have opted to use `AWS Secret Manager`

```r
# get tokens from aws secret manager
sm = paws::secretsmanager()
sm = jsonlite::fromJSON(sm$get_secret_value("<your secret id>")$SecretString)
```

Now lets install [R6sagemaker](https://github.com/DyfanJones/sagemaker-r-sdk).

```r
Sys.setenv("GITHUB_PAT" = sm$github_token)
remotes::install_github("DyfanJones/sagemaker-r-sdk")
```

# Collect Images from Twitter

Now lets collect the image data from twitter.

```r
library(tidyverse)
library(rtweet)
library(magick)
library(fs)
```

Before you work with twitter please read rtweet guide in how to get setup https://docs.ropensci.org/rtweet/articles/auth.html.

```r
# set twitter token
token <- create_token(
  app = "sagemake-demo",
  consumer_key = sm$twitter_consumer_key,
  consumer_secret = sm$twitter_consumer_secret,
  access_token = sm$twitter_access_token,
  access_secret = sm$twitter_access_secret
)
```
Get images from we are dogs tweet feed.

```r
weratedogs <- get_timeline("dog_rates", n = 3200, token = token)
cleaned_weratedogs <- weratedogs %>% 
  mutate(media_url = as.character(media_url)) %>%
  filter(!is.na(media_url), is.na(reply_to_status_id)) %>%
  select(text, media_url) %>%
  mutate(rating = str_extract(text, "\\d+/"), 
         name = str_extract(text, "This is [A-Za-z]+.")) %>%
  filter(!is.na(rating), 
         !is.na(name)) %>%
  mutate(name = str_remove_all(name, "This is |\\.|/| ")) %>%
  filter(rating <= 15) %>%
  mutate(dichotimized_rating = if_else(rating <= 13, 0, 1)) %>%
  # remove duplicate names 
  add_count(name) %>%
  filter(n == 1)
```

Download dog rate images.

```r
# create an folder to contain images
fs::dir_create("data/images")

# Download the images of all the dogs
walk2(cleaned_weratedogs$media_url, cleaned_weratedogs$name, 
      ~download.file(.x, paste0(file.path("data","images", .y), ".jpg")))
```

Resize images

```r
read_scale_and_write <- function(image_name) { 
  image_name %>%
    image_read() %>%
    image_scale("750x1000!") %>%
    image_write(path = file.path("data", "resized_images", path_file(image_name)), format = "jpg")
}
jpg_files <- dir_ls(file.path("data","images"), regexp = "\\.jpg$")

dir_create(file.path("data", "resized_images"))
walk(jpg_files, read_scale_and_write)

#delete original files
dir_delete(file.path("data","images"))
```

Create training, validation splits/
```r
dir_create(file.path("data","holdout"))
dir_create(file.path("data","train"))
dir_create(file.path("data","validation"))

set.seed(42)
resized_images_files <- dir_ls(file.path("data","resized_images"),regexp = "\\.jpg$")
holdout_set <- sample(resized_images_files, length(resized_images_files)/10)
remaining_images <- setdiff(resized_images_files, holdout_set)
remaining_images <- setdiff(resized_images_files, holdout_set)
train_set <- sample(remaining_images, length(remaining_images)*.70)

# move files to train and holdout
file_move(holdout_set, file.path("data", "holdout"))
file_move(train_set, file.path("data","train"))

# move remaining files to validation folder
validation_set <- dir_ls(file.path("data", "resized_images"), regexp = "\\.jpg$")
file_move(validation_set, file.path("data", "validation"))
```

Create lst file format for [image classification model(https://docs.aws.amazon.com/sagemaker/latest/dg/image-classification.html).

```r
files_list = dir_ls("data", recurse = T, type = "file")

pictures_split <- tibble("file_name" = fs::path_file(files_list), "location" = fs::path_dir(files_list))

lst_info <- cleaned_weratedogs %>%
  mutate(file_name = paste0(name, ".jpg")) %>%
  inner_join(pictures_split, by = "file_name") %>%
  mutate(split = fs::path_file(location),
         index = row_number()) %>%
  rename(file_location = location)

train_lst <- lst_info %>%
  filter(split == "train") %>%
  select(index, dichotimized_rating, file_name)

validation_lst <- lst_info %>%
  filter(split == "validation") %>%
  select(index, dichotimized_rating, file_name)

# your_image_directory/train_img_dog1.jpg

fs::dir_delete(file.path("data", "resized_images"))

fs::dir_create("model_data")
write_tsv(validation_lst, file = file.path("model_data","validation_lst.lst"), col_names = FALSE)
write_tsv(train_lst, file = file.path("model_data","train_lst.lst"), col_names = FALSE)
```

# Upload the data

Upload the data onto the s3 bucket. The images are uploaded onto train and validation bucket. The lst files are uploaded to train_lst and validation_lst folders.

```r
# Get R6sagemaker sdk
library(R6sagemaker)
# Part of R6sagemaker ecosystem
library(R6sagemaker.common)

bucket = "my-bucket"
key_prefix="project-we-are-dogs"

sess = Session$new(default_bucket=bucket)

# get sagemaker execution role for model builds
role_arn = get_execution_role()

# Upload data to s3 buckets
s3_up = S3Uploader$new()
s3_up$upload("data", s3_path_join("s3://", bucket, key_prefix))
s3_up$upload(file.path("model_data","validation_lst.lst"), s3_path_join("s3://", bucket, key_prefix, "validation_lst"))
s3_up$upload(file.path("model_data","train_lst.lst"), s3_path_join("s3://", bucket, key_prefix, "train_lst"))
```

# Create Estimator

Get model `AWS ECR` image, this image can be changed to what ever region you are working in. Please change `region="eu-west-1` to your desired area.

```r
training_image <- ImageUris$new()$retrieve(framework = "image-classification", region ="eu-west-1")
```

Set up model estimator class.

```r
ic <- Estimator$new(
  training_image, 
  role_arn,
  instance_count = 1,
  instance_type = "ml.p2.xlarge",
  volume_size = 50,
  input_mode = "File",
  output_path = s3_path_join("s3://", bucket, key_prefix, "output"),
  sagemaker_session = sess
)
```

Set image classifications [hyperparameters](https://docs.aws.amazon.com/sagemaker/latest/dg/IC-Hyperparameter.html).
```r
ic$set_hyperparameters(
    num_layers=18L,
    image_shape="3,1000,750",
    num_classes=2L,
    num_training_samples=length(train_set),
    mini_batch_size=16L,
    epochs=5L,
    top_k=2L,
    precision_dtype="float32"
)
```

Set up model input data.

```r
s3_input <- function(s3_data) {
  TrainingInput$new(
    s3_data,
    distribution="FullyReplicated",
    content_type="application/x-image",
    s3_data_type="S3Prefix"
  )
}


train_data <- s3_input(s3_path_join("s3://", bucket, key_prefix, "data", "train"))
validation_data <- s3_input(s3_path_join("s3://", bucket, key_prefix, "data", "validation"))
train_data_lst <- s3_input(s3_path_join("s3://", bucket, key_prefix, "train_lst"))
validation_data_lst <- s3_input(s3_path_join("s3://", bucket, key_prefix, "validation_lst"))

data_channels = list(
    "train"=train_data,
    "validation"=validation_data,
    "train_lst"=train_data_lst,
    "validation_lst"=validation_data_lst
)
```

## Fit model.

```r
ic$fit(inputs = data_channels, logs = TRUE)

# save ic class
saveRDS(ic, "image-class-estimator.rds")
```

# Create Endpoint

By creating an endpoint we are deploying the model as a rest end.

```r
ic_classifier = ic$deploy(
    initial_instance_count=1,
    instance_type="ml.m4.xlarge",
    serializer=IdentitySerializer$new(content_type="application/x-image")
)
```

## Predictions

We can now send our holdout group to the endpoint to get our predictions. To do this we send each image to the endpoint individually.

```r
img = fs::dir_ls(file.path("data", "holdout"))
obj = lapply(seq_along(img), \(i) readBin(img[[i]], "raw", n = file.size(img[[i]])))

results = lapply(obj, ic_classifier$predict)
con_lst = lapply(results, rawConnection)
pred = bind_rows(lapply(con_lst, jsonlite::stream_in))
names(pred) <- c("low", "high")

pred$image = img
```

At the end we can optionally shut down the endpoint to save on cost.

```r
ic_classifier$delete_endpoint()
```

# Create Batch Tranform

```r
transformer <- ic$transformer(instance_count=1, instance_type='ml.m4.xlarge')
transformer$transform(s3_path_join("s3://", bucket, key_prefix, "data", "validation"))
```

## Predictions

As the Batch Transform Job has already been completed all we have to do is collect the information from `AWS S3`.

```r
s3_dl <- S3Downloader$new()

result <- s3_dl$list(transformer$output_path)

out <- bind_rows(lapply(result, \(x){
    obj = jsonlite::parse_json(s3_dl$read_file(x))
    tibble(low = obj[["prediction"]][[1]], high = obj[["prediction"]][[2]])
    })
)

out$labels <- s3_dl$list(s3_path_join("s3://", bucket, key_prefix, "data", "validation"))
```
