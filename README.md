# image_de_duplication
The code for removing duplicate images in a directory

[de_dup.py]() is a script that removes the duplicate or similar images in a image dataset. 

When training CNN if the data is craled from the internet, there may exists a certain amount of duplicate images, when separating train/test sets, this duplication can lead to data leak and false high accuracy.

The code follows 4 step:

1. walk through the target directory and get a list of image paths (if image are labeled with coreesponding json files, can also take json file as input)
2. loop the image paths and calculate the [pHash](http://www.hackerfactor.com/blog/index.php?/archives/432-Looks-Like-It.html) of each image, save them in a dict so that those with same pHash would be under a same key. Remove the pictures with same pHash and keep 1.
3. For all the images with different pHash, run a KNN based filter to find the similar images under a threshold. Remove this part of duplicate images.

[reference](https://www.kaggle.com/kretes/duplicate-and-similar-images)
