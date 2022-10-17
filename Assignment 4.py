from tensorflow.keras import preprocessing
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Rescaling, Conv2D, Dense, Flatten, MaxPooling2D, Dropout
import PIL
from tensorflow.keras.models import load_model

print("---------- Model with pooling -----------")

# load datasets

#training_set = preprocessing.image_dataset_from_directory("flowers",validation_split=0.2,subset="training",label_mode="categorical",seed=0,image_size=(100,100))
test_set = preprocessing.image_dataset_from_directory("flowers",
                                                              validation_split=0.2,
                                                              subset="validation",
                                                              label_mode="categorical",
                                                              seed=0,
                                                              image_size=(100,100))


# build the model
##m = Sequential()
##m.add(Rescaling(1/255))
##m.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(100,100,3)))
##m.add(MaxPooling2D(pool_size=(2, 2)))
##m.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
##m.add(MaxPooling2D(pool_size=(2, 2)))
##m.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
##m.add(MaxPooling2D(pool_size=(2, 2)))
##m.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
##m.add(MaxPooling2D(pool_size=(2, 2)))
##m.add(Flatten())
##m.add(Dense(128, activation='relu'))
##m.add(Dropout(0.5))
##m.add(Dense(5, activation='softmax'))

# setting and training
##m.compile(loss="categorical_crossentropy", metrics=['accuracy'])
##history  = m.fit(training_set, batch_size=32, epochs=25,verbose=0)
##m.save("ModelWithPooling.h5")
##print(history.history["accuracy"])
##print(training_set.class_names)

print("------ Calling model ------")
pooling_model = load_model("ModelWithPooling.h5")

# testing
print("Testing.")
score = pooling_model.evaluate(test_set, verbose=0)
print('Test accuracy:', score[1])
#return m
print("Completed successfully")
pooling_model.summary()

###def test_image(m, image_file) :
##""" Classifies the given image using the given model. """
### load the image
##image_file = "flowers/rose/12240303_80d87f77a3_n.jpg"
##img = preprocessing.image.load_img(image_file,target_size=(100,100))
##img_arr = preprocessing.image.img_to_array(img)
##
### show the image
##plt.imshow(img_arr.astype("uint8"))
##plt.show()
##
### classify the image
##img_cl = img_arr.reshape(1,100,100,3)
##score = m.predict(img_cl)
##print(score)


print("---------- Model without dropout -----------")

# load datasets
##training_set = preprocessing.image_dataset_from_directory("flowers",
##                                                              validation_split=0.2,
##                                                              subset="training",
##                                                              label_mode="categorical",
##                                                              seed=0,
##                                                              image_size=(100,100))
test_set = preprocessing.image_dataset_from_directory("flowers",
                                                              validation_split=0.2,
                                                              subset="validation",
                                                              label_mode="categorical",
                                                              seed=0,
                                                              image_size=(100,100))


# build the model
##m = Sequential()
##m.add(Rescaling(1/255))
##m.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(100,100,3)))
##m.add(MaxPooling2D(pool_size=(2, 2)))
##m.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
##m.add(MaxPooling2D(pool_size=(2, 2)))
##m.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
##m.add(MaxPooling2D(pool_size=(2, 2)))
##m.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
##m.add(MaxPooling2D(pool_size=(2, 2)))
##m.add(Flatten())
##m.add(Dense(128, activation='relu'))
##m.add(Dense(5, activation='softmax'))
##
### setting and training
##m.compile(loss="categorical_crossentropy", metrics=['accuracy'])
##history  = m.fit(training_set, batch_size=32, epochs=25,verbose=0)
##m.save("ModelWithoutDropout.h5")
##print(history.history["accuracy"])
##print(training_set.class_names)

print("------ Calling model ------")
dropout_model = load_model("ModelWithoutDropout.h5")

# testing
print("Testing.")
score = dropout_model.evaluate(test_set, verbose=0)
print('Test accuracy:', score[1])
#return m
print("Completed successfully")
dropout_model.summary()

###def test_image(m, image_file) :
##""" Classifies the given image using the given model. """
### load the image
##image_file = "flowers/rose/12240303_80d87f77a3_n.jpg"
##img = preprocessing.image.load_img(image_file,target_size=(100,100))
##img_arr = preprocessing.image.img_to_array(img)
##
### show the image
##plt.imshow(img_arr.astype("uint8"))
##plt.show()
##
### classify the image
##img_cl = img_arr.reshape(1,100,100,3)
##score = m.predict(img_cl)
##print(score)

print("---------- Model with different image size -----------")
#loading datasets
##training_set = preprocessing.image_dataset_from_directory("flowers",
##                                                              validation_split=0.2,
##                                                              subset="training",
##                                                              label_mode="categorical",
##                                                              seed=0,
##                                                              image_size=(64,64))
test_set = preprocessing.image_dataset_from_directory("flowers",
                                                              validation_split=0.2,
                                                              subset="validation",
                                                              label_mode="categorical",
                                                              seed=0,
                                                              image_size=(64,64))


### build the model
##m = Sequential()
##m.add(Rescaling(1/255))
##m.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(64,64,3)))
##m.add(MaxPooling2D(pool_size=(2, 2)))
##m.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
##m.add(MaxPooling2D(pool_size=(2, 2)))
##m.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
##m.add(MaxPooling2D(pool_size=(2, 2)))
##m.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
##m.add(MaxPooling2D(pool_size=(2, 2)))
##m.add(Flatten())
##m.add(Dense(128, activation='relu'))
##m.add(Dropout(0.5))
##m.add(Dense(5, activation='softmax'))
##
### setting and training
##m.compile(loss="categorical_crossentropy", metrics=['accuracy'])
##history  = m.fit(training_set, batch_size=32, epochs=25,verbose=0)
##m.save("ModelWithDifferentImageSizes.h5")
##print(history.history["accuracy"])
##print(training_set.class_names)

print("------ Calling model ------")
image_model = load_model("ModelWithDifferentImageSizes.h5")

# testing
print("Testing.")
score = image_model.evaluate(test_set, verbose=0)
print('Test accuracy:', score[1])
#return m
print("Completed successfully")
image_model.summary()

###def test_image(m, image_file) :
##""" Classifies the given image using the given model. """
### load the image
##image_file = "flowers/rose/12240303_80d87f77a3_n.jpg"
##img = preprocessing.image.load_img(image_file,target_size=(64,64))
##img_arr = preprocessing.image.img_to_array(img)
##
### show the image
##plt.imshow(img_arr.astype("uint8"))
##plt.show()
##
### classify the image
##img_cl = img_arr.reshape(1,64,64,3)
##score = m.predict(img_cl)
##print(score)



print("---------- Model with varying the number of filters -----------")

# load datasets
##training_set = preprocessing.image_dataset_from_directory("flowers",
##                                                              validation_split=0.2,
##                                                              subset="training",
##                                                              label_mode="categorical",
##                                                              seed=0,
##                                                              image_size=(100,100))
test_set = preprocessing.image_dataset_from_directory("flowers",
                                                              validation_split=0.2,
                                                              subset="validation",
                                                              label_mode="categorical",
                                                              seed=0,
                                                              image_size=(100,100))


### build the model
##m = Sequential()
##m.add(Rescaling(1/255))
##m.add(Conv2D(32, kernel_size=(5, 5),activation='relu',input_shape=(100,100,3)))
##m.add(MaxPooling2D(pool_size=(2, 2)))
##m.add(Conv2D(64, kernel_size=(5, 5), activation='relu'))
##m.add(MaxPooling2D(pool_size=(2, 2)))
##m.add(Conv2D(128, kernel_size=(5, 5), activation='relu'))
##m.add(MaxPooling2D(pool_size=(2, 2)))
##m.add(Conv2D(256, kernel_size=(5, 5), activation='relu'))
##m.add(MaxPooling2D(pool_size=(2, 2)))
##m.add(Flatten())
##m.add(Dense(128, activation='relu'))
##m.add(Dropout(0.5))
##m.add(Dense(5, activation='softmax'))
##
### setting and training
##m.compile(loss="categorical_crossentropy", metrics=['accuracy'])
##history  = m.fit(training_set, batch_size=32, epochs=25,verbose=0)
##m.save("ModelWithVeryingTheNumberOfFilters.h5")
##print(history.history["accuracy"])
##print(training_set.class_names)

print("------ Calling model ------")
filter_model = load_model("ModelWithVeryingTheNumberOfFilters.h5")

# testing
print("Testing.")
score = filter_model.evaluate(test_set, verbose=0)
print('Test accuracy:', score[1])
#return m
print("Completed successfully")
filter_model.summary()

###def test_image(m, image_file) :
##""" Classifies the given image using the given model. """
### load the image
##image_file = "flowers/rose/12240303_80d87f77a3_n.jpg"
##img = preprocessing.image.load_img(image_file,target_size=(100,100))
##img_arr = preprocessing.image.img_to_array(img)
##
### show the image
##plt.imshow(img_arr.astype("uint8"))
##plt.show()
##
### classify the image
##img_cl = img_arr.reshape(1,100,100,3)
##score = m.predict(img_cl)
##print(score)


print("------ Calling best model ------")
best_model = load_model("ModelWithPooling.h5")

print("Classifies the 10 images using the best model.\n")
# load the image
image_file = ["TenImages/daisy1.jpg", "TenImages/daisy2.jpg",
              "TenImages/dandelion1.jpg", "TenImages/dandelion2.jpg",
              "TenImages/rose1.jpg", "TenImages/rose2.jpg",
              "TenImages/sunflower1.jpg", "TenImages/sunflower2.jpg",
              "TenImages/tulip1.jpg", "TenImages/tulip2.jpg"]

for i in image_file:
    img = preprocessing.image.load_img(i,target_size=(100,100))
    img_arr = preprocessing.image.img_to_array(img)

    # show the image
    plt.imshow(img_arr.astype("uint8"))
    plt.show()

    # classify the image
    img_cl = img_arr.reshape(1,100,100,3)
    score = best_model.predict(img_cl)
    print(score)
