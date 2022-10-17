from tensorflow.keras import preprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Rescaling, Conv2D, Dense, Flatten, MaxPooling2D, Dropout, GlobalAveragePooling2D
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

print("------ Calling best model ------")
best_model = load_model("ModelWithPooling.h5")

print("Classifies the given images using the best model.\n")
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


    
    


                                                              
    
    
