from nets.model_predict import model_predict   
import tifffile

if __name__ == "__main__":
#--------------------------------------------------------------------------------------------------------------#
    count           = False                        #Specifies whether to perform pixel counting (i.e., area calculation) and scaling for the target. Default: False
    name_classes    = ["background","OWT"]         #Category
    image= "./predict_data/1.tif"                  #Input data
    Save=False                                     #Should prediction results be saved? Default: False
    result_out="./predict_result/9.tif"            #Prediction results saved
#--------------------------------------------------------------------------------------------------------------#



    model_predict = model_predict()

    image = tifffile.imread(image)
    r_image = model_predict.detect_image(image, count=count, name_classes=name_classes)
    r_image.show()

    if Save:
        r_image.save(result_out, 'TIFF')
    print("Prediction successful!!!")



