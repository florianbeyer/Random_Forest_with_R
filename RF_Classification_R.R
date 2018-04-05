# Created on Thu Apr 05  2018
# @author: Florian Beyer
# Classification using Random Forest in R


######## required packages ##########
library(randomForest)
library(raster)
library(rgdal)
library(caret)


######## input data ##########
# set working directory
setwd("E:/Klassifikation/")

# training data as shp
training <- readOGR('shp/cal.shp')

# validation data as shp
validation <- readOGR('shp/val.shp')

# geotiff - remote sensing image
img <- brick('images/stack.tif')

######## plot image ##########

plotRGB(img, r=7, g=5, b=4, stretch='lin')

######## preparing data ##########

# extract pixel values from training shp
roi_data <- extract(img,training,df=TRUE)

# add land cover (lc) classes as factor variable
roi_data$lc <- as.factor(training$class[roi_data$ID])
# roi_data$desc <- as.factor(training$class[roi_data$ID])


set.seed(1234567890)

# current column names
colnames(roi_data)

#new column names
colnames(roi_data) <- c('ID','b1','b2','b3','b4','b5','b6','b7','b8','b9','b10','b11','b12','b13','b14','lc')

# subset: only image bands
cols <- c('b1','b2','b3','b4','b5','b6','b7','b8','b9','b10','b11','b12','b13','b14','lc')
rois <- roi_data[cols]

######## train Random Forest ##########
# replace NA's with 0
rois$b14[is.na(rois$b14)]<-0 

# train rf
rf <- randomForest(lc ~ b1+b2+b3+b4+b5+b6+b7+b8+b9+b10+b11+b12+b13+b14,
                   # lc ~ ., # this is the same
                   data=rois,
                   ntree=500 ,
                   importance=TRUE)
# show results
print(rf)

# available attrributes in "rf"
attributes(rf)
rf$confusion


######### show accuracy of training #######


# predicting trainingdata with RF model
p1 <- predict(rf, rois)

head(p1)
head(rois$lc)

confusionMatrix(p1, rois$lc)

######## error rate ##########

plot(rf)

# no better model after 100 Trees


######## tune RF ###########
# tuning with mtry

dim(rois)
names(rois)

tune_RF <- tuneRF(rois[,-15], rois[,15], # rois[,-15] = all bands,  rois[,15] = only our classes
                  stepFactor = 0.5, # at each iteration, mtry is inflated (or deflated) by this value
                  plot = T, # whether to plot the OOB error as function of mtry
                  ntreeTry = 100, # number of trees used at the tuning step, because:
                  # according "plot(rf)" 500 trees are not necessary since the error is stable after 100 trees
                  trace=T, # whether to print the progress of the search
                  improve = 0.05)  # the (relative) improvement in OOB error must be by this much for the search to continue

print(tune_RF)
# mtry with smallest error should be used for train RF
# in this case mtry = 3 is already the best choice

# AFTER TUNING, best choice would be:
rf <- randomForest(lc ~ b1+b2+b3+b4+b5+b6+b7+b8+b9+b10+b11+b12+b13+b14 ,
data=rois,
ntree=100,
mtry = 3,
importance=TRUE)

######## histogram #########
# number of nodes in the tree
hist(treesize(rf),
     main ="number of nodes in the tree",
     col="green")


######## variable importance ##########
varImpPlot(rf)

importance(rf)

### MeanDecreaseAccuracy
  # It tests how worse the model performs without each variabel
    # permutation importance
    # mean decrease in classification accuracy after
    # permuting Xj over all trees
    # for variables of different types: unbiased only when subsampling is used
    importance(rf, type = 1)

    # raw importance
    importance(rf, type = 1, scale = F)
    # scaled importance (z-score)
    importance(rf, type = 1, scale = T)

### MeanDecreaseGini
  # It measures how pure the nodes are at the end of the tree without each variable
    # Gini importance
    # mean Gini gain produced by Xj over all trees
    # for variables of different types: biased in favor of continuous variables and variables with many categories
    importance(rf, type = 2)


###Used variables
# varUsed() give the number, the variables were used in rf model
used_vars <- data.frame(names(rois[-15]),varUsed(rf))
used_vars[order(used_vars$varUsed.rf., decreasing = T),c(1,2)]

### Partial Depence Plot
# rf-model, training data, variable, class
partialPlot(rf, rois, b8, "1")
# in this case, vriable "b8" is important with values more that 41
summary(rois$b8)
hist(rois$b8, col="blue")


### extract single trees:
# get first tree:
getTree(rf, 1, labelVar = T)
# -1 means terminal node


# Multi-dimensional Scaling Plot of Proximity Matrix
### doesn't work!
#MDSplot(rf, rois$lc)


######## predict validation #########

# the same preparation as with the training data
val_data <- extract(img,validation,df=TRUE)
val_data$lc <- as.factor(validation$class[val_data$ID])
set.seed(123)
colnames(val_data)
colnames(val_data) <- c('ID','b1','b2','b3','b4','b5','b6','b7','b8','b9','b10','b11','b12','b13','b14','lc')
cols_val <- c('b1','b2','b3','b4','b5','b6','b7','b8','b9','b10','b11','b12','b13','b14','lc')
rois_val <- val_data[cols]

# predict validation data
p2 <- predict(rf, rois_val)

# show (independent) accuracy
confusionMatrix(p2, rois_val$lc)



### Plotting ROC curve and calculating AUC metric (AUC = accuracy)
library(pROC)
# predict class probablilities
p3 <- predict(rf, rois_val, type='prob')
print(p3) # class probabilities

auc <- auc(rois_val$lc,p3[,2])
plot(roc(rois_val$lc,p3[,2]))


######## predict on entire image ##########
# prepare new image dataset:
class_img <- img
names(class_img) <- c('b1','b2','b3','b4','b5','b6','b7','b8','b9','b10','b11','b12','b13','b14')

# predict the whole image
rf_pred <- predict(class_img, model=rf, na.rm=T)


######## show results ##########

colors_ <- palette(colors()[1:12])

plot(rf_pred, col=colors_)


######## save classification image as geotiff ##########
# georeference is already in the re_pred
classification_image <- writeRaster(rf_pred,'RF_Classification_withR.tif','GTiff', overwrite=TRUE)


