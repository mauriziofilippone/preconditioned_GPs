create_dataset = function(dataset_name, FOLD, type="regression") {
    ## Load data and extract inputs (X) and labels (y)
    if(dataset_name == "concrete") {
        fileall = "../../datasets/regDatasets/Concrete_Data.csv"
        tmp = as.matrix(read.table(fileall, header=F, sep=","))
        X = tmp[,1:8]
        y = tmp[,9]
    }

    if(dataset_name == "powerplant") {
        fileall = "../../datasets/regDatasets/PowerPlant_Data.csv"
        tmp = as.matrix(read.table(fileall, header=F, sep=","))
        X = tmp[,1:4]
        y = tmp[,5]
    }

    if(dataset_name == "protein") {
        fileall = "../../datasets/regDatasets/Protein_Data.csv"
        tmp = as.matrix(read.table(fileall, header=F, sep=","))
        X = tmp[,1:9]
        y = tmp[,10]
    }

    if(dataset_name == "credit") {
        fileall = "../../datasets/classDatasets/Credit_Data.csv"
        tmp = as.matrix(read.table(fileall, header=F, sep=","))
        X = tmp[,1:24]
        y = tmp[,25]
    }

    if(dataset_name == "banknote") {
        fileall = "../../datasets/classDatasets/Banknote_Data.csv"
        tmp = as.matrix(read.table(fileall, header=F, sep=","))
        X = tmp[,1:4]
        y = tmp[,5]
    }

    if(dataset_name == "eeg") {
        fileall = "../../datasets/classDatasets/EEG_Data.csv"
        tmp = as.matrix(read.table(fileall, header=F, sep=","))
        X = tmp[,1:14]
        y = tmp[,15]
    }

    if(dataset_name == "spam") {
        fileall = "../../datasets/classDatasets/Spam_Data.csv"
        tmp = as.matrix(read.table(fileall, header=F, sep=","))
        X = tmp[,1:57]
        y = tmp[,58]
    }

    if(dataset_name == "wilt") {
        fileall = "../../datasets/classDatasets/Wilt_Data.csv"
        tmp = as.matrix(read.table(fileall, header=F, sep=" "))
        X = tmp[,1:5]
        y = tmp[,6]
    }
    
    rm(tmp)    

    n = dim(X)[1]
    d = dim(X)[2]

    ## Shuffle data
    ind.scramble = sample(c(1:n), n)
    X = X[ind.scramble,]
    y = y[ind.scramble]

    if(type=="classification") {
      ## In case some datasets have 0/1 labels turn them into -1/1 labels
      y[y == 0] = -1
    }

    NFOLDS = as.integer(sqrt(n))
    
    ## Associate each input vector with a number between 1 and NFOLDS 
    ind.folds = as.integer(seq(from=1, to=NFOLDS+1-1e-6, length.out=n))

    ind.test = which(ind.folds == FOLD)
    ind.train = which(ind.folds != FOLD)
    tmpX.train = X[ind.train,]
    tmpy.train = y[ind.train]

    tmpX.test = X[ind.test,]
    tmpy.test = y[ind.test]

    ## ## Normalize the features - between zero and one
    ## minxi = maxxi = rep(0, d)
    ## for(i in 1:d) {
    ##     minxi[i] = min(tmpX.train[,i])
    ##     maxxi[i] = max(tmpX.train[,i])
        
    ##     if(minxi[i] == maxxi[i]) browser()
        
    ##     tmpX.train[,i] = tmpX.train[,i] - minxi[i]
    ##     tmpX.train[,i] = tmpX.train[,i] / (maxxi[i] - minxi[i])
        
    ##     tmpX.test[,i] = tmpX.test[,i] - minxi[i]
    ##     tmpX.test[,i] = tmpX.test[,i] / (maxxi[i] - minxi[i])
    ##   }
    
    ## Normalize the features - zero mean and unit stdev
    mxi = sdxi = rep(0, d)
    for(i in 1:d) {
        mxi[i] = mean(tmpX.train[,i])
        sdxi[i] = sd(tmpX.train[,i])
        
        if(sdxi[i] == 0) browser()
        
        tmpX.train[,i] = tmpX.train[,i] - mxi[i]
        tmpX.train[,i] = tmpX.train[,i] / sdxi[i]
        
        tmpX.test[,i] = tmpX.test[,i] - mxi[i]
        tmpX.test[,i] = tmpX.test[,i] / sdxi[i]
      }

    if(type=="regression") {
        
        ## Normalize the labels
        my = mean(tmpy.train)
        sdy = sd(tmpy.train)

        tmpy.train = tmpy.train - my
        tmpy.train = tmpy.train / sdy

        tmpy.test = tmpy.test - my
        tmpy.test = tmpy.test / sdy
    }
    
    ## Create the DATA structure
    DATA = new.env()

    DATA$X = tmpX.train
    DATA$Xtest = tmpX.test
    
    DATA$y = tmpy.train
    DATA$ytest = tmpy.test
    
    DATA$n = dim(DATA$X)[1]
    DATA$ntest = dim(DATA$Xtest)[1]
    DATA$d = dim(DATA$X)[2]

    ## Save training/test data
    filesave = paste("FOLDS/", DATASET, "_", KERNEL_TYPE, "_Xtrain_", "_FOLD_", FOLD, sep="")
    write.table(DATA$X, file=filesave, col.names=F, row.names=F, quote=F)
    
    filesave = paste("FOLDS/", DATASET, "_", KERNEL_TYPE, "_Xtest_", "_FOLD_", FOLD, sep="")
    write.table(DATA$Xtest, file=filesave, col.names=F, row.names=F, quote=F)
    
    filesave = paste("FOLDS/", DATASET, "_", KERNEL_TYPE, "_ytrain_", "_FOLD_", FOLD, sep="")
    write.table(DATA$y, file=filesave, col.names=F, row.names=F, quote=F)
    
    filesave = paste("FOLDS/", DATASET, "_", KERNEL_TYPE, "_ytest_", "_FOLD_", FOLD, sep="")
    write.table(DATA$ytest, file=filesave, col.names=F, row.names=F, quote=F)
        
    DATA
}
