def predict_image(img_path,normalizer,km,best_model):
    # Load the image from a path
    img = Image.open(img_path)  
    numpydata = asarray(img)
    
    # Get the mask of the image
    mask = get_mask(numpydata)

    #Convert to grayscale
    if len(numpydata.shape)==3:
        gray_data = numpydata.mean(axis=2)
        gray_data = gray_data.astype(np.uint8)
    else:
        gray_data = numpydata.astype(np.uint8)

    # Get dense keypoints
    dense_kp = generate_dense_keypoints(gray_data,mask)
    
    # Compute descriptors
    _, des = sift.compute(gray_data, dense_kp, mask)


    # Create the histogram of the image
    im_features = np.zeros(k, dtype=np.float32)
    
    if des is not None:
        for feature in des:
            feature = feature.reshape(1, -1)
            idx = kmeans.predict(feature)
            im_features[idx] += 1
    
    # Normalize the histogram
    im_features = normalizer.transform(im_features.reshape(1,-1))

    # Predict the category using the svm model
    y_pred = best_model.predict(im_features)
    
    return categories[y_pred[0]]