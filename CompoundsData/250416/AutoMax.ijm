// Ask user to choose an output directory
outputDir = getDirectory("Choose directory to save max projections");

// Get list of open image titles
imageList = getList("image.titles");

// Loop through each image
for (i = 0; i < imageList.length; i++) {
    selectWindow(imageList[i]);

    // Check if it's a stack (i.e. has multiple slices)
    stackSize = nSlices;
    if (stackSize > 1) {
        // Run Z Projection
        run("Z Project...", "projection=[Max Intensity]");

        // The new projection is usually titled "MAX_" + original name
        projTitle = "MAX_" + imageList[i];
        selectWindow(projTitle);

        // Save as TIFF
        saveAs("Tiff", outputDir + projTitle + ".tif");

        // Optional: close projection
        close();
    }
}

run("Close All");