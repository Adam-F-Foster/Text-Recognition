RUNNING THE PROJECT

1. packages to install
	a. numpy
	b. opencv
	c. sklearn
	d. matplotlib
	e.time
	f. emnist
	g. cnnKeras
	h. tensorflow
	i. keras
2. how to run the project
	a. create an image of text in any paint-like application
	b. use white text with a black background to write letters
	c. the file does not have to be saved with any specific sizing
	d. run "projectLetter.py"
	e. enter the file name as formatted "file.[extension]"
	f. the first part of the program will display predicted letters and times to predicted
	g. the second part of the program uses metrics.accuracy_score to make a bar plot of all of the accuracies.
	
3. extra details
	a. we would recommend limiting the images and labels size in projectLetter to <= 8000
	b. number of epochs ran can be changed in "cnnKeras" on line 47, in the "fit" method
	c. we used spyder in making this
	d. the pixel values in cnnKeras must be divided by 255 so the neural network has a range of 0-1 to work with
		rather than 0-255
	e. to_categoricals hot encodes the trainY and testY
	f. we only used training samples because it has approximately 98,000 samples, but then did a test_train_split on it with 20% of it being test data
