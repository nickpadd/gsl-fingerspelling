For each of the models:
-> Split the data into sessions and individuals. 
-> Train and validate with a set of the data and leave out 1/2 sessions per individual for the final test.
-> Evaluate each model with f1 (?).

# CNN roadmap
1. With and without hand recogniser.

2. Different Structures:
{ 1->16->linear / 1->16->32-linear/ 1->16->32->64->linear / until overfitting}

3. Different batch size and learning rate (?)

# VGG transfer model roadmap
! DO NOT FORGET TO PUSH DATA TO GPU FOR THE COLLAB FILE !

1. Unfreezing outer layers one by one until we see the model performing clearly worse.

2. Different learning rate and batch size, Dropout.

