Small matlab demo, so.m, to explore an answer to this stackoverflow question:

http://dsp.stackexchange.com/questions/1122/how-to-compute-2d-displacement-vector-for-binary-image-registration

so.m tries to register two (sparse) binary images against each other using a combination of found SIFT points from the edge maps of both images and rigid registration based on cross correlation.

The function needs the VLfeat (www.vlfeat.org) package and the Robust estimation from
Peter Kovesi (http://www.csse.uwa.edu.au/~pk/research/matlabfns) to run.