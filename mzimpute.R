mzimpute <- function(v){ 
	if(mean(v==0,na.rm=TRUE) > 0.5) impt <- 0
	else impt <- mean(v, na.rm=TRUE)
	v[is.na(v)] <- impt
	return(v) }
