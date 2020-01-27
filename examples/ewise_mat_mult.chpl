var training_inputs1  : [1..5, 1..5] real;
for idxy in training_inputs1.domain do
	training_inputs1[idxy] = idxy[1]+idxy[2];
writeln(training_inputs1, "\n\n");


var training_inputs2  : [1..5, 1..5] real;
for idxy in training_inputs2.domain do
	training_inputs2[idxy] = idxy[1];
writeln(training_inputs2, "\n\n");





training_inputs1 = training_inputs1*training_inputs2;
writeln(training_inputs1);
