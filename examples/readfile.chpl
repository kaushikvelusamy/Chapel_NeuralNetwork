use IO;

var c = openreader('../mnist_neural_nets/mnist_dataset/mnist_test_10.csv');
writeln("\n");

var label_arr : [0..9] int;
var pixels_arr : [0..9] int;
var line_counter = 0 : int;

for line in c.lines() {
  label_arr[line_counter] = line[1] : int;
  //pixels_arr[line_counter] = line[2] : int;

  line_counter = line_counter + 1;
}
writeln(label_arr);

writeln("\n");
writeln(pixels_arr);
