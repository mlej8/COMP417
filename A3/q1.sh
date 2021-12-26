#! /usr/bin/bash

# testing different num of components
for num_components in {3..18}
do
rm classifier.state
python geoclass.py --num_components $num_components >> logss
done