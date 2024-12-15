# Configuration
You can configurate the options for grid-search at "grid-search configuration.json", options for random-search at random-search configuration

null values means the hyperpanameter will not be tested in the optimization, these are there so you can easily include and exclude hyperparameters from the testing

Random-search is handled with dicts containing "min_int" and "max_int" for int values.
In float-point values you have to use "min_float" and "max_float"

# Default files
Plese do not change anything at the default files. 
Those are mainly for being used as a basis in case something goes wrong. Just copy the values there to the configuration files.

In case there is an exception because of a change made at the configuration files, the default files are used instead.