# build shell
echo "################ test building... ################"

# clean
echo "################ cleaning... ################"
rm -rf build
mkdir build
cd build
mkdir bin

# comfigure cmake
echo "################ configuring cmake... ################"
cmake ..
make -j8 VERBOSE=1
echo "################ test build finished ################"

# run exe
echo "################ test runing... ################"
./bin/test
echo "################ test run finied ################"