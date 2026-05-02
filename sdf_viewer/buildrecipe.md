cd /Users/erichan/Documents/Development/jardins_racine
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target sdf_viewer -j$(sysctl -n hw.logicalcpu)
./build/sdf_viewer
# or pass your own XML:
./build/sdf_viewer CPlantBox/modelparameter/structural/rootsystem/Zea_mays_6_Leitner_2014.xml