#include <iostream>
#include <string>
#include "config.h"

int main(int argc, char* argv[]) {

    if (argc != 2) {
        //OPTIMIZE  support command test case
        std::cerr <<"Usage: " << argv[0] << " <path/to/<config>.yaml" << std::endl;
    }

    std::string yamlFile = argv[1];
    ConfigParser config;

    YamlConfigParser::parseFile(yamlFile,config);
    return 0;
}

