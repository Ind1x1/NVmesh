#include "config.h"
#include <cstdint>
#include <iostream>
#include <unistd.h>
#include <cstdlib>
#include <stdexcept>
#include <cctype>

bool RangeConfig::isRangeMode() const {
    return !range.empty() && !scale_type.empty();
}

bool RangeConfig::isVectorMode() const {
    return !vector_range.empty();
}

std::vector<std::int64_t> RangeConfig::generateVectorValues() const {
    return vector_range;
}

std::vector<std::int64_t> RangeConfig::getAllValues() const {
    if (isVectorMode()) {
        return generateVectorValues();
    } else if (isRangeMode()) {
        return generateRangeValues();
    }
    return {};
}

std::vector<std::int64_t> RangeConfig::generateRangeValues() const {
    std::vector<std::int64_t> result;
    if (range.empty()) throw std::invalid_argument("Invalid range: " + range);

    size_t colon_pos = range.find(':');
    if (colon_pos != std::string::npos) {
        std::string start_str = range.substr(0, colon_pos);
        std::string end_str = range.substr(colon_pos + 1);
        std::int64_t start_value = std::stoll(start_str);
        std::int64_t end_value = std::stoll(end_str);
        if (scale_type.empty()) {
            throw std::invalid_argument("Invalid scale type: " + scale_type);
        }
        else if (scale_type == "<<") {
            for (std::int64_t i = start_value; i <= end_value; i *= 2) {
                result.push_back(i);
            }
        }
        //OPTIMIZE：support more scale types
        else {
            throw std::invalid_argument("Invalid scale type: " + scale_type);
        }
    }
    else {
        throw std::invalid_argument("Invalid range: " + range);
    }
    return result;
}

bool BufferSizeConfig::isRangeMode() const {
    return !size.empty() && !scale_type.empty();
}

bool BufferSizeConfig::isVectorMode() const {
    return !vector_range.empty();
}

std::vector<std::int64_t> BufferSizeConfig::getAllValues() const {
    if (isVectorMode()) {
        return generateVectorValues();
    } else if (isRangeMode()) {
        return generateRangeValues();
    }
    return {};
}

std::int64_t BufferSizeConfig::convertToBytes(const std::string& size_str) const {
    if (size_str.empty()) {
        throw std::invalid_argument("Invalid size: " + size_str);
    }

    size_t pos = 0;
    while(pos < size_str.length() && std::isdigit(size_str[pos])) {
        pos++;
    }
    double value = std::stod(size_str.substr(0, pos));
    std::string unit = size_str.substr(pos);
    if (unit.empty() || unit == "B") {
        return static_cast<std::int64_t>(value);
    } else if (unit == "KB") {
        return static_cast<std::int64_t>(value * 1024);
    } else if (unit == "MB") {
        return static_cast<std::int64_t>(value * 1024 * 1024);
    } else if (unit == "GB") {
        return static_cast<std::int64_t>(value * 1024 * 1024 * 1024);
    } else {
        throw std::invalid_argument("Invalid unit: " + unit);
    }
}

std::vector<std::int64_t> BufferSizeConfig::generateVectorValues() const {
    std::vector<std::int64_t> result;
    result.reserve(vector_range.size());
    for (const auto& str : vector_range) {
        result.push_back(convertToBytes(str));
    }
    this->max_size = *std::max_element(result.begin(), result.end());
    this->min_size = *std::min_element(result.begin(), result.end());
    return result;
}

std::vector<std::int64_t> BufferSizeConfig::generateRangeValues() const {
    std::vector<std::int64_t> result;
    if (size.empty()) throw std::invalid_argument("Invalid size: " + size);
    size_t colon_pos = size.find(':');
    if (colon_pos != std::string::npos) {
        std::string start_str = size.substr(0, colon_pos);
        std::string end_str = size.substr(colon_pos + 1);
        std::int64_t start_valuse = convertToBytes(start_str);
        std::int64_t end_valuse = convertToBytes(end_str);
        if (scale_type.empty()) {
            throw std::invalid_argument("Invalid scale type: " + scale_type);
        }
        else if (scale_type == "<<") {
            for (std::int64_t i = start_valuse; i <= end_valuse; i *= 2) {
                result.push_back(i);
            }
        }
        //OPTIMIZE：support more scale types
        else {
            throw std::invalid_argument("Invalid scale type: " + scale_type);
        }
    }
    else {
        throw std::invalid_argument("Invalid size: " + size);
    }
    this->max_size = *std::max_element(result.begin(), result.end());
    this->min_size = *std::min_element(result.begin(), result.end());
    return result;
}

std::vector<std::int64_t> TestCaseConfig::getBufferSizes() const {
    return buffer_size.getAllValues();
}

std::vector<std::int64_t> TestCaseConfig::getSMValues() const {
    return num_sm.getAllValues();
}

std::vector<std::int64_t> TestCaseConfig::getPipeValues() const {
    return pipes.getAllValues();
}

bool TestCaseConfig::isBufferSizeRangeMode() const {
    return buffer_size.isRangeMode();
}

bool TestCaseConfig::isBufferSizeVectorMode() const {
    return buffer_size.isVectorMode();
}

bool TestCaseConfig::isSMRangeMode() const {
    return num_sm.isRangeMode();
}

bool TestCaseConfig::isSMVectorMode() const {
    return num_sm.isVectorMode();
}

bool TestCaseConfig::isPipesRangeMode() const {
    return pipes.isRangeMode();
}

bool TestCaseConfig::isPipesVectorMode() const {
    return pipes.isVectorMode();
}

bool YamlConfigParser::parseFile(const std::string& filename, ConfigParser& config) {
    try {
        YAML::Node root = YAML::LoadFile(filename);
        return parseFromNode(root, config)
    } catch (const YAML::Exception& e) {
        std::cerr << "Error: Fail to load ConfigYAML" << e.what() << std::endl;
        return false;
    }
}

bool YamlConfigParser::parseFromNode(const YAML::Node& root, ConfigParser& config) {
    try{
        if (root["Nodes"]) {
            parseNodes(root["Nodes"], config.nodes);
        }

        if (root["Config"]) {
            parseTestCase(root["Config"], config.test_cases);
        }

        return true;
    } catch (const YAML::Execption& e) {
        std::cerr << "Error: Fail to parse from ConfigYAML" << e.what() << std::endl;
        return false;
    }
};

void YamlConfigParser::parseNodes(const YAML::Node& nodesNode, std::vector<NodeConfig>& nodes) {
    for (const auto& node : nodesNode){
        NodeConfig nodeConfig;
        if (node["name"]) {
            nodeConfig.name  = node["name"].as<std::string>();
        }
        if (node["ip"]) {
            nodeConfig.ip = node["ip"].as<std::string>();
        }
        if (node["gpu_indices"]) {
            for (const auto& gpu : node["gpu_indices"]) {
                nodeConfig.gpu_indices.push_back(gpu.as<int>());
            }
        }

        node.push_back(nodeConfig);
    }
}

void YamlConfigParser::parseTestCases(const YAML::Node& configNode, std::vector<TestCaseConfig>& testCases) {
    for (const auto& caseNode : configNode) {
        TestCaseConfig testCase;

        if (caseNode["Case"]) {
            testCase.case_id = caseNode["Case"].as<int>();
        }
        
        if (caseNode["Engine"]) {
            testCase.engine = caseNode["Engine"].as<std::string>();
        }
        
        if (caseNode["iterations"]) {
            testCase.iterations = caseNode["iterations"].as<int>();
        }
        
        if (caseNode["Warmup"]) {
            testCase.warmup = caseNode["Warmup"].as<int>();
        }
        
        if (caseNode["loopCount"]) {
            testCase.loop_count = caseNode["loopCount"].as<int>();
        }
        
        // BufferSize
        if (caseNode["BufferSize"]) {
            parseBufferSize(caseNode["BufferSize"], testCase.buffer_size);
        }
        
        // NumSM
        if (caseNode["NumSM"]) {
            parseRangeConfig(caseNode["NumSM"], testCase.num_sm, "SM");
        }
        
        // Pipes
        if (caseNode["Pipes"]) {
            parseRangeConfig(caseNode["Pipes"], testCase.pipes, "pipes");
        }
        
        testCases.push_back(testCase);
    }
}

void YamlConfigParser::parseBufferSize(const YAML::Node& bufferNode, BufferSizeConfig& bufferConfig) {
    if (bufferNode.IsSequence()) {
        for (const auto& item : bufferNode) {
            if (item["Size"]) {
                if (item["Size"].IsScalar()) {
                    // 范围格式: "1024KB:16MB"
                    bufferConfig.size = item["Size"].as<std::string>();
                    if (item["ScaleType"]) {
                        bufferConfig.scale_type = item["ScaleType"].as<std::string>();
                    }
                } else if (item["Size"].IsSequence()) {
                    // 向量格式: [1024KB, 2MB, 4MB, 8MB, 16MB]
                    for (const auto& size : item["Size"]) {
                        bufferConfig.vector_range.push_back(size.as<std::string>());
                    }
                }
            }
        }
    }
}

void YamlConfigParser::parseRangeConfig(const YAML::Node& rangeNode, RangeConfig& rangeConfig, const std::string& key) {
    if (rangeNode.IsSequence()) {
        for (const auto& item : rangeNode) {
            if (item[key]) {
                if (item[key].IsScalar()) {
                    // 范围格式: "1:32"
                    rangeConfig.range = item[key].as<std::string>();
                    if (item["ScaleType"]) {
                        rangeConfig.scale_type = item["ScaleType"].as<std::string>();
                    }
                } else if (item[key].IsSequence()) {
                    // 向量格式: [1, 2, 4, 8, 16, 32]
                    for (const auto& value : item[key]) {
                        rangeConfig.vector_range.push_back(std::to_string(value.as<int>()));
                    }
                }
            }
        }
    }
}