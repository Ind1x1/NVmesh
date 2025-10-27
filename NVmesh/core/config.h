#pragma once

#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>

struct NodeConfig {
    std::string name;
    std::string ip;
    std::vector<int> gpu_indices;
    
    NodeConfig() = default;
    NodeConfig(const std::string& n, const std::string& i, const std::vector<int>& gpus)
        : name(n), ip(i), gpu_indices(gpus) {}
};

struct BufferSizeConfig {
    std::string size;
    std::string scale_type;
    std::vector<std::string> vector_range;
    mutable std::int64_t max_size;
    mutable std::int64_t min_size;

    BufferSizeConfig() = default;
    BufferSizeConfig(const std::string& s, const std::string& st) : size(s), scale_type(st) {}
    BufferSizeConfig(const std::vector<std::string>& vec) : vector_range(vec) {}

    bool isRangeMode() const;
    bool isVectorMode() const;

    std::vector<std::int64_t> getAllValues() const;
private:
    std::vector<std::int64_t> generateRangeValues() const;
    std::vector<std::int64_t> generateVectorValues() const;
    std::int64_t convertToBytes(const std::string& size_str) const;
};


struct RangeConfig {
    std::string range;
    std::string scale_type;
    std::vector<std::int64_t> vector_range;

    RangeConfig() = default;
    RangeConfig(const std::string& r, const std::string& s) : range(r), scale_type(s) {}
    RangeConfig(const std::vector<std::int64_t>& vec) : vector_range(vec) {}

    bool isRangeMode() const;
    bool isVectorMode() const;

    std::vector<std::int64_t> getAllValues() const;
private:
    std::vector<std::int64_t> generateRangeValues() const;
    std::vector<std::int64_t> generateVectorValues() const;
};

struct TestCaseConfig {
    int case_id;
    BufferSizeConfig buffer_size;
    std::string engine;
    int iterations;
    int warmup;
    int loop_count;
    RangeConfig num_sm;
    RangeConfig pipes;

    TestCaseConfig() : case_id(0), engine(""), iterations(0), warmup(0), loop_count(0) {}

    std::vector<std::int64_t> getBufferSizes() const;
    std::vector<std::int64_t> getSMValues() const;
    std::vector<std::int64_t> getPipeValues() const;

    //FIXME: need to be modified
    bool isBufferSizeRangeMode() const;
    bool isBufferSizeVectorMode() const;
    bool isSMRangeMode() const;
    bool isSMVectorMode() const;
    bool isPipesRangeMode() const;
    bool isPipesVectorMode() const;

};

struct ConfigParser {
    std::vector<NodeConfig> nodes;
    std::vector<TestCaseConfig> test_cases;
};


class YamlConfigParser {
public:
    static bool parseFile(const std::string& filename, ConfigParser& config);
    static bool parseFromNode(const YAML::Node& root, ConfigParser& config);
private:
    static void parseNodes(const YAML::Node& nodesNode, std::vector<NodeConfig>& nodes);
    static void parseTestCases(const YAML::Node& configNode, std::vector<TestCaseConfig>& testCases);
    static void parseBufferSize(const YAML::Node& bufferNode, BufferSizeConfig& bufferConfig);
    static void parseRangeConfig(const YAML::Node& rangeNode, RangeConfig& rangeConfig, const std::string& key);
};