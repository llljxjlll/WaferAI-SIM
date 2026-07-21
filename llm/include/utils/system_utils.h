#pragma once
#include "common/config.h"
#include <iostream>
#include <string>
#include <utility>

using namespace std;

bool RandResult(int threshold);
int GetFromPairedVector(vector<pair<string, int>> &vector, string key);
CoreHWConfig *GetCoreHWConfig(int id);
// 多 die：按全局核 id 取硬件配置（同构复制）——映射到 die 内 local id 的模板，
// 避免隐藏的 % GRID_SIZE。单 die 时 global==local，与 GetCoreHWConfig 等价。
CoreHWConfig *GetCoreHWConfigForGlobal(int global_id);
int CeilingDivision(int a, int b);

void InitGrid(string workload_config_path, string hardware_config_path,
              string simulation_config_path, string mapping_config_path);
void InitGlobalMembers();
void SystemCleanup();

void DeleteCoreLogFiles();
void DeleteMemoryLogFiles();
void CloseLogFiles();
const char *GetCoreColor(int core_id);

void InitializeMemorySpec();
bool modifyNbrOfDevices(const std::string &inputPath,
                        const std::string &outputPath, int x);
void generateAddressMapping(int n, const std::string &outputFilename);
void generateDFAddressMapping(int n, const std::string &outputFilename);
bool generateGPUCacheJsonFile(int numDevices,
                              const std::string &filename = "output.json");

// void initialize_cache_structures();
// void init_dram_areas();
// void destroy_dram_areas();