//
//  ShapeInference.h
//  MNN
//
//  Created by MNN on 2020/04/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_PLUGIN_PLUGIN_KERNEL_HPP_
#define MNN_PLUGIN_PLUGIN_KERNEL_HPP_

#include <functional>
#include <string>
#include <unordered_map>
#include <utility>

#include <MNN/plugin/PluginContext.hpp>

namespace MNN {
namespace plugin {

template <typename KernelContextT>
class MNN_PUBLIC ComputeKernel {
public:
    ComputeKernel()                           = default;
    virtual ~ComputeKernel()                  = default;
    virtual bool compute(KernelContextT* ctx) = 0;
};

class MNN_PUBLIC CPUComputeKernel : public ComputeKernel<CPUKernelContext> {
public:
    using ContextT = CPUKernelContext;
    using KernelT  = CPUComputeKernel;

    CPUComputeKernel()                          = default;
    virtual ~CPUComputeKernel()                 = default;
    virtual bool init(CPUKernelContext* ctx) = 0;
    virtual bool compute(CPUKernelContext* ctx) = 0;
};

template <typename PluginKernelT>
class MNN_PUBLIC ComputeKernelRegistry {
public:
    typedef std::function<PluginKernelT*()> Factory;
    static std::unordered_map<std::string, Factory>* getFactoryMap() {
        static auto* map = new std::unordered_map<std::string, Factory>();
        return map;
    }

    static bool add(const std::string& name, Factory factory) {
        auto* map = getFactoryMap();
        return map->emplace(name, std::move(factory)).second;
    }

    static PluginKernelT* get(const std::string& name) {
        auto* map = getFactoryMap();
        const auto it = map->find(name);
        if (it == map->end()) {
            return nullptr;
        }
        return it->second ? it->second() : nullptr;
    }
};

template <typename PluginKernelT>
struct ComputeKernelRegistrar {
    ComputeKernelRegistrar(const std::string& name) {
        ComputeKernelRegistry<typename PluginKernelT::KernelT>::add(name, []() { // NOLINT
            return new PluginKernelT;                                            // NOLINT
        });
    }
};

#define REGISTER_PLUGIN_COMPUTE_KERNEL(name, computeKernel)                \
    namespace {                                                            \
    static auto _plugin_compute_kernel_##name##_ __attribute__((unused)) = \
        ComputeKernelRegistrar<computeKernel>(#name);                      \
    } // namespace

} // namespace plugin
} // namespace MNN

#endif // MNN_PLUGIN_PLUGIN_KERNEL_HPP_
