#ifndef _MISH_PLUGIN_H
#define _MISH_PLUGIN_H

#include <string>
#include <vector>
#include "NvInfer.h"


//https://github.com/wang-xinyu/tensorrtx
namespace nvinfer1
{
    class MishPlugin: public IPluginV2DynamicExt
    {
        public:
            explicit MishPlugin();
            MishPlugin(const void* data, size_t length);

            ~MishPlugin();

            int getNbOutputs() const  noexcept override
            {
                return 1;
            }

            DimsExprs getOutputDimensions(int index, const DimsExprs* inputs, int nbInputDims,
                IExprBuilder& exprBuilder) noexcept  override;

            int initialize() noexcept  override;

            virtual void terminate() noexcept  override {}

            virtual size_t getWorkspaceSize(const PluginTensorDesc* inputs, int nbInputs,
                const PluginTensorDesc* outputs, int nbOutputs) const noexcept  override { return 0;}

			int enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc,
				const void* const* inputs, void* const* outputs, void* workspace,
				cudaStream_t stream) noexcept override;

            virtual size_t getSerializationSize() const noexcept  override;

            virtual void serialize(void* buffer) const noexcept  override;

            bool supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept override {
                return inOut[pos].format == TensorFormat::kLINEAR && inOut[pos].type == DataType::kFLOAT;
            }

            const char* getPluginType() const noexcept  override;

            const char* getPluginVersion() const noexcept  override;

            void destroy()  noexcept override;

            IPluginV2DynamicExt* clone() const noexcept  override;

            void setPluginNamespace(const char* pluginNamespace) noexcept  override;

            const char* getPluginNamespace() const  noexcept override;

            DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept override;

            void attachToContext(
                    cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator)noexcept override;

            void configurePlugin(const DynamicPluginTensorDesc* in, int nbInput, const DynamicPluginTensorDesc* out, int nbOutput)noexcept override;

            void detachFromContext()noexcept override;

            int input_size_;
        private:
            void forwardGpu(const float *const * inputs, float* output, cudaStream_t stream, int batchSize = 1);
            int thread_count_ = 256;
            std::string mPluginNamespace;
    };

    class MishPluginCreator : public IPluginCreator
    {
	        public:
            MishPluginCreator();

            ~MishPluginCreator() override = default;

            const char* getPluginName() const noexcept  override;

            const char* getPluginVersion() const  noexcept override;

            const PluginFieldCollection* getFieldNames() noexcept  override;

            IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc)  noexcept override;

            IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept  override;

			void setPluginNamespace(const char* libNamespace) noexcept  override;

			const char* getPluginNamespace() const noexcept  override;

	        private:
            std::string mNamespace;
            static PluginFieldCollection mFC;
            static std::vector<PluginField> mPluginAttributes;
    };
}
#endif 
