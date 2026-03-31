#ifndef _DETECT_H_
#define _DETECT_H_

#include <string>
#include <vector>
#include "NvInfer.h"

namespace nvinfer1
{
	template <typename T>
	void write(char*& buffer, const T& val)
	{
		*reinterpret_cast<T*>(buffer) = val;
		buffer += sizeof(T);
	}

	template <typename T>
	void read(const char*& buffer, T& val)
	{
		val = *reinterpret_cast<const T*>(buffer);
		buffer += sizeof(T);
	}

	class Detect :public IPluginV2DynamicExt
	{
	public:
		Detect();
		Detect(const void* data, size_t length);
		Detect(const uint32_t n_anchor_, const uint32_t _n_classes_,
			const uint32_t n_grid_h_, const uint32_t n_grid_w_/*,
			const uint32_t &n_stride_h_, const uint32_t &n_stride_w_*/);
		~Detect();
		int getNbOutputs()const noexcept override
		{
			return 1;
		}
		DimsExprs getOutputDimensions(int index, const DimsExprs* inputs, int nbInputDims,
			IExprBuilder& exprBuilder) noexcept  override;
		int initialize() noexcept  override
		{
			return 0;
		}
		void terminate() noexcept  override
		{
		}
		size_t getWorkspaceSize(const PluginTensorDesc* inputs, int nbInputs, const PluginTensorDesc* outputs,
			int nbOutputs) const noexcept  override;
		int enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc,
			const void* const* inputs, void* const* outputs, void* workspace,
			cudaStream_t stream) noexcept override;

		size_t getSerializationSize() const noexcept  override;
		void serialize(void* buffer) const noexcept  override;
		const char* getPluginType() const noexcept  override
		{
			return "DETECT_TRT";
		}
		const char* getPluginVersion() const noexcept  override
		{
			return "2.0";
		}
		void destroy() noexcept  override
		{
			delete this;
		}
		void setPluginNamespace(const char* pluginNamespace) noexcept  override
		{
			_s_plugin_namespace = pluginNamespace;
		}
		const char* getPluginNamespace() const  noexcept override
		{
			return _s_plugin_namespace.c_str();
		}
		DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept override;
		void attachToContext(
			cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) noexcept override;
		void configurePlugin(const DynamicPluginTensorDesc* in, int nbInput, const DynamicPluginTensorDesc* out, int nbOutput) noexcept override;
		void detachFromContext() noexcept override;
		bool supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept override
		{
			return inOut[pos].format == TensorFormat::kLINEAR && inOut[pos].type == DataType::kFLOAT;
		}
		IPluginV2DynamicExt* clone() const noexcept override;
	private:
		
		uint32_t _n_anchor;
		uint32_t _n_classes;
		uint32_t _n_grid_h;
		uint32_t _n_grid_w;
		//uint32_t _n_stride_h;
	//	uint32_t _n_stride_w;
		uint64_t _n_output_size;
		std::string _s_plugin_namespace;
	}; //end detect

	class DetectPluginCreator : public IPluginCreator
	{
	public:
		DetectPluginCreator();
		~DetectPluginCreator() override = default;
		const char* getPluginName()const noexcept  override;
		const char* getPluginVersion() const  noexcept override;
		const PluginFieldCollection* getFieldNames() noexcept  override;
		IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept  override;
		IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept  override;
		void setPluginNamespace(const char* libNamespace)  noexcept override;
		const char* getPluginNamespace() const noexcept  override;
	private:
		std::string _s_name_space;
		static PluginFieldCollection _fc;
		static std::vector<PluginField> _vec_plugin_attributes;
	};//end detect creator

}//end namespace nvinfer1



#endif
