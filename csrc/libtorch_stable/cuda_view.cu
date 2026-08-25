#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/csrc/stable/device.h>
#include <torch/csrc/stable/c/shim.h>
#include <torch/headeronly/version.h>
#include <cuda_runtime.h>

#include <array>
#include <optional>

// torch < 2.11 compatibility (e.g. the ROCm ATOM image's torch 2.10):
// torch::stable::Tensor has no layout() and torch::stable::from_blob has no
// deleter parameter there, and ATen headers refuse to compile in a
// TORCH_TARGET_VERSION TU. Implement with the 2.10 stable API only. Without a
// deleter the view cannot own its backing memory, so the backing CPU tensor
// (the caller's pinned tensor, or the pinned copy made here for non-pinned
// inputs) is kept alive in a process-lifetime registry keyed by device
// pointer. Views are long-lived in every vLLM caller (UVA buffers, CPU-offloaded
// weights), so the never-freed registry is an accepted trade-off for this
// compatibility path.
#if (TORCH_VERSION_MAJOR < 2) || \
    (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR < 11)
#include <mutex>
#include <unordered_map>

namespace {
std::unordered_map<void*, torch::stable::Tensor>& uva_keepalive_registry() {
  static std::unordered_map<void*, torch::stable::Tensor> reg;
  return reg;
}
std::mutex& uva_keepalive_mutex() {
  static std::mutex m;
  return m;
}
void uva_keepalive(void* device_ptr, const torch::stable::Tensor& backing) {
  std::lock_guard<std::mutex> g(uva_keepalive_mutex());
  uva_keepalive_registry().insert_or_assign(device_ptr, backing);
}
bool uva_is_pinned(const torch::stable::Tensor& t) {
  std::array<StableIValue, 2> stack{torch::stable::detail::from(t),
                                    torch::stable::detail::from(std::nullopt)};
  TORCH_ERROR_CODE_CHECK(torch_call_dispatcher("aten::is_pinned", "",
                                               stack.data(), TORCH_ABI_VERSION));
  return torch::stable::detail::to<bool>(stack[0]);
}
}  // namespace

torch::stable::Tensor get_cuda_view_from_cpu_tensor(
    torch::stable::Tensor& cpu_tensor) {
  STD_TORCH_CHECK(cpu_tensor.device().is_cpu(), "Input tensor must be on CPU");
  const auto dtype = cpu_tensor.scalar_type();
  const torch::stable::Device cuda_dev(torch::headeronly::DeviceType::CUDA);
  if (cpu_tensor.numel() == 0) {
    return torch::stable::empty(cpu_tensor.sizes(), dtype,
                                torch::headeronly::Layout::Strided, cuda_dev);
  }
  torch::stable::Tensor backing = cpu_tensor;
  if (!uva_is_pinned(backing)) {
    // Make a pinned, contiguous copy (torch handles the host allocation).
    torch::stable::Tensor contiguous_cpu = torch::stable::contiguous(cpu_tensor);
    torch::stable::Tensor pinned = torch::stable::empty(
        contiguous_cpu.sizes(), dtype, torch::headeronly::Layout::Strided,
        cpu_tensor.device(), /*pin_memory=*/true);
    torch::stable::copy_(pinned, contiguous_cpu);
    backing = pinned;
  }
  void* host_ptr = const_cast<void*>(backing.mutable_data_ptr());
  void* device_ptr = nullptr;
  cudaError_t err = cudaHostGetDevicePointer(&device_ptr, host_ptr, 0);
  STD_TORCH_CHECK(err == cudaSuccess,
                  "cudaHostGetDevicePointer failed: ", cudaGetErrorString(err));
  uva_keepalive(device_ptr, backing);
  return torch::stable::from_blob(device_ptr, backing.sizes(), backing.strides(),
                                  cuda_dev, dtype);
}

#else  // torch >= 2.11: stable-ABI implementation

// This function assumes that `cpu_tensor` is a CPU tensor,
// and that UVA (Unified Virtual Addressing) is enabled.
torch::stable::Tensor get_cuda_view_from_cpu_tensor(
    torch::stable::Tensor& cpu_tensor) {
  STD_TORCH_CHECK(cpu_tensor.device().is_cpu(), "Input tensor must be on CPU");

  const auto dtype = cpu_tensor.scalar_type();
  const auto layout = cpu_tensor.layout();
  const torch::stable::Device cuda_dev(torch::headeronly::DeviceType::CUDA);

  // handle empty tensor
  if (cpu_tensor.numel() == 0) {
    return torch::stable::empty(cpu_tensor.sizes(), dtype, layout, cuda_dev);
  }

  std::array<StableIValue, 2> is_pinned_stack{
      torch::stable::detail::from(cpu_tensor),
      torch::stable::detail::from(std::nullopt)};
  TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
      "aten::is_pinned", "", is_pinned_stack.data(), TORCH_ABI_VERSION));
  if (torch::stable::detail::to<bool>(is_pinned_stack[0])) {
    // If CPU tensor is pinned, directly get the device pointer.
    void* host_ptr = const_cast<void*>(cpu_tensor.mutable_data_ptr());
    void* device_ptr = nullptr;
    cudaError_t err = cudaHostGetDevicePointer(&device_ptr, host_ptr, 0);
    STD_TORCH_CHECK(err == cudaSuccess, "cudaHostGetDevicePointer failed: ",
                    cudaGetErrorString(err));

    return torch::stable::from_blob(
        device_ptr, cpu_tensor.sizes(), cpu_tensor.strides(), cuda_dev, dtype,
        [base = cpu_tensor](void*) {});  // keep cpu tensor alive
  }

  // If CPU tensor is not pinned, allocate a new pinned memory buffer.
  torch::stable::Tensor contiguous_cpu = torch::stable::contiguous(cpu_tensor);
  size_t nbytes = contiguous_cpu.numel() * contiguous_cpu.element_size();

  void* host_ptr = nullptr;
  cudaError_t err = cudaHostAlloc(&host_ptr, nbytes, cudaHostAllocMapped);
  if (err != cudaSuccess) {
    STD_TORCH_CHECK(false, "cudaHostAlloc failed: ", cudaGetErrorString(err));
  }

  err = cudaMemcpy(host_ptr, contiguous_cpu.const_data_ptr(), nbytes,
                   cudaMemcpyDefault);
  if (err != cudaSuccess) {
    cudaFreeHost(host_ptr);
    STD_TORCH_CHECK(false, "cudaMemcpy failed: ", cudaGetErrorString(err));
  }

  void* device_ptr = nullptr;
  err = cudaHostGetDevicePointer(&device_ptr, host_ptr, 0);
  if (err != cudaSuccess) {
    cudaFreeHost(host_ptr);
    STD_TORCH_CHECK(
        false, "cudaHostGetDevicePointer failed: ", cudaGetErrorString(err));
  }

  auto deleter = [host_ptr](void*) { cudaFreeHost(host_ptr); };

  return torch::stable::from_blob(device_ptr, contiguous_cpu.sizes(),
                                  contiguous_cpu.strides(), cuda_dev,
                                  contiguous_cpu.scalar_type(), deleter);
}

#endif  // torch version guard
