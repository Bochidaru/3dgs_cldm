1. install gcc g++ 9
   * sudo apt update
   * sudo apt install gcc-9 g++-9
   * sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 90 && sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 90
   * gcc --version

2. download cudatoolkit 11.8:
   * wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
   * sudo sh cuda_11.8.0_520.61.05_linux.run  (remember to only tick X on CudaToolkit, untick other section, and chooose yes when symlink question appear)
   * nvcc --version (Build cuda_11.8 is ok)

3. install some important package:
   * conda install -c conda-forge ittapi -y
   * sudo apt install libglm-dev colmap libtiff-dev libtiff-tools
   * sudo ln -s /usr/lib/x86_64-linux-gnu/libtiff.so.6 /usr/lib/x86_64-linux-gnu/libtiff.so.5

4. create stub libittnotify
`
echo '#ifdef __cplusplus
extern "C" {
#endif
#include <stdarg.h>
void iJIT_NotifyEvent(int a, ...) {}
void iJIT_NotifyEventW(int a, ...) {}
int  iJIT_IsProfilingActive(void) { return 0; }
int  iJIT_GetNewMethodID(void) { return 1; }
int  iJIT_GetNewMethodIDEx(void) { return 1; }
void iJIT_NotifyEventStr(int a, ...) {}
void iJIT_NotifyEventEx(int a, ...) {}
#ifdef __cplusplus
}
#endif' > "$CONDA_PREFIX/lib/itt_stub.c"
`
   * gcc -shared -fPIC -O2 -o "$CONDA_PREFIX/lib/libittnotify.so" "$CONDA_PREFIX/lib/itt_stub.c"
   * ls -l "$CONDA_PREFIX/lib/libittnotify.so"

5. set environment variables (copy all the below and run)
`
export LD_PRELOAD=$CONDA_PREFIX/lib/libittnotify.so
export TORCH_CUDA_ARCH_LIST="8.6"
echo 'export LD_PRELOAD=$CONDA_PREFIX/lib/libittnotify.so${LD_PRELOAD:+:$LD_PRELOAD}' >> ~/.zshrc
echo 'export TORCH_CUDA_ARCH_LIST="8.6"' >> ~/.zshrc
source ~/.zshrc
echo $LD_PRELOAD
echo $TORCH_CUDA_ARCH_LIST
`

6. install environment.yml
   * conda env update -f environment.yml

7. create run.sh and run (only on linux)
   * chmod +x run.sh
   * ./run.sh