1. install gcc g++ 9
   * sudo apt update
   * sudo apt install gcc-9 g++-9
   * sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 90 && sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 90
   * gcc --version, sudo update-alternatives --config gcc && sudo update-alternatives --config g++

2. download cudatoolkit 11.8:
   * wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
   * sudo sh cuda_11.8.0_520.61.05_linux.run
   * nvcc --version

3. install some important package:
   * conda install -c conda-forge ittapi -y
   * sudo apt install libglm-dev colmap libtiff5 libtiff-dev libtiff-tools
   * sudo ln -s /usr/lib/x86_64-linux-gnu/libtiff.so.6 /usr/lib/x86_64-linux-gnu/libtiff.so.5

4. create stub libittnotify
   * execstack -c "$CONDA_PREFIX/lib/libtorch_cpu.so"
   * cat > "$CONDA_PREFIX/lib/itt_stub.c" <<'EOF'
     #ifdef __cplusplus
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
     #endif
     EOF
   * gcc -shared -fPIC -O2 -o "$CONDA_PREFIX/lib/libittnotify.so" "$CONDA_PREFIX/lib/itt_stub.c"
   * ls -l "$CONDA_PREFIX/lib/libittnotify.so"

5. set environment variables
   * export LD_PRELOAD=$CONDA_PREFIX/lib/libittnotify.so
   * export TORCH_CUDA_ARCH_LIST="8.6"
   * echo 'export LD_PRELOAD=$CONDA_PREFIX/lib/libittnotify.so${LD_PRELOAD:+:$LD_PRELOAD}' >> ~/.zshrc
   * echo 'export TORCH_CUDA_ARCH_LIST="8.6"' >> ~/.zshrc
   * source ~/.zshrc
   * echo $LD_PRELOAD
   * echo $TORCH_CUDA_ARCH_LIST

6. install environment.yml
   * conda env update -f environment.yml