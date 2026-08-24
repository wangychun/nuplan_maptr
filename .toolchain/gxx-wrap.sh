#!/bin/bash
# mmdet3d 编译链接 wrapper：
# - 编译（无 -shared）：使用 conda gcc9（兼容 torch cu111）
# - 链接（含 -shared）：改用系统 g++-11，规避 conda ld 找不到 /lib64/libm.so.6
CONDA_CXX=/home/xiaoxuan/miniconda3/envs/maptr/bin/x86_64-conda-linux-gnu-c++
SYS_CXX=/usr/bin/g++-11

is_link=false
for a in "$@"; do
  if [[ "$a" == "-shared" ]]; then is_link=true; fi
done

if $is_link; then
  args=()
  skip_next=false
  for a in "$@"; do
    if $skip_next; then skip_next=false; continue; fi
    case "$a" in
      -B) skip_next=true; continue ;;            # 丢弃 -B <path>
      -B*) continue ;;
      --sysroot=*|-Wl,--sysroot=*) continue ;;
      --sysroot|-Wl,--sysroot) skip_next=true; continue ;;
      -Wl,-rpath=/home/xiaoxuan/miniconda3/envs/maptr/lib) args+=("$a") ;;
      *) args+=("$a") ;;
    esac
  done
  exec "$SYS_CXX" "${args[@]}"
else
  exec "$CONDA_CXX" "$@"
fi
