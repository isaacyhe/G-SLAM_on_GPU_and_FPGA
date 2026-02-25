#!/bin/bash

source /opt/intel/inteloneapi/setvars.sh
icpx -O3 -fintelfpga -fsycl -Xshardware -Xsboard=intel_a10gx_pac:pac_a10 -Xsdsp-mode=prefer-dsp -Xsread-only-cache-size=12288 *.c -o 'GSLAM.a10'
#icpx -O3 -fintelfpga -fsycl -Xshardware -Xsboard=intel_s10sx_pac:pac_s10 -Xsdsp-mode=prefer-dsp -Xsread-only-cache-size=32768 *.c -o 'GSLAM.s10'
