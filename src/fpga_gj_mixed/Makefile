cpu_test:
	icpx -O3 -fsycl *.c -lm -o GSLAM.cpu_test

gpu_test:
	icpx -O3 -fsycl -DUSE_GPU *.c -lm -o GSLAM.gpu_test

s10:
	icpx -O3 -fintelfpga -fsycl -DFPGA_HARDWARE -Xshardware \
		-Xsboard=intel_s10sx_pac:pac_s10 -Xsdsp-mode=prefer-dsp \
		-Xsread-only-cache-size=32768 *.c -o GSLAM.s10

a10:
	icpx -O3 -fintelfpga -fsycl -DFPGA_HARDWARE -Xshardware \
		-Xsboard=intel_a10gx_pac:pac_a10 -Xsdsp-mode=prefer-dsp \
		-Xsread-only-cache-size=32768 *.c -o GSLAM.a10

emu:
	icpx -O3 -fintelfpga -fsycl -DFPGA_EMULATOR *.c -o GSLAM.emu

x86:
	gcc -O3 *.c -lm -o GSLAM

.PHONY: clean

clean:
	rm -f GSLAM GSLAM.cpu_test GSLAM.gpu_test GSLAM.emu GSLAM.s10 GSLAM.a10
