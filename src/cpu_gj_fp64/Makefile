cpu:
	icpx -O3 -fsycl *.c -o 'GSLAM.cpu'

gpu:
	icpx -O3 -fsycl *.c -o 'GSLAM.gpu'
    
s10:
	icpx -O3 -fintelfpga -fsycl -Xshardware -Xsboard=intel_s10sx_pac:pac_s10 -Xsdsp-mode=prefer-dsp -Xsread-only-cache-size=32768 *.c -o 'GSLAM.s10'
 
a10:
	icpx -O3 -fintelfpga -fsycl -Xshardware -Xsboard=intel_a10gx_pac:pac_a10 -Xsdsp-mode=prefer-dsp -Xsread-only-cache-size=32768 *.c -o 'GSLAM.a10'
    
emu:
	icpx -O3 -fintelfpga -fsycl *.c -DFPGA_EMULATOR -o 'GSLAM.emu'

x86:
	gcc -O3 *.c -lm -o 'GSLAM'

.PHONY: clean

clean:
	rm GSLAM
