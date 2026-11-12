CC      ?= gcc
UNAME_M := $(shell uname -m)
# PORTABLE=1 builds the plain C code paths (no SIMD intrinsics): slow, for checking them and for new platforms
ifeq ($(PORTABLE),1)
ARCHFLAGS ?= -DTERN_FORCE_SCALAR
BUILD     ?= build-portable
else ifeq ($(AVX2),1)
# the AVX2 code path on a machine that also has AVX-512 (for testing it)
ARCHFLAGS ?= -march=x86-64-v3 -mtune=native
BUILD     ?= build-avx2
else ifeq ($(UNAME_M),aarch64)
ARCHFLAGS ?= -mcpu=native
else
ARCHFLAGS ?= -march=native
endif
CFLAGS  ?= -O3 $(ARCHFLAGS) -fno-strict-aliasing -Wall -Wextra -Wno-unused-parameter -pthread
LDFLAGS ?= -pthread -lm
BUILD   ?= build
SRC     := src/gguf.c src/pool.c src/quant.c src/kernels.c src/tokenizer.c src/model.c src/head.c src/draft.c src/main.c
OBJ     := $(SRC:src/%.c=$(BUILD)/%.o)

all: $(BUILD)/tern

$(BUILD)/tern: $(OBJ)
	$(CC) $(OBJ) -o $@ $(LDFLAGS)

$(BUILD)/%.o: src/%.c src/tern.h | $(BUILD)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD):
	mkdir -p $(BUILD)

# kernel self-test and kernel benchmark (no model needed)
$(BUILD)/kern_test: tests/kern_test.c $(BUILD)/kernels.o $(BUILD)/quant.o $(BUILD)/pool.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)
$(BUILD)/kernbench: bench/kernbench.c $(BUILD)/kernels.o $(BUILD)/quant.o $(BUILD)/pool.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)
$(BUILD)/poolbench: bench/poolbench.c $(BUILD)/pool.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)
$(BUILD)/burst: bench/burst.c $(BUILD)/pool.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)
$(BUILD)/membw: bench/membw_x86.c | $(BUILD)
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

# the small measuring tools of bench/ (x86-64): memory bandwidth, kernel speed, thread-pool cost, reads in short regions
benchtools: $(BUILD)/membw $(BUILD)/kernbench $(BUILD)/poolbench $(BUILD)/burst

test: $(BUILD)/kern_test
	$(BUILD)/kern_test

# no credentials, addresses or logins in the files of the repository
secrets:
	python3 tests/check_no_secrets.py

clean:
	rm -rf $(BUILD)

.PHONY: all clean test secrets benchtools
